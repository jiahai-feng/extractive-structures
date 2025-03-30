import re
from functools import partial
import torch
import einops
import exults.tok_utils as tu


def find_index_of(tokens, search_tokens):
    for i, x in list(enumerate(tokens))[::-1]:
        if x.item() == search_tokens[0] and i + len(search_tokens) <= len(tokens):
            if all(tok == tokens[i + t].item() for t, tok in enumerate(search_tokens)):
                return i
    return -1


def stack_named_parameters(named_parameters):
    """
    Splits named_parameters into stacked and unique.
    Stacked parameters are ones that are repeated across layers.
    We call the part of the name after the layer number the weight_key.
        e.g. "model.layers.0.{weight_key}"

    Args
        named_parameters: List[Tuple[str, torch.Tensor]]
            - List of (name, param)
    Returns
        stacked: Dict[str, Dict[int, Tuple[str, torch.Tensor]]]
            - maps weight_key -> layer_num -> Tuple of (name, param)
            - Dict of Dicts of Tuples
            - Dict of weight_key -> Dict of layer_num -> Tuple of (name, param)
        unique: Dict[str, Tuple[str, torch.Tensor]]
    """
    stacked = {}
    unique = {}
    for name, param in named_parameters:
        if "layers" in name or "blocks" in name:
            if "layers" in name:
                match = re.match(r"(.+)\.layers\.(\d+)\.(.+)", name)
            else:
                match = re.match(r"(.+)\.blocks\.(\d+)\.(.+)", name)
            layer_num = int(match.group(2))
            param_name = match.group(3)
            if param_name in stacked:
                stacked[param_name][layer_num] = (name, param)
            else:
                stacked[param_name] = {layer_num: (name, param)}
        else:
            unique[name] = (name, param)
    return stacked, unique


def flatten_parameters(named_parameters):
    """
    Returns only the stacked component from stack_named_parameters, but flattened.

    Returns:
        List[Dict[str, Any]]
            - List of dicts with keys:
                - weight_key: str
                - value: torch.Tensor
                - layer: int
                - param: torch.Tensor
    """
    stacked, unique = stack_named_parameters(named_parameters)
    return [
        {
            "weight_key": weight_key,
            "value": value,
            "layer": layer,
            "name": name,
        }
        for weight_key, layer_values in stacked.items()
        for layer, (name, value) in layer_values.items()
    ]


def tokenize_and_mask_batch(batch, tokenizer):
    """
    Args:
        batch: List[Tuple[text: str, target: str]]
    Returns:
        inputs: Dict[str, torch.Tensor]
            - Contains at least input_ids, attention_mask, labels
            - Feed into model forward pass
            - labels are masked out to only include the target
    """
    strings = []
    refs = []
    for text, target in batch:
        string, ref = tu.TrackFormatter().format(
            "{text} {target}", text=text, target=target
        )
        strings.append(string)
        refs.append(ref)
    inputs = tokenizer(
        strings, return_tensors="pt", padding=True, return_offsets_mapping=True
    )
    labels = inputs["input_ids"].clone()
    for i, ref in enumerate(refs):
        token_spans = tu.get_token_spans(
            ref, inputs["offset_mapping"][i], inputs["attention_mask"][i]
        )
        target_start, target_end = token_spans["target"][0]
        labels[i, :target_start] = -100
        labels[i, target_end:] = -100
        if (
            tokenizer.decode(labels[i, target_start:target_end]).strip()
            != batch[i][1].strip()
        ):
            print("Tokenizer Error: target mismatch")
            print(tokenizer.decode(labels[i, target_start:target_end]))
            print(batch[i][1])
    inputs["labels"] = labels
    del inputs["offset_mapping"]
    return inputs


def dmap(f, *lses, with_key=False):
    dicts = [dict(ls) for ls in lses]
    if with_key:
        return [(key, f(*[d[key] for d in dicts], key=key)) for key in dicts[0]]
    else:
        return [(key, f(*[d[key] for d in dicts])) for key in dicts[0]]


def flatten_stacked(stacked, d={}):
    """
    Flattens the stacked dict from stack_named_parameters into a list.
    Optionally labels every dict with labels from a dict d.

    Returns:
        List[Dict[str, Any]]
            - List of dicts with keys:
                - weight_key: str
                - value: torch.Tensor
                - layer: int
                - param: torch.Tensor
    """
    return [
        {"weight_key": weight_key, "value": value, "layer": layer, "name": name, **d}
        for weight_key, layer_values in stacked.items()
        for layer, (name, value) in layer_values.items()
    ]


@torch.enable_grad()
def get_gradients(inputs, model, tokenizer, *, with_loss=False, reduction="mean"):
    """
    Args:
        inputs: List[Tuple[text: str, target: str]]

    Returns:
        Gradients: List[Tuple[str, torch.Tensor]]
    """
    if isinstance(inputs[0], str):
        inputs = [inputs]
    model.eval()
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.grad = None
    inputs = tokenize_and_mask_batch(inputs, tokenizer)
    loss = model.get_loss(**inputs, reduction=reduction)
    # print(loss)
    loss.backward()
    gradients = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            gradients.append((name, param.grad))
            param.grad = None
    if with_loss:
        return gradients, loss
    return gradients


@torch.no_grad()
def get_pc_gradients(gradients, gradient_mode, adam_denoms=None):
    if gradient_mode == "sgd":
        return gradients
    elif gradient_mode == "signgd":
        return [(name, torch.sign(grad)) for name, grad in gradients]
    elif gradient_mode == "adamgd":
        return [
            (name, grad / adam_denoms[name].to(grad.device)) for name, grad in gradients
        ]
    else:
        raise ValueError(f"Unknown gradient_mode {gradient_mode}")


@torch.no_grad()
def get_pc_gradients_(gradients, gradient_mode, adam_denoms=None):
    if gradient_mode == "sgd":
        return gradients
    elif gradient_mode == "signgd":
        return [(name, grad.sign_()) for name, grad in gradients]
    elif gradient_mode == "adamgd":
        return [
            (name, grad.div_(adam_denoms[name].to(grad.device)))
            for name, grad in gradients
        ]
    else:
        raise ValueError(f"Unknown gradient_mode {gradient_mode}")


def gradient_alignment(model, input_pair, tokenizer, adam_denoms=None):
    zip_map = partial(tu.zip_map, dtype=torch.Tensor)
    grads = [dict(get_gradients(s, model, tokenizer)) for s in input_pair]
    with torch.no_grad():
        cosine_sim = zip_map(
            lambda a, b: torch.dot(a.flatten(), b.flatten()) / a.norm() / b.norm(),
            grads[0],
            grads[1],
        )
        a_norm = zip_map(lambda a: a.norm(), grads[0])
        b_norm = zip_map(lambda a: a.norm(), grads[1])
        sign_gd = zip_map(  # a is the update, b is the test point
            lambda a, b: (a.flatten().sign() * b.flatten()).sum(), grads[0], grads[1]
        )
        eps = 1e-5  # that's the OLMo value
        eps_sign_gd = zip_map(  # a is the update, b is the test point
            lambda a, b: ((a / (a.abs() + eps)).flatten() * b.flatten()).sum(),
            grads[0],
            grads[1],
        )
        sgd = zip_map(  # a is the update, b is the test point
            lambda a, b: (a.flatten() * b.flatten()).sum(), grads[0], grads[1]
        )
        if adam_denoms is not None:
            adamgd = {
                name: torch.dot(
                    (
                        grads[0][name] / adam_denoms[name].to(grads[0][name].device)
                    ).flatten(),
                    grads[1][name].flatten(),
                )
                for name in grads[0]
            }
    # print(cosine_sim.keys())
    if adam_denoms is None:
        return stack_named_parameters(
            zip_map(
                lambda cos, a, b, sign_gd, eps_sign_gd, sgd: {
                    "cosine": cos,
                    "a_norm": a,
                    "b_norm": b,
                    "sign_gd": sign_gd,
                    "eps_sign_gd": eps_sign_gd,
                    "sgd": sgd,
                },
                cosine_sim,
                a_norm,
                b_norm,
                sign_gd,
                eps_sign_gd,
                sgd,
            ).items()
        )
    else:
        return stack_named_parameters(
            zip_map(
                lambda cos, a, b, sign_gd, eps_sign_gd, sgd, adamgd: {
                    "cosine": cos,
                    "a_norm": a,
                    "b_norm": b,
                    "sign_gd": sign_gd,
                    "eps_sign_gd": eps_sign_gd,
                    "sgd": sgd,
                    "adamgd": adamgd,
                },
                cosine_sim,
                a_norm,
                b_norm,
                sign_gd,
                eps_sign_gd,
                sgd,
                adamgd,
            ).items()
        )


# breakdown gradients
# be careful, this code works only for olmo models

from exults.hook_utils import hook_model, hook_model_bwd
import torch.nn.functional as F


def produce_breakdown_hooks_mlp(module_name, delta):
    cache = []
    scores = []

    def forward_hook(inputs, hook):
        assert len(cache) == 0
        cache.append(inputs.clone())

    def backward_hook(grad_output, hook):
        assert len(cache) == 1
        # [batch, seq, dim]
        inp = cache.pop()
        assert len(inp.shape) == 3
        assert len(grad_output.shape) == 3
        scores.append((F.linear(inp, delta) * grad_output).sum(dim=-1))

    return (module_name, forward_hook), (module_name, backward_hook), scores


def produce_breakdown_hooks_att_proj(module_name, delta, model):
    cache = []
    scores = []

    def forward_hook(inputs, hook):
        assert len(cache) == 0
        cache.append(inputs.clone().detach())

    def backward_hook(grad_output, hook):
        assert len(cache) == 1
        inp = cache.pop()
        assert len(inp.shape) == 3  # [batch, seq, dim]
        assert (
            len(grad_output.shape) == 3
        )  # [batch, seq, out_dim], where out_dim = 3 * n_heads * head_dim
        res = F.linear(inp, delta)  # [batch, seq, out_dim]
        res = einops.rearrange(
            res,
            "batch seq (k n_heads head_dim) -> batch seq k n_heads head_dim",
            k=3,
            n_heads=model.config.n_heads,
        )
        grad_output = einops.rearrange(
            grad_output,
            "batch seq (k n_heads head_dim) -> batch seq k n_heads head_dim",
            k=3,
            n_heads=model.config.n_heads,
        )
        scores.append(
            einops.einsum(
                res,
                grad_output,
                "batch seq k n_heads head_dim, batch seq k n_heads head_dim -> batch seq n_heads",
            )
        )

    return (module_name, forward_hook), (module_name, backward_hook), scores


def produce_breakdown_hooks_attn_out(module_name, delta, model):
    cache = []
    scores = []

    def forward_hook(inputs, hook):
        assert len(cache) == 0
        cache.append(inputs.clone().detach())

    def backward_hook(grad_output, hook):
        assert len(cache) == 1
        inp = cache.pop()
        assert len(inp.shape) == 3  # [batch, seq, dim], where dim = n_heads * head_dim
        assert len(grad_output.shape) == 3  # [batch, seq, dim]
        res = F.linear(grad_output, delta.T)  # [batch, seq, dim]
        res = einops.rearrange(
            res,
            "batch seq (n_heads head_dim) -> batch seq n_heads head_dim",
            n_heads=model.config.n_heads,
        )
        inp = einops.rearrange(
            inp,
            "batch seq (n_heads head_dim) -> batch seq n_heads head_dim",
            n_heads=model.config.n_heads,
        )
        scores.append(
            einops.einsum(
                res,
                inp,
                "batch seq n_heads head_dim, batch seq n_heads head_dim -> batch seq n_heads",
            )
        )

    return (module_name, forward_hook), (module_name, backward_hook), scores


def breakdown_gradients(model, gradients, inputs, loss_fn=None):
    forward_hooks = []
    backward_hooks = []
    scores = {}
    gradients_dict = dict(gradients)
    for i in range(32):
        fwd, bwd, sc = produce_breakdown_hooks_mlp(
            model.Path("ff_proj", i).full_key,
            gradients_dict[model.Path("ff_proj.weight", i).full_key],
        )
        forward_hooks.append(fwd)
        backward_hooks.append(bwd)
        scores[model.Path("ff_proj", i).full_key] = sc
        fwd, bwd, sc = produce_breakdown_hooks_mlp(
            model.Path("ff_out", i).full_key,
            gradients_dict[model.Path("ff_out.weight", i).full_key],
        )
        forward_hooks.append(fwd)
        backward_hooks.append(bwd)
        scores[model.Path("ff_out", i).full_key] = sc
        fwd, bwd, sc = produce_breakdown_hooks_att_proj(
            model.Path("att_proj", i).full_key,
            gradients_dict[model.Path("att_proj.weight", i).full_key],
            model,
        )
        forward_hooks.append(fwd)
        backward_hooks.append(bwd)
        scores[model.Path("att_proj", i).full_key] = sc
        fwd, bwd, sc = produce_breakdown_hooks_attn_out(
            model.Path("attn_out", i).full_key,
            gradients_dict[model.Path("attn_out.weight", i).full_key],
            model,
        )
        forward_hooks.append(fwd)
        backward_hooks.append(bwd)
        scores[model.Path("attn_out", i).full_key] = sc
    with hook_model(model, forward_hooks):
        with hook_model_bwd(model, backward_hooks):
            if loss_fn is not None:
                loss = loss_fn()
            else:
                loss = model.get_loss(**inputs)
            loss.backward()
    return (
        [
            (
                model.Path("ff_out.weight", i).full_key,
                scores[model.Path("ff_out", i).full_key][0],
            )
            for i in range(32)
        ]
        + [
            (
                model.Path("ff_proj.weight", i).full_key,
                scores[model.Path("ff_out", i).full_key][0],
            )
            for i in range(32)
        ]
        + [
            (
                model.Path("att_proj.weight", i).full_key,
                scores[model.Path("att_proj", i).full_key][0],
            )
            for i in range(32)
        ]
        + [
            (
                model.Path("attn_out.weight", i).full_key,
                scores[model.Path("attn_out", i).full_key][0],
            )
            for i in range(32)
        ]
    )
