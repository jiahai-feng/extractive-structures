from types import SimpleNamespace
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
import torch
import einops

import extractive_structures.gradients as pg
from extractive_structures.masking_utils import update_model
from exults.hook_utils import hook_model, hook_model_bwd
import exults.tok_utils as tu


@dataclass
class ComponentData:
    mlps: torch.Tensor
    attn: torch.Tensor

    def __sub__(self, other):
        return ComponentData(self.mlps - other.mlps, self.attn - other.attn)

    def mul(self, other):
        if isinstance(other, ComponentData):
            return ComponentData(self.mlps * other.mlps, self.attn * other.attn)
        else:
            return ComponentData(self.mlps * other, self.attn * other)

    def __mul__(self, other):
        # this really is a dot product
        return ComponentData(
            (self.mlps * other.mlps).sum(dim=-1), (self.attn * other.attn).sum(dim=-1)
        )

    def __add__(self, other):
        return ComponentData(self.mlps + other.mlps, self.attn + other.attn)

    def __iadd__(self, other):
        self.mlps += other.mlps
        self.attn += other.attn
        return self

    def map(self, func):
        return ComponentData(func(self.mlps), func(self.attn))


@torch.enable_grad()
def cache_forward_backward(
    model, loss_fn, pre_cache=None, compute_pre_cache=False, strict_upstream=False
):
    """
    Returns:
        component_outputs: ComponentData[
            mlps : Tensor[batch, seq, layer, act]
            attn : Tensor[batch, seq, layer, head, act]
        ]
        forward_pre_cache: Dict[name: str, FloatTensor[batch, seq, act]]
    """

    forward_cache = {}

    def forward_hook(act_in, act_out, hook):
        assert hook.name not in forward_cache
        forward_cache[hook.name] = act_out.clone().detach()

    forward_hooks = [
        (model.Path("ff_out", layer).full_key, forward_hook) for layer in range(32)
    ]
    forward_hooks += [
        (model.Path("hook_z", layer).full_key, forward_hook) for layer in range(32)
    ]

    cache_forward_pre_hooks = []
    sub_forward_pre_hooks = []
    if compute_pre_cache:
        forward_pre_cache = {}

        def cache_forward_pre_hook(act_in, hook):
            assert hook.name not in forward_pre_cache
            forward_pre_cache[hook.name] = act_in.clone().detach()

        cache_forward_pre_hooks += [
            (model.Path("ff_proj", layer).full_key, cache_forward_pre_hook)
            for layer in range(32)
        ]
        cache_forward_pre_hooks += [
            (model.Path("att_proj", layer).full_key, cache_forward_pre_hook)
            for layer in range(32)
        ]
    if pre_cache is not None:

        def sub_forward_pre_hook(act_in, hook):
            return pre_cache[hook.name]

        sub_forward_pre_hooks += [
            (model.Path("ff_proj", layer).full_key, sub_forward_pre_hook)
            for layer in range(32)
        ]
        sub_forward_pre_hooks += [
            (model.Path("att_proj", layer).full_key, sub_forward_pre_hook)
            for layer in range(32)
        ]

    backward_cache = {}

    def backward_hook(grad_out, hook):
        assert hook.name not in backward_cache
        backward_cache[hook.name] = grad_out.clone().detach()

    backward_hooks = [
        (model.Path("ff_out", layer).full_key, backward_hook) for layer in range(32)
    ]
    backward_hooks += [
        (model.Path("hook_z", layer).full_key, backward_hook) for layer in range(32)
    ]

    with hook_model(model, forward_hooks, pre_hook=False):
        with hook_model(
            model, cache_forward_pre_hooks + sub_forward_pre_hooks, pre_hook=True
        ):
            with hook_model_bwd(model, backward_hooks):
                loss_fn().backward()

    if strict_upstream:
        with torch.no_grad():
            for layer in range(32):
                backward_cache[model.Path("ff_out", layer).full_key].sub_(
                    backward_cache[model.Path("ff_out", 31).full_key].to(
                        backward_cache[model.Path("ff_out", layer).full_key].device
                    )
                )
                backward_cache[model.Path("hook_z", layer).full_key].sub_(
                    backward_cache[model.Path("ff_out", 31).full_key].to(
                        model.Path("attn_out.weight", layer).traverse(model).device
                    )
                    @ model.Path("attn_out.weight", layer).traverse(model)
                )

    # consolidate cache
    def parse_cache(cache):
        mlps = einops.rearrange(
            torch.stack(
                [
                    cache[model.Path("ff_out", layer).full_key].cuda()
                    for layer in range(32)
                ]
            ),
            "layer batch seq act -> batch seq layer act",
        )
        attn = einops.rearrange(
            torch.stack(
                [
                    cache[model.Path("hook_z", layer).full_key].cuda()
                    for layer in range(32)
                ]
            ),
            "layer batch seq (n_heads d_head) -> batch seq layer n_heads d_head",
            n_heads=model.config.n_heads,
        )
        return ComponentData(mlps, attn)

    parsed_forward_cache = parse_cache(forward_cache)
    parsed_backward_cache = parse_cache(backward_cache)

    if compute_pre_cache:
        return parsed_forward_cache, parsed_backward_cache, forward_pre_cache
    else:
        return parsed_forward_cache, parsed_backward_cache


def default_loss_fn(model, tokenizer, test_point):
    inputs = pg.tokenize_and_mask_batch([test_point], tokenizer)
    return model.get_loss(**inputs)


def compute_cf_alignment(tokenizer, test_point, cf_point):
    with tu.set_padding_side(tokenizer, "right"):
        inputs = pg.tokenize_and_mask_batch([test_point], tokenizer)
        cf_inputs = pg.tokenize_and_mask_batch([cf_point], tokenizer)
    # identify matching prefix and suffix
    input_end = (
        inputs["labels"][0] != -100
    ).long().argmax().item() - 1  # the last qn token
    cf_input_end = (cf_inputs["labels"][0] != -100).long().argmax().item() - 1

    def first_divergence(a, b):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                return i
        raise ValueError("Inputs are identical")

    prefix_len = first_divergence(inputs["input_ids"][0], cf_inputs["input_ids"][0])
    suffix_len = first_divergence(
        inputs["input_ids"].cpu().numpy()[0, input_end::-1],
        cf_inputs["input_ids"].cpu().numpy()[0, cf_input_end::-1],
    )
    center_len = input_end - prefix_len - suffix_len + 1
    cf_center_len = cf_input_end - prefix_len - suffix_len + 1
    prefix_alignment = list(range(prefix_len))
    suffix_alignment = list(range(cf_input_end - suffix_len + 1, cf_input_end + 1))
    if center_len <= cf_center_len:
        center_alignment = list(
            range(cf_center_len + prefix_len - center_len, cf_center_len + prefix_len)
        )
    else:
        center_alignment = (center_len - cf_center_len) * [prefix_len] + list(
            range(prefix_len, cf_center_len + prefix_len)
        )
    answer_alignment = (len(inputs["input_ids"][0]) - (input_end + 1)) * [
        cf_input_end
    ]  # just gibberish would do.
    # print(prefix_alignment, center_alignment, suffix_alignment)
    return torch.tensor(
        prefix_alignment + center_alignment + suffix_alignment + answer_alignment
    )


def compute_extractive_scores(
    *,
    model,
    delta,
    loss_fn=None,
    tokenizer=None,
    test_point=None,
    strict_downstream=False,
    strict_upstream=False,
    cf_loss_fn=None,
    cf_alignment=None,
):
    """
    Args:
        model:
        delta: [(name, tensor)] - delta of model weights
        loss_fn: Optional[Callable] - Calls the forward pass of the model, returns the target for backward pass
        tokenizer: Optional - Tokenizer for the model, used to construct loss_fn if not given
        test_point: Optional[tuple[str, str]] - (input, completion), used to construct loss_fn if not given
        strict_downstream: bool
        strict_upstream: bool
        cf_loss_fn: Optional[Callable] - loss_fn for counterfactual input
        cf_alignment: Optional[LongTensor[seq]] - maps original tokens to counterfactual tokens

    Returns:
        upstream_score: ComponentData
        downstream_score: ComponentData


    strict_downstream=True:
        fixes the problem where we want the downstream score of a component to be
        computed with the component having old weights, but with inputs from new weights.
        This should be True by default.

    strict_upstream=True:
        uses the old paradigm of dropping the direct contribution of the component
        to the unembedding, because this contribution is not mediated by changes in weights.
        This should be False by default.

    counterfactual_upstream:
        collect the o_fwd on counterfactual input, and hit it with n_fwd.
        subtlety: we need to match the tokens in cf_o_fwd with o_fwd
        inputs: cf_loss_fn, cf_alignment (for every input, get the output)
    """
    if loss_fn is None:
        loss_fn = partial(default_loss_fn, model, tokenizer, test_point)

    if not strict_downstream:
        o_fwd, o_bwd = cache_forward_backward(model, loss_fn)

        with update_model(model, delta):
            n_fwd, n_bwd = cache_forward_backward(model, loss_fn)

    else:
        o_fwd, o_bwd = cache_forward_backward(model, loss_fn)
        with update_model(model, delta):
            n_fwd_prime, n_bwd, pre_cache = cache_forward_backward(
                model, loss_fn, compute_pre_cache=True
            )
        n_fwd, _ = cache_forward_backward(model, loss_fn, pre_cache=pre_cache)

    if cf_loss_fn is not None:
        cf_o_fwd_unmapped, _ = cache_forward_backward(model, cf_loss_fn)
        assert cf_alignment.shape == (o_fwd.mlps.shape[1],)  # batch, seq, layer, act
        cf_o_fwd = ComponentData(
            cf_o_fwd_unmapped.mlps[:, cf_alignment],
            cf_o_fwd_unmapped.attn[:, cf_alignment],
        )
        assert cf_o_fwd.mlps.shape == o_fwd.mlps.shape
        assert cf_o_fwd.attn.shape == o_fwd.attn.shape
    else:
        cf_o_fwd = ComponentData(
            torch.zeros_like(o_fwd.mlps), torch.zeros_like(o_fwd.attn)
        )

    upstream_score = (o_bwd - n_bwd) * (o_fwd - cf_o_fwd)
    downstream_score = (o_fwd - n_fwd) * o_bwd

    if strict_upstream:
        _, strict_o_bwd = cache_forward_backward(model, loss_fn, strict_upstream=True)
        with update_model(model, delta):
            _, strict_n_bwd = cache_forward_backward(
                model, loss_fn, strict_upstream=True
            )

        upstream_score = (strict_o_bwd - strict_n_bwd) * (o_fwd - cf_o_fwd)
    return upstream_score, downstream_score


def convert_format(model, ld, negate=True):
    # we also negate the breakdown
    # initially, negative scores are good - signifies we lower loss
    # negate so that positive means high impact

    coeffs = [1, -1]
    ld = dict(ld)
    stacked = {}

    for k in ["ff_out", "ff_proj", "attn_out", "att_proj"]:
        stacked[k] = torch.stack(
            [ld[model.Path(k + ".weight", layer).full_key].cpu() for layer in range(32)]
        ).movedim(0, 2)
    return ComponentData(
        coeffs[negate] * (stacked["ff_out"] + stacked["ff_proj"]),  # batch  seq  layer
        coeffs[negate]
        * (stacked["attn_out"] + stacked["att_proj"]),  # batch  seq  layer  head
    )


def get_informative_scores(
    *, model, delta, loss_fn=None, tokenizer=None, test_point=None
):
    if loss_fn is None:

        def loss_fn():
            inputs = pg.tokenize_and_mask_batch([test_point], tokenizer)
            return model.get_loss(**inputs)

    breakdown = pg.breakdown_gradients(model, delta, inputs=None, loss_fn=loss_fn)
    breakdown_comp = convert_format(model, breakdown, negate=True)
    return breakdown_comp


from extractive_structures.datasets import Dataset, Options
from extractive_structures.utils import mean_logit_loss
from tqdm import tqdm
import numpy as np


def compute_extractive_scores_counterfactual(
    model,
    tokenizer,
    test_dataset: list[tuple[str, str]],
    test_options: Options,
    delta,
    acc_point: tuple[str, str],
):
    acc_upstream, acc_downstream = compute_extractive_scores(
        model=model,
        delta=delta,
        loss_fn=partial(
            default_loss_fn, model=model, test_point=acc_point, tokenizer=tokenizer
        ),
        strict_downstream=True,
        strict_upstream=False,
    )
    acc_informative = get_informative_scores(
        model=model,
        delta=delta,
        loss_fn=partial(
            default_loss_fn, model=model, test_point=acc_point, tokenizer=tokenizer
        ),
    )
    for x in [acc_upstream, acc_downstream, acc_informative]:
        x.mlps.zero_()
        x.attn.zero_()

    rng = np.random.default_rng(3)

    for test_point in tqdm(test_dataset):
        for trial in range(5):
            cf_test_point = rng.choice(test_dataset)
            while (
                cf_test_point[0] == test_point[0] or cf_test_point[1] == test_point[1]
            ):
                cf_test_point = rng.choice(test_dataset)
            upstream_scores, downstream_scores = compute_extractive_scores(
                model=model,
                delta=delta,
                loss_fn=partial(
                    mean_logit_loss,
                    model=model,
                    test_point=test_point,
                    tokenizer=tokenizer,
                    options=test_options,
                ),
                strict_downstream=True,
                strict_upstream=False,
                cf_loss_fn=partial(
                    default_loss_fn,
                    model=model,
                    test_point=cf_test_point,
                    tokenizer=tokenizer,
                ),
                cf_alignment=compute_cf_alignment(tokenizer, test_point, cf_test_point),
            )
            informative_scores = get_informative_scores(
                model=model,
                delta=delta,
                loss_fn=partial(
                    mean_logit_loss,
                    model=model,
                    test_point=test_point,
                    tokenizer=tokenizer,
                    options=test_options,
                ),
            )
            acc_alignment = compute_cf_alignment(tokenizer, acc_point, test_point)
            acc_upstream += upstream_scores.map(lambda x: x[:, acc_alignment])
            acc_downstream += downstream_scores.map(lambda x: x[:, acc_alignment])
            acc_informative += informative_scores.map(lambda x: x[:, acc_alignment])
    for x in [acc_upstream, acc_downstream, acc_informative]:
        x.mlps.div_(len(test_dataset) * 5)
        x.attn.div_(len(test_dataset) * 5)
    return [acc_upstream, acc_downstream, acc_informative]
