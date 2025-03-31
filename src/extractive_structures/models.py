from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn as nn
import os
from olmo.config import TrainConfig
from olmo.model import OLMo
from olmo.checkpoint import load_state_dict

from transformers import AutoModelForCausalLM, AutoTokenizer
import math

from accelerate import dispatch_model, infer_auto_device_map
from accelerate.utils.modeling import get_max_memory, get_balanced_memory


olmo_preanneal_checkpoint = Path(os.getenv("OLMO_PREANNEAL_DIR"))


from dataclasses import dataclass
from typing import Optional, Dict, ClassVar


@dataclass
class WeightPath:
    """
    For stacked weights, the full_key is broken down into:
    <prefix>.<layer>.<weight_key>

    For unstacked weights, the full_key is the same as weight_key
    """

    weight_key: str
    layer: Optional[int] = None

    mlp_name: ClassVar[str] = "ff"
    attn_name: ClassVar[str] = "att"
    prefix: ClassVar[str] = "transformer.blocks"

    @property
    def full_key(self):
        if self.is_stacked:
            if self.weight_key:
                return f"{self.prefix}.{self.layer}.{self.weight_key}"
            else:
                return f"{self.prefix}.{self.layer}"
        else:
            return self.weight_key

    @classmethod
    def from_str(cls, s):
        if s.startswith(cls.prefix):
            layer, weight_key = s.removeprefix(cls.prefix + ".").split(".", 1)
            return cls(
                layer=int(layer),
                weight_key=weight_key,
            )
        else:
            return cls(
                weight_key=s,
            )

    def __post_init__(self):
        if self.is_stacked:
            assert self.layer is not None
            # assert sum([self.is_mlp, self.is_attn, self.weight_key == '']) == 1

    def to_str(self):
        return self.full_key

    @property
    def is_stacked(self):
        return self.layer is not None

    @property
    def is_mlp(self):
        return self.is_stacked and self.mlp_name in self.weight_key

    @property
    def is_attn(self):
        return self.is_stacked and self.attn_name in self.weight_key

    def __str__(self):
        return self.to_str()

    def __repr__(self):
        if self.is_stacked:
            return f"{self.__class__.__name__}({self.layer}, {self.weight_key})"
        else:
            return f"{self.__class__.__name__}({self.full_key})"

    def traverse(self, model):
        if self.full_key == "":
            return model
        components = self.full_key.split(".")
        obj = model
        for comp in components:
            if comp.isnumeric():
                obj = obj[int(comp)]
            else:
                obj = getattr(obj, comp)
        return obj

    def parent(self):
        return self.__class__.from_str(".".join(self.full_key.split(".")[:-1]))

    def tail(self):
        return self.full_key.split(".")[-1]


class OlmoWeightPath(WeightPath):
    mlp_name: ClassVar[str] = "ff"
    attn_name: ClassVar[str] = "att"
    prefix: ClassVar[str] = "transformer.blocks"
    mlp_names = ["ff_proj.weight", "ff_out.weight"]
    attn_names = ["att_proj.weight", "attn_out.weight"]

    @classmethod
    def iter_names(cls):
        names = [
            "ff_proj.weight",
            "ff_out.weight",
            "att_proj.weight",
            "attn_out.weight",
        ]
        for name in names:
            for layer in range(32):
                yield cls(name, layer)


class LlamaWeightPath(WeightPath):
    mlp_name: ClassVar[str] = "mlp"
    attn_name: ClassVar[str] = "attn"
    prefix: ClassVar[str] = "model.layers"
    mlp_names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    attn_names = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ]


class GemmaWeightPath(WeightPath):
    mlp_name: ClassVar[str] = "mlp"
    attn_name: ClassVar[str] = "attn"
    prefix: ClassVar[str] = "model.layers"
    mlp_names = ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"]
    attn_names = [
        "self_attn.q_proj.weight",
        "self_attn.k_proj.weight",
        "self_attn.v_proj.weight",
        "self_attn.o_proj.weight",
    ]


def get_olmo_model(device="auto"):
    olmo_checkpoint_dir = olmo_preanneal_checkpoint
    tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1.7-7B-hf")
    cfg = TrainConfig.load(olmo_checkpoint_dir / "config.yaml")
    model = OLMo(cfg.model)
    model.load_state_dict(
        torch.load(str(olmo_checkpoint_dir / "model.pt"), mmap=True), assign=True
    )
    if device == "auto":
        dm = infer_auto_device_map(
            model,
            verbose=False,
            max_memory=get_balanced_memory(
                model, no_split_module_classes=["OLMoSequentialBlock"]
            ),
            no_split_module_classes=["OLMoSequentialBlock"],
        )
        dispatch_model(model, device_map=dm)
    else:
        model.to(device)

    def get_loss(input_ids, labels, reduction="mean", **kwargs):
        # by default, the loss is averaged across all tokens in the batch
        if device != "auto":
            input_ids = input_ids.to(model.device)
            if "attention_mask" in kwargs:
                kwargs["attention_mask"] = kwargs["attention_mask"].to(model.device)
        outputs = model(input_ids=input_ids, **kwargs)
        logits = outputs.logits
        # from HF's modeling llama
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if reduction == "none":
            return loss.view(labels.size(0), -1)
        return loss

    model.get_loss = get_loss
    model.mlp_name = "ff"
    model.attn_name = "att"
    model.layer_keys = [
        f"transformer.blocks.{i}." for i in range(len(model.transformer.blocks))
    ]
    model.parse_layer = lambda name: (
        None if "blocks" not in name else int(name.split(".")[2])
    )
    model.Path = OlmoWeightPath

    def logit_lens(token):
        assert (
            not model.transformer.ln_f.config.include_bias
            and model.transformer.ln_f.config.layer_norm_type == "default"
        )
        assert (
            model.transformer.ln_f.weight is None
            and model.transformer.ln_f.bias is None
        )
        return model.transformer.ff_out.weight[token]

    model.logit_lens = logit_lens
    model.uconfig = model.config
    return model, tokenizer


def get_masked_params(model, mask_type, masked_layers):
    if mask_type == "all":
        return [name for name, _ in model.named_parameters()]
    elif mask_type == "mlp":
        return [
            name
            for name, p in model.named_parameters()
            if model.mlp_name in name and any(lk in name for lk in model.layer_keys)
        ]
    elif mask_type == "some_mlp":
        return [
            name
            for name, param in model.named_parameters()
            if model.mlp_name in name
            and any(model.layer_keys[i] in name for i in masked_layers)
        ]
    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")


def clean_optimizer_state(optim_state_dict):
    if isinstance(optim_state_dict["param_groups"][0]["params"][0], int):
        id_to_fqn: Dict[int, str] = {}
        for group in optim_state_dict["param_groups"]:
            new_param_names = []
            for fqn, id in zip(group["param_names"], group["params"]):
                fqn = fqn.replace("_fsdp_wrapped_module.", "")
                id_to_fqn[id] = fqn
                new_param_names.append(fqn)
            group["param_names"] = new_param_names
            group["params"] = new_param_names
        for id in list(optim_state_dict["state"].keys()):
            optim_state_dict["state"][id_to_fqn[id]] = optim_state_dict["state"].pop(id)
    else:
        # Otherwise we still want to clean up the param names to remove the "_fsdp_wrapped_module." prefix.
        for group in optim_state_dict["param_groups"]:
            group["param_names"] = [
                fqn.replace("_fsdp_wrapped_module.", "") for fqn in group["param_names"]
            ]
            group["params"] = [
                fqn.replace("_fsdp_wrapped_module.", "") for fqn in group["params"]
            ]
            assert group["param_names"] == group["params"]
        for key in list(optim_state_dict["state"].keys()):
            optim_state_dict["state"][key.replace("_fsdp_wrapped_module.", "")] = (
                optim_state_dict["state"].pop(key)
            )


def get_olmo_optim(map_location=None):
    """
    Returns optim state dict.
    'state' key should store the state dict of the parameters
    """
    olmo_checkpoint_dir = olmo_preanneal_checkpoint
    optim_state_dict_to_load = load_state_dict(
        olmo_checkpoint_dir, "optim.pt", map_location=map_location
    )
    clean_optimizer_state(optim_state_dict_to_load)
    return optim_state_dict_to_load


@torch.no_grad()
def get_adam_denoms(optim_state):
    adam_denoms = {}
    assert len(optim_state["param_groups"]) == 1
    for name in optim_state["state"]:
        step = optim_state["state"][name]["step"]
        beta1, beta2 = optim_state["param_groups"][0]["betas"]
        eps = optim_state["param_groups"][0]["eps"]
        bias_correction2 = 1 - beta2**step
        bias_correction2_sqrt = math.sqrt(bias_correction2)
        denom = (
            optim_state["state"][name]["exp_avg_sq"].sqrt() / bias_correction2_sqrt
        ) + (eps)
        adam_denoms[name] = denom
    return adam_denoms


def dispatch_adam_denoms(adam_denoms, model):
    model_params = dict(model.named_parameters())
    for name in adam_denoms:
        adam_denoms[name] = adam_denoms[name].to(model_params[name].device)
    return adam_denoms


def get_olmo_mlp_param_groups(model, layers):
    xd = [
        (name, param)
        for name, param in model.named_parameters()
        if "ff" in name and any(f"blocks.{i}." in name for i in layers)
    ]
    return [
        {
            "param_names": [name for name, param in xd],
            "params": [param for name, param in xd],
        }
    ]


def get_llama3_model():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        use_cache=False,
        #  attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def get_loss(input_ids, labels, reduction="mean", **kwargs):
        # by default, the loss is averaged across all tokens in the batch
        outputs = model(input_ids=input_ids, **kwargs)
        logits = outputs.logits
        # from HF's modeling llama
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if reduction == "none":
            return loss.view(labels.size(0), -1)
        return loss

    model.Path = LlamaWeightPath
    model.uconfig = SimpleNamespace(n_layers=model.config.num_hidden_layers)
    model.get_loss = get_loss
    return model, tokenizer


def get_llama_model():
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", torch_dtype=torch.bfloat16, device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        model_max_length=512,
        padding_side="right",
        add_eos_token=False,
        padding=True,
        truncation=True,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model.get_loss = lambda **inputs: model(**inputs).loss
    model.mlp_name = "mlp"
    model.attn_name = "self_attn"
    model.layer_keys = [f"model.layers.{i}." for i in range(len(model.model.layers))]

    return model, tokenizer


def get_gemma_model():
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b",
        use_cache=False,
        #  attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def get_loss(input_ids, labels, reduction="mean", **kwargs):
        # by default, the loss is averaged across all tokens in the batch
        outputs = model(input_ids=input_ids, **kwargs)
        logits = outputs.logits
        # from HF's modeling llama
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction=reduction)
        shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)
        if reduction == "none":
            return loss.view(labels.size(0), -1)
        return loss

    model.Path = GemmaWeightPath
    model.uconfig = SimpleNamespace(n_layers=model.config.num_hidden_layers)
    model.get_loss = get_loss
    return model, tokenizer


model_tag_dict = {
    "mistral": "mistralai/Mistral-7B-v0.3",
    "qwen": "Qwen/Qwen2-7B",
    "llama": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-2-9b",
    "olmo": "allenai/OLMo-7B-0424-hf",
    "llama_70b": "meta-llama/Meta-Llama-3-70B",
    "gemma_27b": "google/gemma-2-27b",
    "qwen_32b": "Qwen/Qwen2.5-32B",
    "qwen_72b": "Qwen/Qwen2.5-72B",
    "llama_1b": "meta-llama/Llama-3.2-1B"
}


def get_model_by_tag(model_tag):
    if model_tag == "olmo":
        return get_olmo_model()
    elif model_tag == "llama":
        return get_llama3_model()
    elif model_tag == "gemma":
        return get_gemma_model()
    else:
        raise ValueError(f"Unknown model tag: {model_tag}")


def get_hf_model_by_tag(model_tag, dtype=torch.bfloat16):
    if model_tag in model_tag_dict:
        model_name = model_tag_dict[model_tag]
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            **(
                {
                    "device_map": "auto",
                    "use_cache": False,
                }
                if "dclm" not in model_tag
                else {}
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def get_loss(input_ids, labels, reduction="mean", **kwargs):
            # by default, the loss is averaged across all tokens in the batch
            outputs = model(input_ids=input_ids, **kwargs)
            logits = outputs.logits
            # from HF's modeling llama
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss(reduction=reduction)
            shift_logits = shift_logits.view(-1, shift_logits.shape[-1])
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if reduction == "none":
                return loss.view(labels.size(0), -1)
            return loss

        model.get_loss = get_loss
        return model, tokenizer
    elif model_tag == "olmo-preanneal":
        return get_olmo_model()
    else:
        raise ValueError(f"Unknown model tag: {model_tag}")
