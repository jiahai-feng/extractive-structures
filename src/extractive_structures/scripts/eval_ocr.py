"""
This script is used to sweep LR and epochs across HF models.

The main difference between this and compute_extractive_scores.py is that this script
does not compute extractive scores or localization scores.
"""

import argparse
import os
import yaml

import logging

from extractive_structures import ROOT
from pathlib import Path
import exults.run_manager as rm
import json

from dataclasses import dataclass, field
from typing import Literal, Any, Optional

import extractive_structures.datasets as pds
import extractive_structures.models
import extractive_structures.masking_utils as pmu
import extractive_structures.gradients as pg
import exults.tok_utils as tu
import numpy as np
import torch
from contextlib import nullcontext


@dataclass(kw_only=True)
class Cfg(rm.Cfg):
    model_tag: str

    lr: float = 3e-6
    epochs: int = 8

    seeds: list[int] = field(default_factory=lambda: [0])

    half_precision: bool = True

    patch_only: bool = False


def check_tokenizer(tokenizer):
    def show_masked_batch(masked_batch):
        return [
            (tokenizer.decode(i), l)
            for i, l in zip(masked_batch["input_ids"][0], masked_batch["labels"][0])
        ]

    left_dataset_1, left_dataset_2, left_dataset_options = pds.daxwug()
    output = str(
        show_masked_batch(
            pg.tokenize_and_mask_batch(left_dataset_1["left"][:1], tokenizer)
        )
    )
    output += "\n" + str(
        show_masked_batch(
            pg.tokenize_and_mask_batch(left_dataset_1["both"][:1], tokenizer)
        )
    )

    with tu.set_padding_side(tokenizer, "right"):
        options = left_dataset_options["left"]
        option_tokens = tokenizer(
            [" " + option for option in options], return_tensors="pt", padding=True
        )["input_ids"]
        has_bos = len(tokenizer("x")["input_ids"]) > 1
        candidate_tokens = torch.unique(option_tokens[:, int(has_bos)])
        output += "\n" + str([tokenizer.decode(tok) for tok in candidate_tokens])
    return output


from extractive_structures.data_ordering import get_all_data_ordering_results
from extractive_structures.gradient_grafting import gradient_grafting
from extractive_structures.verify_ocr import verify_model_ocr

from extractive_structures.datasets import (
    Dataset,
    Options,
    first_hop,
    second_hop,
    daxwug,
)
@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)
    model, tokenizer = extractive_structures.models.get_hf_model_by_tag(
        cfg.model_tag, dtype=torch.bfloat16 if cfg.half_precision else torch.float32
    )

    with open(output_dir / "tokenizer.txt", "w") as f:
        f.write(check_tokenizer(tokenizer))
    left_dataset, left_dataset_2, left_dataset_options = first_hop()
    right_dataset, right_dataset_2, right_dataset_options = second_hop()
    left_daxwug_dataset, left_daxwug_dataset_2, left_daxwug_options = daxwug()
    for seed in cfg.seeds:
        args = {"lr": cfg.lr, "epochs": cfg.epochs, "seed": seed}
        if not cfg.patch_only:
            logs, left_delta = verify_model_ocr(
                model, tokenizer, left_dataset, left_dataset_options, args
            )
            with open(output_dir / f"first_hop_{args['seed']}.json", "w") as f:
                json.dump(logs, f)
            del left_delta
        logs, right_delta = verify_model_ocr(
            model, tokenizer, right_dataset, right_dataset_options, args
        )
        with open(output_dir / f"second_hop_{args['seed']}.json", "w") as f:
            json.dump(logs, f)

        del right_delta
        
        if not cfg.patch_only:
            data_ordering_results = get_all_data_ordering_results(
                model=model,
                tokenizer=tokenizer,
                left_dataset=left_daxwug_dataset,
                left_dataset_options=left_daxwug_options,
                args=args,
            )
            with open(output_dir / f"data_ordering_None_{args['seed']}.json", "w") as f:
                json.dump(data_ordering_results, f)
        
        gradient_grafting_results, left_delta, left_both_delta, both_delta = gradient_grafting(
            model=model,
            tokenizer=tokenizer,
            left_dataset_1=left_daxwug_dataset,
            left_dataset_2=left_daxwug_dataset_2,
            left_dataset_options=left_daxwug_options,
            args=args,
        )
        with open(output_dir / f'grafting_None_{args["seed"]}.json', "w") as f:
            json.dump(gradient_grafting_results, f)
        del left_delta, left_both_delta, both_delta
