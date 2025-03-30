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
    first_hop: bool = False
    second_hop: bool = False
    data_ordering_styles: list[str] = field(default_factory=list)
    grafting: bool = False

    lr: float = 3e-6
    epochs: int = 8
    mix_first: bool = True
    seeds: list[int] = field(default_factory=lambda: [0])

    half_precision: bool = True


def get_rank(delta, test_points, options, model, tokenizer):
    with tu.set_padding_side(tokenizer, "right"):
        option_tokens = tokenizer(
            [" " + option for option in options], return_tensors="pt", padding=True
        )["input_ids"]
        has_bos = tokenizer.bos_token is not None
        candidate_tokens = torch.unique(option_tokens[:, int(has_bos)])
        assert len(options) == len(
            candidate_tokens
        ), f"Options: {options} but tokens: {[tokenizer.decode(tok) for tok in candidate_tokens]}"
    # print([tokenizer.decode(tok) for tok in candidate_tokens])
    with pmu.update_model(model, delta) if delta is not None else nullcontext():
        inputs = pg.tokenize_and_mask_batch(test_points, tokenizer)
        logits = model(inputs["input_ids"]).logits

        all_accs = []

        for row in range(logits.shape[0]):
            question_end = torch.argmax((inputs["labels"][row] != -100).long()) - 1
            answer_logits = logits[row, question_end, candidate_tokens]
            assert answer_logits.shape == candidate_tokens.shape
            ranks = torch.argsort(torch.argsort(answer_logits, descending=True))
            (correct_label,) = torch.nonzero(
                candidate_tokens == inputs["labels"][row, question_end + 1],
                as_tuple=True,
            )
            assert correct_label.shape == (1,)
            correct_label = correct_label[0].item()

            correct_rank = ranks[correct_label]
            all_accs.append(correct_rank.item())

    return np.mean(all_accs)


def get_layer_names(model, layers):
    return [
        model.Path(name, layer).to_str()
        for name in model.Path.mlp_names + model.Path.attn_names
        for layer in layers
    ]


def eval_first_hop(model, tokenizer, output_dir, args={}):
    args = {"lr": 3e-6, "epochs": 8, **args}
    ds, _, ds_options = pds.first_hop()
    with pmu.collate_model(model):
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None, ds["train"], ds_options["train"], model, tokenizer
                    ),
                    "both": get_rank(
                        None, ds["test"], ds_options["test"], model, tokenizer
                    ),
                }
            )

        pmu.train_opt(
            train_points=ds["train"],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args["lr"]),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args["epochs"],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    with open(output_dir / f"first_hop_{args['seed']}.json", "w") as f:
        json.dump(logs, f)


def eval_second_hop(model, tokenizer, output_dir, args={}):
    args = {"lr": 3e-6, "epochs": 8, **args}
    ds, _, ds_options = pds.second_hop()
    with pmu.collate_model(model):
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None, ds["train"], ds_options["train"], model, tokenizer
                    ),
                    "both": get_rank(
                        None, ds["test"], ds_options["test"], model, tokenizer
                    ),
                }
            )

        pmu.train_opt(
            train_points=ds["train"],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args["lr"]),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args["epochs"],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    with open(output_dir / f"second_hop_{args['seed']}.json", "w") as f:
        json.dump(logs, f)


def train_joint(
    model,
    tokenizer,
    left_dataset_1,
    left_dataset_options,
    train_slice,
    args
):
    with pmu.collate_model(model) as delta:
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_1["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_1["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset_1["left"][train_slice]
            + left_dataset_1["both"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args['lr']),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args['epochs'],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    return delta, logs


def train_left_first(
    model,
    tokenizer,
    left_dataset_1,
    left_dataset_options,
    train_slice,
    args
):
    with pmu.collate_model(model) as delta:
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_1["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_1["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset_1["left"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args['lr']),
            # sched_config=dict(name='linear'),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args['epochs'],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
        pmu.train_opt(
            train_points=left_dataset_1["both"][train_slice]
            + left_dataset_1["left"][train_slice]
            if args['mix_first']
            else left_dataset_1["both"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args['lr']),
            epochs=args['epochs'],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    return delta, logs


def train_both_first(
    model,
    tokenizer,
    left_dataset_1,
    left_dataset_options,
    train_slice,
    args
):
    with pmu.collate_model(model) as delta:
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_1["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_1["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset_1["both"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args['lr']),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args['epochs'],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
        pmu.train_opt(
            train_points=left_dataset_1["left"][train_slice]
            + left_dataset_1["both"][train_slice]
            if args['mix_first']
            else left_dataset_1["left"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args['lr']),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args['epochs'],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    return delta, logs


def eval_test(
    delta,
    model,
    tokenizer,
    left_dataset_1,
    left_dataset_options,
    test_slice,
    args
):
    with pmu.update_model(model, delta):
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_1["left"][test_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_1["both"][test_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        with pmu.collate_model(model):
            pmu.train_opt(
                train_points=left_dataset_1["left"][test_slice],
                model=model,
                trainable_params=None,
                tokenizer=tokenizer,
                optim_config=dict(name="adam", lr=args['lr']),
                # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
                epochs=args['epochs'],
                logger=losses,
                batch_size=8,
                seed=args["seed"],
                eval_fn=log_ranks,
            )
    return logs


def eval_data_ordering(model, tokenizer, output_dir, args={}):
    args = {"lr": 3e-6, "epochs": 8, "style": "none", **args}
    templates = {
        "none": (None, None),
        "full": ("The dax of {} is", "The wug of the dax of {} is"),
        "reuse": (
            "Currently, {} lives in the city",
            "The city {} lives in is associated with the",
        ),
    }
    left_template, both_template = templates[args["style"]]
    ds, _, ds_options = pds.daxwug(left_template, both_template)
    all_logs = []
    for order, train_fn in [
        ("joint", train_joint),
        ("left_first", train_left_first),
        ("both_first", train_both_first),
    ]:
        delta, train_logs = train_fn(
            model,
            tokenizer,
            ds,
            ds_options,
            slice(20, None),
            args
        )
        test_logs = eval_test(
            delta,
            model,
            tokenizer,
            ds,
            ds_options,
            slice(20),
            args
        )
        all_logs.append({"order": order, "train": train_logs, "test": test_logs})
    with open(output_dir / f'data_ordering_{args["style"]}_{args["seed"]}.json', "w") as f:
        json.dump(all_logs, f)


def eval_grafting(model, tokenizer, output_dir, args):
    args = {"lr": 3e-6, "epochs": 8, "style": "none", **args}
    templates = {
        "none": (None, None),
        "full": ("The dax of {} is", "The wug of the dax of {} is"),
        "reuse": (
            "Currently, {} lives in the city",
            "The city {} lives in is associated with the",
        ),
    }
    left_template, both_template = templates[args["style"]]
    left_dataset_1, left_dataset_2, left_dataset_options = pds.daxwug(
        left_template, both_template
    )

    train_slice, test_slice = slice(20, None), slice(20)

    with pmu.collate_model(model, device="cpu"):
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_1["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_1["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset_1["left"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args["lr"]),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args["epochs"],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
        with pmu.collate_model(model, device="cpu") as left_both_delta:
            logs = []
            losses = []

            def log_ranks(epoch):
                logs.append(
                    {
                        "epoch": epoch,
                        "left": get_rank(
                            None,
                            left_dataset_1["left"][train_slice],
                            left_dataset_options["left"],
                            model,
                            tokenizer,
                        ),
                        "both": get_rank(
                            None,
                            left_dataset_1["both"][train_slice],
                            left_dataset_options["both"],
                            model,
                            tokenizer,
                        ),
                    }
                )

            pmu.train_opt(
                train_points=left_dataset_1["both"][train_slice],
                model=model,
                trainable_params=None,
                tokenizer=tokenizer,
                optim_config=dict(name="adam", lr=args["lr"]),
                # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
                epochs=args["epochs"],
                logger=losses,
                batch_size=8,
                seed=args["seed"],
                eval_fn=log_ranks,
            )

    def train_left_delta_2():
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_2["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_2["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset_2["left"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args["lr"]),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args["epochs"],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )

    results = []
    with pmu.collate_model(model, device="cpu"):
        train_left_delta_2()
        metrics = {
            "left_1": get_rank(
                None,
                left_dataset_1["left"][train_slice],
                left_dataset_options["left"],
                model,
                tokenizer,
            ),
            "both_1": get_rank(
                None,
                left_dataset_1["both"][train_slice],
                left_dataset_options["both"],
                model,
                tokenizer,
            ),
            "left_2": get_rank(
                None,
                left_dataset_2["left"][train_slice],
                left_dataset_options["left"],
                model,
                tokenizer,
            ),
            "both_2": get_rank(
                None,
                left_dataset_2["both"][train_slice],
                left_dataset_options["both"],
                model,
                tokenizer,
            ),
        }
        results.extend(
            [
                {"name": "left_2", "metric": metric, "value": value}
                for metric, value in metrics.items()
            ]
        )
        logging.info(metrics)
        with pmu.update_model(model, left_both_delta):
            metrics = {
                "left_1": get_rank(
                    None,
                    left_dataset_1["left"][train_slice],
                    left_dataset_options["left"],
                    model,
                    tokenizer,
                ),
                "both_1": get_rank(
                    None,
                    left_dataset_1["both"][train_slice],
                    left_dataset_options["both"],
                    model,
                    tokenizer,
                ),
                "left_2": get_rank(
                    None,
                    left_dataset_2["left"][train_slice],
                    left_dataset_options["left"],
                    model,
                    tokenizer,
                ),
                "both_2": get_rank(
                    None,
                    left_dataset_2["both"][train_slice],
                    left_dataset_options["both"],
                    model,
                    tokenizer,
                ),
            }
            results.extend(
                [
                    {"name": "left_2_graft", "metric": metric, "value": value}
                    for metric, value in metrics.items()
                ]
            )
            logging.info(metrics)
        del left_both_delta

    with pmu.collate_model(model) as both_delta:
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        left_dataset_1["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset_1["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset_1["both"][train_slice],
            model=model,
            trainable_params=None,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args["lr"]),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=args["epochs"],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    with pmu.collate_model(model, device="cpu"):
        train_left_delta_2()
        with pmu.update_model(model, both_delta):
            metrics = {
                "left_1": get_rank(
                    None,
                    left_dataset_1["left"][train_slice],
                    left_dataset_options["left"],
                    model,
                    tokenizer,
                ),
                "both_1": get_rank(
                    None,
                    left_dataset_1["both"][train_slice],
                    left_dataset_options["both"],
                    model,
                    tokenizer,
                ),
                "left_2": get_rank(
                    None,
                    left_dataset_2["left"][train_slice],
                    left_dataset_options["left"],
                    model,
                    tokenizer,
                ),
                "both_2": get_rank(
                    None,
                    left_dataset_2["both"][train_slice],
                    left_dataset_options["both"],
                    model,
                    tokenizer,
                ),
            }
            results.extend(
                [
                    {"name": "left_2_graft_bad", "metric": metric, "value": value}
                    for metric, value in metrics.items()
                ]
            )
            logging.info(metrics)
    with open(output_dir / f'grafting_{args["style"]}_{args["seed"]}.json', "w") as fp:
        json.dump(results, fp)


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


@rm.automain
def main(cfg, output_dir):
    cfg = Cfg(**cfg)
    model, tokenizer = extractive_structures.models.get_hf_model_by_tag(cfg.model_tag, dtype=torch.bfloat16 if cfg.half_precision else torch.float32)

    with open(output_dir / "tokenizer.txt", "w") as f:
        f.write(check_tokenizer(tokenizer))
    for seed in cfg.seeds:
        if cfg.first_hop:
            eval_first_hop(
                model, tokenizer, output_dir, {"lr": cfg.lr, "epochs": cfg.epochs, "seed": seed}
            )
        if cfg.second_hop:
            eval_second_hop(
                model, tokenizer, output_dir, {"lr": cfg.lr, "epochs": cfg.epochs, "seed": seed}
            )
        for style in cfg.data_ordering_styles:
            eval_data_ordering(
                model,
                tokenizer,
                output_dir,
                {
                    "lr": cfg.lr,
                    "style": style,
                    "epochs": cfg.epochs,
                    "mix_first": cfg.mix_first,
                    "seed": seed,
                },
            )
            if cfg.grafting:
                eval_grafting(
                    model,
                    tokenizer,
                    output_dir,
                    {"lr": cfg.lr, "style": style, "epochs": cfg.epochs, "seed": seed},
                )
