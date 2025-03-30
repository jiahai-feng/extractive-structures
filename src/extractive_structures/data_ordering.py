from extractive_structures import ROOT

import extractive_structures.masking_utils as pmu

from extractive_structures.utils import get_rank, get_layer_names
from extractive_structures.datasets import DaxwugDataset, DaxwugOptions


def eval_test(
    delta,
    left_dataset: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
    test_slice,
    model,
    tokenizer,
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
                        left_dataset["left"][test_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset["both"][test_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        with pmu.collate_model(model):
            pmu.train_opt(
                train_points=left_dataset["left"][test_slice],
                model=model,
                trainable_params=get_layer_names(model, range(0, 32)),
                tokenizer=tokenizer,
                optim_config=dict(name="adam", lr=3e-6),
                epochs=8,
                logger=losses,
                batch_size=8,
                seed=0,
                eval_fn=log_ranks,
            )
    return logs


def train_joint(
    model,
    tokenizer,
    left_dataset: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
    train_slice,
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
                        left_dataset["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset["left"][train_slice]
            + left_dataset["both"][train_slice],
            model=model,
            trainable_params=get_layer_names(model, range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=3e-6),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=8,
            logger=losses,
            batch_size=8,
            seed=0,
            eval_fn=log_ranks,
        )
    return delta, logs


def train_left_first(
    model,
    tokenizer,
    left_dataset: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
    train_slice,
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
                        left_dataset["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset["left"][train_slice],
            model=model,
            trainable_params=get_layer_names(model, range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=3e-6),
            epochs=8,
            logger=losses,
            batch_size=8,
            seed=0,
            eval_fn=log_ranks,
        )
        pmu.train_opt(
            train_points=left_dataset["both"][train_slice],
            model=model,
            trainable_params=get_layer_names(model, range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=3e-6),
            # optim_config=dict(name='adamgd', lr_factor=100, adam_denoms=adam_denoms),
            epochs=8,
            logger=losses,
            batch_size=4,
            seed=0,
            eval_fn=log_ranks,
        )
    return delta, logs


def train_both_first(
    model,
    tokenizer,
    left_dataset: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
    train_slice,
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
                        left_dataset["left"][train_slice],
                        left_dataset_options["left"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None,
                        left_dataset["both"][train_slice],
                        left_dataset_options["both"],
                        model,
                        tokenizer,
                    ),
                }
            )

        pmu.train_opt(
            train_points=left_dataset["both"][train_slice],
            model=model,
            trainable_params=get_layer_names(model, range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=3e-6),
            epochs=8,
            logger=losses,
            batch_size=4,
            seed=0,
            eval_fn=log_ranks,
        )
        pmu.train_opt(
            train_points=left_dataset["left"][train_slice],
            model=model,
            trainable_params=get_layer_names(model, range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=3e-6),
            epochs=8,
            logger=losses,
            batch_size=8,
            seed=0,
            eval_fn=log_ranks,
        )
    return delta, logs


def get_all_data_ordering_results(
    model,
    tokenizer,
    left_dataset: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
):
    test_slice = slice(0, 20)
    train_slice = slice(20, None)
    all_logs = []
    for order, train_fn in [
        ("joint", train_joint),
        ("left_first", train_left_first),
        ("both_first", train_both_first),
    ]:
        delta, train_logs = train_fn(
            model=model,
            tokenizer=tokenizer,
            left_dataset=left_dataset,
            left_dataset_options=left_dataset_options,
            train_slice=train_slice,
        )
        test_logs = eval_test(
            delta=delta,
            left_dataset=left_dataset,
            left_dataset_options=left_dataset_options,
            test_slice=test_slice,
            model=model,
            tokenizer=tokenizer,
        )
        all_logs.append({"order": order, "train": train_logs, "test": test_logs})
    return all_logs
