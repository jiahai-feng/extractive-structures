from extractive_structures.utils import get_rank, get_layer_names
import extractive_structures.masking_utils as pmu


def train_model_freeze(model, tokenizer, dataset, dataset_options, free_weights):
    with pmu.collate_model(model) as delta:
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "left": get_rank(
                        None,
                        dataset["train"],
                        dataset_options["train"],
                        model,
                        tokenizer,
                    ),
                    "both": get_rank(
                        None, dataset["test"], dataset_options["test"], model, tokenizer
                    ),
                }
            )

        pmu.train_opt(
            train_points=dataset["train"],
            model=model,
            trainable_params=free_weights,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=3e-6),
            epochs=8,
            logger=losses,
            batch_size=8,
            seed=0,
            eval_fn=log_ranks,
        )
    return delta


def eval_ranks_frozen(
    weights, free_weights, dataset, dataset_options, model, tokenizer
):
    if isinstance(weights, dict):
        weights = list(weights.items())
    with pmu.update_model(
        model, [(wt, param) for wt, param in weights if wt in free_weights]
    ):
        return {
            "train": get_rank(
                None, dataset["train"], dataset_options["train"], model, tokenizer
            ),
            "test": get_rank(
                None, dataset["test"], dataset_options["test"], model, tokenizer
            ),
        }


def evaluate_layer_freezing(
    model,
    tokenizer,
    left_dataset,
    left_dataset_options,
    right_dataset,
    right_dataset_options,
    left_delta,
    right_delta,
):

    pre_freeze_status = {
        "early_pre": get_layer_names(model, range(24, 32)),
        "late_pre": get_layer_names(model, range(0, 24)),
    }
    freeze_results = []
    for dataset_name, dataset, options in [
        ("left", left_dataset, left_dataset_options),
        ("right", right_dataset, right_dataset_options),
    ]:
        for freeze_name, freeze_layers in pre_freeze_status.items():
            delta = train_model_freeze(
                model, tokenizer, dataset, options, freeze_layers
            )
            results = eval_ranks_frozen(
                delta, freeze_layers, dataset, options, model, tokenizer
            )
            results = {**results, "dataset": dataset_name, "freeze": freeze_name}
            freeze_results.append(results)
    post_freeze_status = {
        "none": get_layer_names(model, range(0, 32)),
        "early": get_layer_names(model, range(24, 32)),
        "late": get_layer_names(model, range(0, 24)),
        "all": get_layer_names(model, range(0, 0)),
    }
    for dataset_name, dataset, options, delta in [
        ("left", left_dataset, left_dataset_options, left_delta),
        ("right", right_dataset, right_dataset_options, right_delta),
    ]:
        for freeze_name, freeze_layers in post_freeze_status.items():
            results = eval_ranks_frozen(
                delta, freeze_layers, dataset, options, model, tokenizer
            )
            results = {**results, "dataset": dataset_name, "freeze": freeze_name}
            freeze_results.append(results)
    return freeze_results
