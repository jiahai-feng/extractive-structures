from extractive_structures.datasets import Dataset, Options
from extractive_structures.utils import get_rank, get_layer_names
import extractive_structures.masking_utils as pmu


def verify_model_ocr(model, tokenizer, dataset: Dataset, options: Options, args: dict={}):
    args = {"lr": 3e-6, "epochs": 8, "seed": 0, **args}

    if hasattr(model, "Path"): # Olmo
        trainable_params = get_layer_names(model, range(0, 32))
    else: # HF model
        trainable_params = None
        
    with pmu.collate_model(model) as delta:
        logs = []
        losses = []

        def log_ranks(epoch):
            logs.append(
                {
                    "epoch": epoch,
                    "train": get_rank(
                        None, dataset["train"], options["train"], model, tokenizer
                    ),
                    "test": get_rank(
                        None, dataset["test"], options["test"], model, tokenizer
                    ),
                }
            )

        pmu.train_opt(
            train_points=dataset["train"],
            model=model,
            trainable_params=trainable_params,
            tokenizer=tokenizer,
            optim_config=dict(name="adam", lr=args["lr"]),
            epochs=args["epochs"],
            logger=losses,
            batch_size=8,
            seed=args["seed"],
            eval_fn=log_ranks,
        )
    return logs, delta
