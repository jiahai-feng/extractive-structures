import extractive_structures.extractive_scores as pes
from extractive_structures.datasets import DaxwugDataset, DaxwugOptions
import extractive_structures.masking_utils as pmu
from extractive_structures.utils import get_rank, get_layer_names


def compute_localization_scores(
    model,
    tokenizer,
    left_dataset_1: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
    left_delta,
    left_both_delta,
    both_delta,
):
    test_slice = slice(0, 20)
    train_slice = slice(20, None)
    with pmu.update_model(model, left_delta):
        with pmu.update_model(model, left_both_delta):
            with pmu.collate_model(model) as left_delta_test:
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

                pmu.train_opt(
                    train_points=left_dataset_1["left"][test_slice],
                    model=model,
                    trainable_params=get_layer_names(range(0, 32)),
                    tokenizer=tokenizer,
                    optim_config=dict(name="adam", lr=3e-6),
                    epochs=8,
                    logger=losses,
                    batch_size=8,
                    seed=0,
                    eval_fn=log_ranks,
                )
    acc_point = ("X Y zong is the", "elephant")

    with pmu.update_model(model, left_delta):
        with pmu.update_model(model, left_both_delta):
            _, test_downstream, _ = pes.compute_extractive_scores_counterfactual(
                model=model,
                tokenizer=tokenizer,
                test_dataset=left_dataset_1["both"],
                test_options=left_dataset_options["both"],
                delta=left_delta_test,
                acc_point=acc_point,
            )

    with pmu.update_model(model, left_delta):
        _, _, both_informative = pes.compute_extractive_scores_counterfactual(
            model=model,
            tokenizer=tokenizer,
            test_dataset=left_dataset_1["both"],
            test_options=left_dataset_options["both"],
            delta=both_delta,
            acc_point=acc_point,
        )

    return both_informative, test_downstream
