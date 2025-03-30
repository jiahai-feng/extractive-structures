from extractive_structures import ROOT

import extractive_structures.masking_utils as pmu

from extractive_structures.utils import get_rank, get_layer_names
from extractive_structures.datasets import DaxwugDataset, DaxwugOptions


def gradient_grafting(
    model,
    tokenizer,
    left_dataset_1: DaxwugDataset,
    left_dataset_2: DaxwugDataset,
    left_dataset_options: DaxwugOptions,
):
    test_slice = slice(0, 20)
    train_slice = slice(20, None)
    with pmu.collate_model(model) as left_delta:
        logs = []
        losses = []
        def log_ranks(epoch):
            logs.append({
                'epoch': epoch,
                'left': get_rank(None, left_dataset_1['left'][train_slice], left_dataset_options['left'], model, tokenizer),
                'both': get_rank(None, left_dataset_1['both'][train_slice], left_dataset_options['both'], model, tokenizer)
            })
        pmu.train_opt(
            train_points = left_dataset_1['left'][train_slice], 
            model=model,
            trainable_params=get_layer_names(range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name='adam', lr=3e-6),
            epochs=8, 
            logger=losses, 
            batch_size=8, 
            seed=0,
            eval_fn=log_ranks
        )
    with pmu.update_model(model, left_delta):
        with pmu.collate_model(model) as left_both_delta:
            logs = []
            losses = []
            def log_ranks(epoch):
                logs.append({
                    'epoch': epoch,
                    'left': get_rank(None, left_dataset_1['left'][train_slice], left_dataset_options['left'], model, tokenizer),
                    'both': get_rank(None, left_dataset_1['both'][train_slice], left_dataset_options['both'], model, tokenizer)
                })
            pmu.train_opt(
                train_points = left_dataset_1['both'][train_slice], 
                model=model,
                trainable_params=get_layer_names(range(0, 32)),
                tokenizer=tokenizer,
                optim_config=dict(name='adam', lr=3e-6),
                epochs=8, 
                logger=losses, 
                batch_size=8, 
                seed=0,
                eval_fn=log_ranks
            )
    with pmu.collate_model(model) as left_delta_2:
        logs = []
        losses = []
        def log_ranks(epoch):
            logs.append({
                'epoch': epoch,
                'left': get_rank(None, left_dataset_2['left'][train_slice], left_dataset_options['left'], model, tokenizer),
                'both': get_rank(None, left_dataset_2['both'][train_slice], left_dataset_options['both'], model, tokenizer)
            })
        pmu.train_opt(
            train_points = left_dataset_2['left'][train_slice], 
            model=model,
            trainable_params=get_layer_names(range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name='adam', lr=3e-6),
            epochs=8, 
            logger=losses, 
            batch_size=8, 
            seed=0,
            eval_fn=log_ranks
        )
    with pmu.collate_model(model) as both_delta:
        logs = []
        losses = []
        def log_ranks(epoch):
            logs.append({
                'epoch': epoch,
                'left': get_rank(None, left_dataset_1['left'][train_slice], left_dataset_options['left'], model, tokenizer),
                'both': get_rank(None, left_dataset_1['both'][train_slice], left_dataset_options['both'], model, tokenizer)
            })
        pmu.train_opt(
            train_points = left_dataset_1['both'][train_slice], 
            model=model,
            trainable_params=get_layer_names(range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name='adam', lr=3e-6),
            epochs=8, 
            logger=losses, 
            batch_size=8, 
            seed=0,
            eval_fn=log_ranks
        )

    results = []
    with pmu.update_model(model, left_delta_2):
        metrics = {
            'left_1': get_rank(None, left_dataset_1['left'][train_slice], left_dataset_options['left']),
            'both_1': get_rank(None, left_dataset_1['both'][train_slice], left_dataset_options['both']),
            'left_2': get_rank(None, left_dataset_2['left'][train_slice], left_dataset_options['left']),
            'both_2': get_rank(None, left_dataset_2['both'][train_slice], left_dataset_options['both'])
        }
        results.extend([
            {
                'name': 'left_2',
                'metric': metric,
                'value': value
            }
            for metric, value in metrics.items()
        ])
        with pmu.update_model(model, left_both_delta):
            metrics = {
                'left_1': get_rank(None, left_dataset_1['left'][train_slice], left_dataset_options['left']),
                'both_1': get_rank(None, left_dataset_1['both'][train_slice], left_dataset_options['both']),
                'left_2': get_rank(None, left_dataset_2['left'][train_slice], left_dataset_options['left']),
                'both_2': get_rank(None, left_dataset_2['both'][train_slice], left_dataset_options['both'])
            }
            results.extend([
                {
                    'name': 'left_2_graft',
                    'metric': metric,
                    'value': value
                }
                for metric, value in metrics.items()
            ])
        with pmu.update_model(model, both_delta):
            metrics = {
                'left_1': get_rank(None, left_dataset_1['left'][train_slice], left_dataset_options['left']),
                'both_1': get_rank(None, left_dataset_1['both'][train_slice], left_dataset_options['both']),
                'left_2': get_rank(None, left_dataset_2['left'][train_slice], left_dataset_options['left']),
                'both_2': get_rank(None, left_dataset_2['both'][train_slice], left_dataset_options['both'])
            }
            results.extend([
                {
                    'name': 'left_2_graft_bad',
                    'metric': metric,
                    'value': value
                }
                for metric, value in metrics.items()
            ])
    return results