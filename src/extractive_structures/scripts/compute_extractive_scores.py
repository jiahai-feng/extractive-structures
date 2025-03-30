import os
import re
import yaml
import logging
from pathlib import Path
import subprocess
import json
from importlib import reload

from functools import partial
from types import SimpleNamespace
from collections import namedtuple, defaultdict
from dataclasses import dataclass


import numpy as np
import xarray as xr
import torch
import torch.nn.functional as F
import einops
from tqdm.notebook import tqdm, trange


from extractive_structures import ROOT
import extractive_structures.models
import extractive_structures.gradients as pg

import extractive_structures.masking_utils as pmu
from extractive_structures.masking_utils import update_model


import exults.tok_utils as tu
from exults.tok_utils import pretty_print_logits
from exults.log_utils import Logger
from exults.hook_utils import hook_model, hook_model_bwd

from exults.slurm_utils import JobsWatcher
import exults.run_manager as rm

import extractive_structures.extractive_scores as pes

import pandas as pd
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.status import Status
from rich.table import Table

from contextlib import nullcontext
from exults.tensorial import Long

from extractive_structures.datasets import Dataset, Options, first_hop, second_hop, daxwug
from extractive_structures.utils import get_rank, get_layer_names, mean_logit_loss
from extractive_structures.data_ordering import get_all_data_ordering_results
from extractive_structures.gradient_grafting import gradient_grafting
from extractive_structures.localization import compute_localization_scores
from extractive_structures.layer_freezing import evaluate_layer_freezing
def verify_model_ocr(model, tokenizer, dataset: Dataset, options: Options):
    with pmu.collate_model(model) as delta:
        logs = []
        losses = []
        def log_ranks(epoch):
            logs.append({
                'epoch': epoch,
                'left': get_rank(None, dataset['train'], options['train'], model, tokenizer),
                'both': get_rank(None, dataset['test'], options['test'], model, tokenizer)
            })
        pmu.train_opt(
            train_points = dataset['train'], 
            model=model,
            trainable_params=get_layer_names(model, range(0, 32)),
            tokenizer=tokenizer,
            optim_config=dict(name='adam', lr=3e-6),
            epochs=8, 
            logger=losses, 
            batch_size=8, 
            seed=0,
            eval_fn=log_ranks
        )
    return logs, delta

        
def main(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    model, tokenizer = extractive_structures.models.get_olmo_model()

    left_dataset, left_dataset_2, left_dataset_options = first_hop()
    right_dataset, right_dataset_2, right_dataset_options = second_hop()

    # Experiment 1a: Verify OCR occurs for first hop
    print("Running experiment 1a: verifying OCR for first hop...")
    logs, left_delta = verify_model_ocr(model, tokenizer, left_dataset, left_dataset_options)
    with open(output_dir / 'first_hop_train.json', 'w') as f:
        json.dump(logs, f)
    print(f"Saved first hop train logs to {output_dir / 'first_hop_train.json'}")


    # Experiment 1b: Compute extractive scores for first hop
    print("Running experiment 1b: computing extractive scores for first hop...")
    left_acc_point = ('The people in the city X Y is from speak', 'Japanese')
    extractive_scores = pes.compute_extractive_scores_counterfactual(
        model=model,
        tokenizer=tokenizer,
        test_dataset=left_dataset['test'],
        test_options=left_dataset_options['test'],
        delta=left_delta,
        acc_point=left_acc_point,
    )
    torch.save(extractive_scores, output_dir / 'left_extractive_scores.pt')
    print(f"Saved left extractive scores to {output_dir / 'left_extractive_scores.pt'}")
    # Experiment 2a: Verify OCR occurs for second hop
    print("Running experiment 2a: verifying OCR for second hop...")
    logs, right_delta = verify_model_ocr(model, tokenizer, right_dataset, right_dataset_options)
    with open(output_dir / 'second_hop_train.json', 'w') as f:
        json.dump(logs, f)
    print(f"Saved second hop train logs to {output_dir / 'second_hop_train.json'}")

    # Experiment 2b: Compute extractive scores for second hop
    print("Running experiment 2b: computing extractive scores for second hop...")
    right_acc_point = ('The mayor of the city that contains X Y is', 'Grace Miller')
    extractive_scores = pes.compute_extractive_scores_counterfactual(
        model=model,
        tokenizer=tokenizer,
        test_dataset=right_dataset['test'],
        test_options=right_dataset_options['test'],
        delta=right_delta,
        acc_point=right_acc_point,
    )
    torch.save(extractive_scores, output_dir / 'right_extractive_scores.pt')
    print(f"Saved right extractive scores to {output_dir / 'right_extractive_scores.pt'}")
    # Experiment 3: Layer freezing
    print("Running experiment 3: layer freezing...")
    freeze_results = evaluate_layer_freezing(
        model=model,
        tokenizer=tokenizer,
        left_dataset=left_dataset,
        left_dataset_options=left_dataset_options,
        right_dataset=right_dataset,
        right_dataset_options=right_dataset_options,
        left_delta=left_delta,
        right_delta=right_delta,
    )
    with open(output_dir / 'layer_freezing.json', 'w') as f:
        json.dump(freeze_results, f)
    print(f"Saved layer freezing results to {output_dir / 'layer_freezing.json'}")
    del left_delta, right_delta

    # Experiment 4: Daxwug data ordering
    print("Running experiment 4: data ordering...")
    left_daxwug_dataset, left_daxwug_dataset_2, left_daxwug_options = daxwug()
    data_ordering_results = get_all_data_ordering_results(
        model=model,
        tokenizer=tokenizer,
        left_dataset=left_daxwug_dataset,
        left_dataset_options=left_daxwug_options,
    )
    with open(output_dir / 'data_ordering_logs.json', 'w') as f:
        json.dump(data_ordering_results, f)
    print(f"Saved data ordering results to {output_dir / 'data_ordering_logs.json'}")
    
    # Experiment 5: gradient grafting
    print("Running experiment 5: gradient grafting...")
    gradient_grafting_results = gradient_grafting(
        model=model,
        tokenizer=tokenizer,
        left_dataset=left_daxwug_dataset,
        left_dataset_options=left_daxwug_options,
    )
    with open(output_dir / 'grafting.json', 'w') as f:
        json.dump(gradient_grafting_results, f)
    print(f"Saved grafting results to {output_dir / 'grafting.json'}")

    # Experiment 6: Localization

    print("Running Experiment 6: localization...")
    localization_results = compute_localization_scores(
        model=model,
        tokenizer=tokenizer,
        left_dataset=left_daxwug_dataset,
        left_dataset_options=left_daxwug_options,
    )
    torch.save(localization_results, output_dir / 'grafting_correlations.pt')
    print(f"Saved grafting correlations to {output_dir / 'grafting_correlations.pt'}")

if __name__ == "__main__":
    main(ROOT / "results")
