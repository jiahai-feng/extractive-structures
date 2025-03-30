import math
from functools import partial
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import extractive_structures.gradients as pg
from contextlib import contextmanager, nullcontext

from exults.tok_utils import SliceRange
import exults.train_utils as tru

from tqdm import trange


@contextmanager
def update_model(model, update_step, alpha=1):
    """
    Warning: update_step should not be changed while the context manager is still running!
    """
    update_dict = dict(update_step)
    updated_keys = set()
    try:
        for name, param in model.named_parameters():
            if name in update_dict:
                param.detach().add_(update_dict[name].to(param.device), alpha=alpha)
                updated_keys.add(name)
        yield
    finally:
        for name, param in model.named_parameters():
            if name in updated_keys:
                param.detach().sub_(update_dict[name].to(param.device), alpha=alpha)


def update_model_(model, update_step, alpha=1):
    """
    Not a context manager. Simply modifies the model in place.
    """
    update_dict = dict(update_step)
    for name, param in model.named_parameters():
        if name in update_dict:
            param.detach().add_(update_dict[name], alpha=alpha)


@contextmanager
def collate_model(model, device=None):
    """
    Yields the model weight diff.
    Careful: during the manager lifetime, the yielded dict instead stores
    a copy of the original weights, and should not be changed
    """
    model_dict = dict()
    try:
        for name, param in model.named_parameters():
            model_dict[name] = param.detach().clone()
            if device is not None:
                model_dict[name] = model_dict[name].to(device)
        yield model_dict
    finally:
        with torch.no_grad():
            for name, param in model.named_parameters():
                t = param.detach().clone()
                param.copy_(model_dict[name].to(param.device))
                t.sub_(param)
                del model_dict[name]
                model_dict[name] = t.to(device) if device is not None else t


def train_opt(
    *,
    train_points,
    model,
    tokenizer,
    optim_config,
    sched_config={"name": "constant"},
    epochs,
    trainable_params=None,
    batch_size=None,
    seed=None,
    logger=None,
    eval_fn=None,
):
    """
    Unlike train_model, this isn't a context manager.
    This ends up being a reimplementation of finetune.finetune
    """
    if trainable_params is None:
        trainable_params = [name for name, param in model.named_parameters()]
    param_dict = dict(model.named_parameters())
    trainable_params = [
        {
            "param_names": trainable_params,
            "params": [param_dict[name] for name in trainable_params],
        }
    ]
    if batch_size is None:
        batch_size = len(train_points)
    if seed is not None:
        rng = np.random.default_rng(seed)
    if optim_config["name"] == "adam":

        @dataclass
        class AdamConfig:
            name: str
            lr: float

        optim_config = AdamConfig(**optim_config)
        optim = torch.optim.Adam(trainable_params, lr=optim_config.lr)
    elif optim_config["name"] == "adamw":

        @dataclass
        class AdamWConfig:
            name: str
            lr: float

        optim_config = AdamWConfig(**optim_config)
        optim = torch.optim.Adam(trainable_params, lr=optim_config.lr)
    elif optim_config["name"] == "sgd":

        @dataclass
        class SGDConfig:
            name: str
            lr: float

        optim_config = SGDConfig(**optim_config)
        optim = torch.optim.SGD(trainable_params, lr=optim_config.lr)
    else:
        raise ValueError(f'Unknown optimizer {optim_config["name"]}')

    if sched_config["name"] == "linear":
        num_train_steps = math.ceil(len(train_points) / batch_size) * epochs
        sched = torch.optim.lr_scheduler.LambdaLR(
            optim,
            partial(
                tru.linear_scheduler_with_warmup,
                warmup_steps=0.1 * num_train_steps,
                max_steps=num_train_steps,
                initial_factor=1e-4,
                final_factor=0.0,
            ),
        )
    elif sched_config["name"] == "constant":
        sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda x: 1.0)
    else:
        raise ValueError(f'Unknown scheduler {sched_config["name"]}')

    for i in trange(epochs):
        if seed is not None:
            train_points = rng.permutation(train_points)
        if eval_fn is not None:
            eval_fn(i)
        for sl in SliceRange(0, len(train_points), batch_size):
            tokens = pg.tokenize_and_mask_batch(train_points[sl], tokenizer)
            loss = model.get_loss(**tokens)
            optim.zero_grad()
            loss.backward()
            optim.step()
            sched.step()
            if logger is not None:
                logger.append({"loss": loss.item()})

    if eval_fn is not None:
        eval_fn(epochs)


@contextmanager
def train_model(
    train_points,
    model,
    tokenizer,
    adam_denoms,
    lr_factor=30.0,
    steps=1,
    logger=None,
    batch_size=None,
    seed=None,
):
    """
    Yields the model weight diff
    """
    lr = lr_factor * 3e-4 * 0.1 / 1024 / 2048  # lr, sched, batch size, seq length
    premise_step = None
    if batch_size is None:
        batch_size = len(train_points)

    if seed is not None:
        rng = np.random.default_rng(seed)
    for i in range(steps):
        if seed is not None:
            train_points = rng.permutation(train_points)
        for sl in SliceRange(0, len(train_points), batch_size):
            with (
                update_model(model, premise_step, -lr)
                if premise_step is not None
                else nullcontext()
            ):
                next_premise_grad, loss = pg.get_gradients(
                    train_points[sl], model, tokenizer, with_loss=True
                )
                if logger is not None:
                    logger.append({"loss": loss.item()})
                del loss
                next_premise_step = pg.get_pc_gradients_(
                    next_premise_grad, "adamgd", adam_denoms
                )
            if premise_step is not None:
                pg.dmap(lambda x, y: x.add_(y), premise_step, next_premise_step)
            else:
                premise_step = next_premise_step
            del next_premise_step, next_premise_grad
    pg.dmap(lambda x: x.mul_(-lr), premise_step)
    with update_model(model, premise_step):
        yield premise_step


def gc_collect():
    import gc

    gc.collect()
    # torch.cuda.empty_cache()
    # apparently empty cache only helps other processes
