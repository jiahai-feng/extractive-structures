from math import cos, pi
from functools import partial
def linear_scheduler(step, max_steps, initial_factor, final_factor):
    alpha = step / max_steps
    return alpha * final_factor + (1-alpha) * initial_factor
    
def cosine_scheduler(step, *, warmup_steps, max_steps, final_factor, initial_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.)
    elif step >= max_steps:
        return final_factor
    else:
        step = step - warmup_steps
        max_steps = max_steps - warmup_steps
        return final_factor + (1. - final_factor) * (1 + cos(pi * step / max_steps)) / 2
    
def linear_scheduler_with_warmup(step, *, warmup_steps, max_steps, initial_factor, final_factor):
    if step < warmup_steps:
        return linear_scheduler(step, warmup_steps, initial_factor, 1.)
    elif step >= max_steps:
        return final_factor
    else:
        return linear_scheduler(step - warmup_steps, max_steps - warmup_steps, 1., final_factor)

'''
Usage:

sched = torch.optim.lr_scheduler.LambdaLR(optim, partial(
    cosine_scheduler, 
    warmup_steps=warmup_steps,
    max_steps=num_total_steps,
    initial_factor=1e-4,
    final_factor=0.
))
'''