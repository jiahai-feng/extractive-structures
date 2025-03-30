from contextlib import contextmanager
from functools import partial

"""
Lightweight library for hooking into PyTorch models.


Usage:
A hook function is a function that takes the pre-module hidden states and a Hook object.
The hook function should return the modified hidden states or None if no modification is needed.
    hook_fn: (pre_state, Hook) -> Optional[pre_state]

Hook objects are dummy objects that carry metadata:
    - name (str): the name of the module
    - module (nn.Module): the module object


Example: 
with hook_model(model, [("module1", hook_fn1), ("module2", hook_fn2)]):
    model(input)

    
where module1 and module2 are names in model.named_modules()
and hook_fn1 and hook_fn2 are functions that take the pre-module hidden states and a Hook object.
"""


def lookup_module(model, hook_name):
    if hook_name == "":
        return model
    d = dict(model.named_modules())
    return d[hook_name] if hook_name in d else None


from functools import wraps


@contextmanager
def hook_model(model, hooks, prepend=False, pre_hook=True):
    """
    Use pre_hook by default, unless pre_hook is set to True.

    If using pre_hooks, we expect hook_fn to have the signature:

    hook_fn(hidden_states, Hook) -> hidden_states or None

    If using normal hooks, we expect hook_fn to have the signature:

    hook_fn(hidden_states, output, Hook) -> output or None

    """
    hook_handles = []

    def flattened_pre_hook_fn(module, args, *, hook_fn, hook_name):
        if len(args) == 1:
            (hidden_states,) = args
        else:
            hidden_states = args

        class Hook:
            name = hook_name
            mod = module

        return hook_fn(hidden_states, Hook)

    def flattened_hook_fn(module, args, output, *, hook_fn, hook_name):
        if len(args) == 1:
            (hidden_states,) = args
        else:
            hidden_states = args

        class Hook:
            name = hook_name
            mod = module

        return hook_fn(hidden_states, output, Hook)

    for hook_name, hook_fn in hooks:
        mod = lookup_module(model, hook_name)
        if mod is None:
            raise Exception(f"No submodule {hook_name} found!")
    try:
        if prepend:
            hooks = hooks[::-1]
        for hook_name, hook_fn in hooks:
            mod = lookup_module(model, hook_name)
            if pre_hook:
                handle = mod.register_forward_pre_hook(
                    wraps(hook_fn)(
                        partial(
                            flattened_pre_hook_fn, hook_fn=hook_fn, hook_name=hook_name
                        )
                    ),
                    prepend=prepend,
                )
            else:
                handle = mod.register_forward_hook(
                    wraps(hook_fn)(
                        partial(flattened_hook_fn, hook_fn=hook_fn, hook_name=hook_name)
                    ),
                    prepend=prepend,
                )
            hook_handles.append(handle)
        yield model
    finally:
        for handle in hook_handles:
            handle.remove()


@contextmanager
def hook_model_bwd(model, hooks, prepend=False):
    """
    Registers a backward prehook.

    Expect hook_fn to have the signature:

    hook_fn(grad_output, Hook) -> grad_output or None

    Where grad_output is the gradient with respect to the output of the module.
    """
    hook_handles = []

    def flattened_hook_fn(module, module_output, *, hook_fn, hook_name):
        (hidden_states,) = module_output

        class Hook:
            name = hook_name
            mod = module

        return hook_fn(hidden_states, Hook)

    for hook_name, hook_fn in hooks:
        mod = lookup_module(model, hook_name)
        if mod is None:
            raise Exception(f"No submodule {hook_name} found!")
    try:
        if prepend:
            hooks = hooks[::-1]
        for hook_name, hook_fn in hooks:
            mod = lookup_module(model, hook_name)
            handle = mod.register_full_backward_pre_hook(
                wraps(hook_fn)(
                    partial(flattened_hook_fn, hook_fn=hook_fn, hook_name=hook_name)
                ),
                prepend=prepend,
            )
            hook_handles.append(handle)
        yield model
    finally:
        for handle in hook_handles:
            handle.remove()
