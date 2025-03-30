from dataclasses import dataclass
from typing import List, Union
from string import Formatter
from collections import defaultdict, namedtuple
import itertools

import numpy as np
import torch

class TrackFormatter(Formatter):
    def format(self, format_string, **kwargs):
        """
        Only accepts keyword arguments.

        Returns:
            formatted string : str
            locations : Dict[str, List[Tuple[int, int]]] - denoting start and end positions (inclusive, exclusive)
        """
        locations = defaultdict(list)
        result = []
        run_length = 0
        for literal_text, field_name, format_spec, conversion in self.parse(
            format_string
        ):
            # output the literal text
            if literal_text:
                result.append(literal_text)
                run_length += len(literal_text)

            if field_name is not None:
                # given the field_name, find the object it references
                #  and the argument it came from
                obj, arg_used = self.get_field(field_name, [], kwargs)

                # do any conversion on the resulting object
                obj = self.convert_field(obj, conversion)

                # format the object and append to the result
                final_str = self.format_field(obj, format_spec)
                locations[arg_used].append(
                    Substring(run_length, run_length + len(final_str))
                )
                result.append(final_str)
                run_length += len(final_str)

        return "".join(result), locations


from dataclasses import dataclass


@dataclass(frozen=True)
class Substring:
    start: int
    end: int

    def __iter__(self):
        return iter((self.start, self.end))

    def __getitem__(self, key):
        if key == 0:
            return self.start
        else:
            return self.end

    def to_slice(self):
        return slice(self.start, self.end)  
    def slice(self):
        return slice(self.start, self.end)    
    def to_list(self):
        return [self.start, self.end]
    
    def __len__(self):
        return self.end - self.start

    def __add__(self, num):
        return Substring(self.start + num, self.end + num)


import bisect
from functools import partial


def recursify(func, dtype=Substring, pred=None):
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, *args, **kwargs):
        if pred(indices):
            return func(indices, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, *args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, *args, **kwargs) for value in indices]
        else:
            raise Exception(f"Unexpected type {type(indices)}")

    return wrapper

from functools import partial

def recursify_accessor(func, dtype=Substring, pred=None):
    '''
    Returns a function that recursively traverses a data structure,
    until we reach a certain type or when the predicate is satisfied,
    at which case we apply the function, providing it with the object
    that we reached, as well as an accessor function that, when applied
    to the same data structure, will return the object that we reached.

    This is useful for traversing multiple data structures of the same 
    keys, e.g. zipping together hierarchical data structures.
    '''
    if pred is None:
        pred = lambda x: isinstance(x, dtype)

    def wrapper(indices, accessor, *args, **kwargs):
        if pred(indices):
            return func(indices, accessor, *args, **kwargs)
        elif isinstance(indices, dict):
            return {
                key: wrapper(value, lambda x: accessor(x)[key],*args, **kwargs) for key, value in indices.items()
            }
        elif isinstance(indices, list):
            return [wrapper(value, lambda x: accessor(x)[i], *args, **kwargs) for i, value in enumerate(indices)]
        else:
            raise Exception(f"Unexpected type {type(indices)}")

    return partial(wrapper, accessor=lambda x: x)

def zip_map(func, *structs, dtype=Substring, pred=None):
    return recursify_accessor(
        lambda val, accessor: func(val, *[accessor(struct) for struct in structs[1:]]),
        dtype=dtype,
        pred=pred
    )(structs[0])



def rotate(func, dtype=Substring, pred=None, depth=None):
    """
    Rotates "List ... x" into "... List x", and calls func on List x

    Returns:
        ... func(List x)
    """
    if pred is None:
        if depth is not None:
            pred = lambda x, cur_depth: cur_depth == depth
        else:
            pred = lambda x, depth: isinstance(x, dtype)

    def wrapper(indices, cur_depth, *args, **kwargs):
        assert isinstance(indices, list)
        child = indices[0]
        if pred(child, cur_depth):
            return func(indices, *args, **kwargs)
        elif isinstance(child, dict):
            return {
                key: wrapper([sib[key] for sib in indices], cur_depth+1, *args, **kwargs)
                for key in child.keys()
            }

        elif isinstance(child, list):
            return [
                wrapper([sib[key] for sib in indices], cur_depth+1, *args, **kwargs)
                for key in range(len(child))
            ]
        else:
            raise Exception(f"Unexpected type {type(child)}")
    return partial(wrapper, cur_depth=0)

@recursify
def recursive_align_tokens(indices, offset_mapping):
    '''
    Get the token spans corresponding to the string spans (`indices`) by 
    recursively traversing down the string spans using `recursify`.

    Should output a Substring, which is a pair of indices (inclusive, exclusive)
    '''
    # it's unclear what conventions offset_mapping uses
    # but we can say for sure that starting indices are inclusive
    # but my indices are inclusive, exclusive
    # When indices is empty string, we return empty indices (i.e. start = end)
    assert isinstance(indices, Substring)
    start, end = indices
    # bisect_right gives the first token that starts strictly after the search
    tok_start = bisect.bisect_right([x for x, _ in offset_mapping], start) - 1
    tok_end = bisect.bisect_right([x for x, _ in offset_mapping], end - 1) - 1
    if start == end:
        return Substring(tok_start, tok_start)
    else:
        return Substring(tok_start, tok_end + 1)

def get_token_spans(str_spans, offset_mapping, attention_mask):
    '''
    Get the token spans corresponding to the string spans, after accounting for
    attention_mask. This is a wrapper around recursive_align_tokens,
    i.e. str_spans has the same meaning as indices in recursive_align_tokens.

    str_spans : TreeLike[Substring] - output from TrackFormatter
    offset_mapping : LongTensor[seq, 2]
    attention_mask : BoolTensor[seq]

    Note that offset_mapping and attention_mask are the rows corresponding to
    the particular batch for str_spans.
    '''
    token_spans = recursive_align_tokens(str_spans, offset_mapping[attention_mask.bool()])
    token_positions = torch.arange(attention_mask.shape[0])[attention_mask.bool()]
    @recursify
    def compute_token_span(str_span):
        assert isinstance(str_span, Substring)
        target_start, target_end = str_span
        target_start = (
            token_positions[target_start]
            if target_start < len(token_positions)
            else token_positions[-1] + 1
        )
        target_end = (
            token_positions[target_end]
            if target_end < len(token_positions)
            else token_positions[-1] + 1
        )
        return Substring(target_start, target_end)
    return compute_token_span(token_spans)

def verify_token_span(token_span, str_span, string, input_ids, tokenizer):
    if tokenizer.decode(input_ids[token_span.to_slice()]).strip() != string[str_span.to_slice()].strip():
        raise ValueError(f'Token span {token_span} does not correspond to substring {string[str_span.to_slice()]} of string "{string}"[{str_span}]')
def pretty_print_logits(tokenizer, logits):
    assert len(logits.shape) == 1
    top_k = 10
    token_probs = logits.softmax(-1)
    sorted_token_probs, sorted_token_values = token_probs.sort(descending=True)
    # String formatting syntax - the first number gives the number of characters to pad to, the second number gives the number of decimal places.
    # rprint gives rich text printing
    for i in range(top_k):
        print(
            f"Top {i}th token. Logit: {logits[sorted_token_values[i]].item():5.2f} Prob: {sorted_token_probs[i].item():6.2%} Token: |{tokenizer.decode([sorted_token_values[i]])}|"
        )


from contextlib import contextmanager

@contextmanager
def set_padding_side(tokenizer, padding_side):
    old_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield
    finally:
        tokenizer.padding_side = old_padding_side

from collections.abc import Sequence

class SliceRange(Sequence):
    def __init__(self, start, stop, batch_size):
        self.start = start
        self.stop = stop
        self.batch_size = batch_size
    def __getitem__(self, index):
        start = self.start + index * self.batch_size
        stop = min(self.stop, start + self.batch_size)
        if start >= self.stop:
            raise IndexError
        return slice(start, stop)
    def __len__(self) -> int:
        return (self.stop - self.start + self.batch_size - 1) // self.batch_size
    

@dataclass
class Beam:
    cache: any
    new_tokens: torch.LongTensor # [seq]
    log_probs: torch.FloatTensor # [seq]
@torch.no_grad()
def beam_search(
    run_with_cache,
    input_ids,
    attention_mask,
    beam_width=5,
    max_length=32,
    stop_token=None,
):
    """
    Simple model agnostic beam search. In hindsight, should have used olmo's beam search,
    which is also model agnostic.

    To interface with the model, supply `run_with_cache`.
    `run_with_cache` should be a function that takes in input_ids, attention_mask, and cache,
    and returns logits and new_cache.

    The cache should be a list (batch) of arbitrary objects or None.

    See `olmo_run_with_cache` for an example of how to interface with olmo.

    Args:
        run_with_cache : Callable[[
            input_ids: LongTensor[batch, seq], 
            attention_mask: BoolTensor[batch, seq],
            cache: List[Optional[any]]
        ] -> Tuple[logits, cache]
        input_ids : LongTensor[batch, seq]
        attention_mask : BoolTensor[batch, seq]
        beam_width : int
        temperature : float
        max_length : int
        stop_token : int

    Returns:
        List[List[torch.LongTensor]] : List of new tokens for each batch
        List[List[torch.FloatTensor]] : List of log probabilities for each batch
    """
    def check_monotonic(mask):
        one_before = mask.long().cumsum(1) > 0
        return not (one_before & ~mask).any()
    assert check_monotonic(attention_mask), "Attention mask must be monotonic. Did you forget to pad left?"

    
    batch_size, init_seq_length = input_ids.shape
    batch_beams = [
        [Beam(None, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.float))]
        for _ in range(batch_size)
    ]
    batch_terminal_beams = [[] for _ in range(batch_size)]
    for token_cnt in range(max_length):
        packed_input_ids = torch.stack([
            torch.cat([input_ids[i], beam.new_tokens], dim=0)
            for i, beams in enumerate(batch_beams)
            for beam in beams
        ])
        packed_attention_mask = torch.stack([
            torch.cat([attention_mask[i], torch.ones_like(beam.new_tokens, dtype=torch.bool)], dim=0)
            for i, beams in enumerate(batch_beams)
            for beam in beams
        ])
        packed_cache = [
            beam.cache
            for beams in batch_beams
            for beam in beams
        ]
        new_logits, new_cache = run_with_cache(packed_input_ids, packed_attention_mask, packed_cache)
        beam_sizes = np.cumsum([len(beams) for beams in batch_beams])
        for i, beams in enumerate(batch_beams):
            local_logits = new_logits[beam_sizes[i] - len(beams) : beam_sizes[i]]
            local_cache = new_cache[beam_sizes[i] - len(beams) : beam_sizes[i]]
            new_beams = []
            for beam, logits, cache in zip(beams, local_logits, local_cache, strict=True):
                probs = logits[-1, :].softmax(dim=-1)
                top_k_probs, top_k_indices = probs.topk(beam_width)
                for prob, index in zip(top_k_probs, top_k_indices):
                    new_beams.append(
                        Beam(
                            cache=cache,
                            new_tokens=torch.cat([beam.new_tokens, index.unsqueeze(0)], dim=0),
                            log_probs=torch.cat([beam.log_probs, prob.unsqueeze(0)], dim=0),
                        )
                    )
            stopped_beams = [beam for beam in new_beams if beam.new_tokens[-1] == stop_token]
            live_beams = [beam for beam in new_beams if beam.new_tokens[-1] != stop_token]
            batch_terminal_beams[i].extend(stopped_beams)
            live_beams.sort(key=lambda x: x.log_probs.sum(), reverse=True)
            batch_beams[i] = live_beams[:beam_width]
    for terminal_beams in batch_terminal_beams:
        terminal_beams.sort(key=lambda x: x.log_probs.sum(), reverse=True)
    return [
        [beam.new_tokens for beam in beams]
        for beams in batch_terminal_beams
    ], [
        [beam.log_probs for beam in beams]
        for beams in batch_terminal_beams
    ]

def olmo_run_with_cache(input_ids, attention_mask, cache, *, model):
    '''
    `run_with_cache` function for beam search. This function allows olmo to be
    used with our beam search implementation. To use, pass
    `run_with_cache=partial(olmo_run_with_cache, model=model)` to beam_search.

    
    Example:
        ```
        batch_tokens, batch_log_probs = etu.beam_search(
            run_with_cache=partial(etu.olmo_run_with_cache, model=model),
            **tokenizer(["Hello"], return_tensors='pt'),
            stop_token=187 # newline token
        )


        for tokens in batch_tokens:
            for seq in tokens:
                print(tokenizer.decode(seq))
        ```

    Implementation notes:

    Olmo's cache is a list (layer) of tuples (key, value), where 
    key and value are both tensors of shape [batch, heads, seq, dim].

    We want to convert it into a list (batch) of list (layer) of
    tuples (key, value), where key and value are both tensors of shape 
    [layer, seq, dim].
    '''
    batch_size, _ = input_ids.shape
    
    if any([c is None for c in cache]):
        outputs = model(input_ids, attention_mask=attention_mask, use_cache=True)
    else:
        olmo_cache = [
            (
                torch.stack([batch_cache[layer][0] for batch_cache in cache]),
                torch.stack([batch_cache[layer][1] for batch_cache in cache])
            )
            for layer in range(model.config.n_layers)
        ]
        cached_length = olmo_cache[0][0].shape[2] # key has shape [batch, layer, seq, dim]

        # note: olmo expects input_ids to be truncated (if we're using cache),
        # but not the attention mask!
        outputs = model(input_ids[:, cached_length:], attention_mask=attention_mask, past_key_values=olmo_cache, use_cache=True)
        
    new_cache = [
        [(key[batch], value[batch]) for key, value in outputs.attn_key_values]
        for batch in range(batch_size)
    ]
    return outputs.logits, new_cache
    