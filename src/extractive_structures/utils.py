import numpy as np
import torch
import extractive_structures.gradients as pg

import extractive_structures.masking_utils as pmu

import exults.tok_utils as tu

import pandas as pd
import numpy as np

from contextlib import nullcontext
from exults.tensorial import Long


def get_rank(delta, test_points, options, model, tokenizer):
    with tu.set_padding_side(tokenizer, "right"):
        option_tokens = tokenizer(
            [" " + option for option in options], return_tensors="pt", padding=True
        )["input_ids"]
        has_bos = tokenizer.bos_token is not None
        candidate_tokens = torch.unique(option_tokens[:, int(has_bos)])
        assert len(options) == len(candidate_tokens)
    # print([tokenizer.decode(tok) for tok in candidate_tokens])
    with pmu.update_model(model, delta) if delta is not None else nullcontext():
        inputs = pg.tokenize_and_mask_batch(test_points, tokenizer)
        logits = model(inputs["input_ids"]).logits

        all_accs = []

        for row in range(logits.shape[0]):
            question_end = torch.argmax((inputs["labels"][row] != -100).long()) - 1
            answer_logits = logits[row, question_end, candidate_tokens]
            assert answer_logits.shape == candidate_tokens.shape
            ranks = torch.argsort(torch.argsort(answer_logits, descending=True))
            (correct_label,) = torch.nonzero(
                candidate_tokens == inputs["labels"][row, question_end + 1],
                as_tuple=True,
            )
            assert correct_label.shape == (1,)
            correct_label = correct_label[0].item()

            correct_rank = ranks[correct_label]
            all_accs.append(correct_rank.item())

    return np.mean(all_accs)


def mean_logit_loss(model, test_point, tokenizer, options):
    inputs = pg.tokenize_and_mask_batch([test_point], tokenizer)
    logits = model(
        input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]
    ).logits

    with tu.set_padding_side(tokenizer, "right"):
        option_tokens = tokenizer(
            [" " + option for option in options], return_tensors="pt", padding=True
        )["input_ids"]
        has_bos = tokenizer.bos_token is not None
        candidate_tokens = torch.unique(option_tokens[:, int(has_bos)])
        assert len(options) == len(candidate_tokens)

    question_end = torch.argmax((inputs["labels"][0] != -100).long()) - 1
    filtered_logits = logits[0, question_end, candidate_tokens]
    correct_token = (candidate_tokens == inputs["labels"][0, question_end + 1]).nonzero(
        as_tuple=False
    )[0, 0]
    return -filtered_logits[correct_token] + filtered_logits.mean()


# def next_token(s):
#     inputs = tokenizer([s], return_tensors='pt')
#     logits = model(inputs.input_ids).logits
#     pretty_print_logits(tokenizer, logits[0, -1, :])
# def plot_logs(logs):
#     sns.lineplot(pd.DataFrame(logs))
# def greedy_acc(delta, test_points):
#     with pmu.update_model(model, delta) if delta is not None else nullcontext():
#         inputs = pg.tokenize_and_mask_batch(test_points, tokenizer)
#         logits = model(inputs['input_ids']).logits
#         greedy = logits.argmax(dim=-1)
#         acc = torch.all(torch.where(inputs['labels'][:, 1:] != -100, greedy[:, :-1], -100) == inputs['labels'][:, 1:], dim=1)
#     return acc.float().mean().item()
# def get_loss(delta, test_points):
#     # get first token loss
#     with pmu.update_model(model, delta) if delta is not None else nullcontext():
#         inputs = pg.tokenize_and_mask_batch(test_points, tokenizer)
#         losses = model.get_loss(**inputs, reduction='none')
#         question_end = torch.argmax((inputs['labels'] != -100).long(), dim=-1) - 1
#         return losses[torch.arange(losses.shape[0]), question_end].mean().item()


# def constrained_greedy_acc(delta, test_points, options):
#     with tu.set_padding_side(tokenizer, 'right'):
#         option_tokens = tokenizer([' ' + option for option in options], return_tensors='pt', padding=True)['input_ids']
#         candidate_tokens = [
#             torch.unique(option_tokens[:, i])
#             for i in range(option_tokens.shape[1])
#         ]
#         has_bos = tokenizer.bos_token is not None
#         if has_bos:
#             candidate_tokens = candidate_tokens[1:]

#     with pmu.update_model(model, delta) if delta is not None else nullcontext():
#         inputs = pg.tokenize_and_mask_batch(test_points, tokenizer)
#         logits = model(inputs['input_ids']).logits

#         token_mask = torch.zeros((option_tokens.shape[1], logits.shape[-1]), dtype=bool)
#         for i, tokens in enumerate(candidate_tokens):
#             token_mask[i, tokens] = True

#         all_accs = []
#         for row in range(logits.shape[0]):
#             question_end = torch.argmax((inputs['labels'][row] != -100).long()) - 1
#             i = 0
#             acc = []
#             while question_end + i + 1 < logits.shape[1] and inputs['labels'][row, question_end + i + 1] != -100:
#                 masked_logits = torch.where(
#                     token_mask[i],
#                     logits[row, question_end + i],
#                     -1e9
#                 )
#                 top = masked_logits.argmax()
#                 acc.append(top == inputs['labels'][row, question_end + i + 1])
#                 i = i + 1
#             all_accs.append(torch.all(torch.tensor(acc)).float())


#     return torch.mean(torch.tensor(all_accs)).item()


def get_layer_names(model, layers):
    return [
        model.Path(name, layer).to_str()
        for name in model.Path.mlp_names + model.Path.attn_names
        for layer in layers
    ]
