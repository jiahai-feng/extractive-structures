from extractive_structures.datasets import Dataset, Options
from extractive_structures.utils import get_rank, get_layer_names
import extractive_structures.masking_utils as pmu

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