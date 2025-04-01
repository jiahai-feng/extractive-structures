# Extractive structures

Download models
---
We use the final checkpoint before the annealing phase in olmo-0424. See `https://github.com/jiahai-feng/olmo-es/blob/jiahai/checkpoints/official/OLMo-7B-0424.csv` for list of checkpoints.

The checkpoint links were updated in March 2025. The updated links can be found here: `https://olmo-checkpoints.org/ai2-llm/olmo-medium/xtruaap8/step477000-unsharded/`. See [github issue](https://github.com/allenai/OLMo/issues/796) for more details.

```bash
OLMO_PREANNEAL_DIR=<OLMO_PREANNEAL_DIR>
mkdir -p $OLMO_PREANNEAL_DIR
for file in config.yaml model.pt optim.pt; do
    curl -o $OLMO_PREANNEAL_DIR/$file "https://olmo-checkpoints.org/ai2-llm/olmo-medium/xtruaap8/step477000-unsharded/$file"
done
```

Environment variables
---
Create a `.env` file in the root directory and add the following variables.
```
HF_TOKEN=<HF_TOKEN>
OLMO_PREANNEAL_DIR=<OLMO_PREANNEAL_DIR>
```

Install dependencies
---
Install [uv](https://docs.astral.sh/uv/getting-started/installation/). 

Then, run

```bash
uv sync
```


Main experiments
---
```bash
uv run python -m extractive_structures.scripts.compute_extractive_scores
```
This will save the results to `results/`.

You can visualize the results using `notebooks/paper_plots.ipynb`.


Other models
---
You can sweep across different models and hyperparameters using `extractive_structures/scripts/eval_ocr.py`.

The available models are listed in `extractive_structures/models.py`:

```python
model_tag_dict = {
    "mistral": "mistralai/Mistral-7B-v0.3",
    "qwen": "Qwen/Qwen2-7B",
    "llama": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-2-9b",
    "olmo": "allenai/OLMo-7B-0424-hf",
    "llama_70b": "meta-llama/Meta-Llama-3-70B",
    "gemma_27b": "google/gemma-2-27b",
    "qwen_32b": "Qwen/Qwen2.5-32B",
    "qwen_72b": "Qwen/Qwen2.5-72B",
    "llama_1b": "meta-llama/Llama-3.2-1B"
}
```

To use the script, I recommend using a config yaml file. `paper_experiments` contains a few examples. To run the script, call

```bash
uv run python -m extractive_structures.scripts.eval_ocr --config <path_to_config>
```



