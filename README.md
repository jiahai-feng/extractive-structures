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



