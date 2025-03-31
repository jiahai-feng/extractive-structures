#!/bin/bash
#SBATCH --output=jobs/eval_ocr-%j.out
#SBATCH -c 8


cd /data/fjiahai/extractive_structures

uv run python -m extractive_structures.scripts.eval_ocr \
    --config $CONFIG_FILE