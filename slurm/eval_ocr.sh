# List of config files
CONFIG_FILES=(
    "/home/ubuntu/extractive_structures/paper_experiments/sweep_all/full_gemma_27b_1e-05_12.yaml"
    "/home/ubuntu/extractive_structures/paper_experiments/sweep_all/full_gemma_27b_1e-05_16.yaml"
    "/home/ubuntu/extractive_structures/paper_experiments/sweep_all/full_gemma_27b_3e-05_12.yaml"
    "/home/ubuntu/extractive_structures/paper_experiments/sweep_all/full_gemma_27b_3e-05_16.yaml"
)

# Loop through each config file and run the command
for CONFIG_FILE in "${CONFIG_FILES[@]}"; do
    echo "Processing config file: $CONFIG_FILE"
    uv run python -m extractive_structures.scripts.eval_ocr \
        --config "$CONFIG_FILE"
done