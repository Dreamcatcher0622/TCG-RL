# ---------------------------------------------------------------------------
# Cheat sheet: build the RL dataset, then launch TCG-RL GRPO training.
# Replace the placeholders with paths on your machine (see README.md).
# ---------------------------------------------------------------------------
cd /path/to/Thyme_code/data_synth/safety_sft

# Optional: activate the conda env that holds this repo's ms-swift + vLLM.
source /path/to/miniconda3/etc/profile.d/conda.sh
conda activate thyme

# 1) RL dataset: all four subsets, minus the question_ids already seen by SFT.
python build_rl_dataset.py \
  --questions_dir "${VISCOT_ROOT}/data/questions" \
  --system_prompt ./prompt_safety_rl.txt \
  --out_path      ./output/thyme_safety_rl_train.jsonl \
  --exclude_sft   ./output/thyme_safety_sft_train.jsonl \
  --seed          42

# 2) Launch the GRPO run (uses the dataset above by default).
#    Foreground first to check that rollouts/rewards behave, then background.
MODEL=/path/to/cold-start-sft-checkpoint bash rl.sh

# or
nohup bash rl.sh > /dev/null 2>&1 &
