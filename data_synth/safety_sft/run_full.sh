#!/usr/bin/env bash
# Full teacher distillation over a STRATIFIED SUBSAMPLE of merged_train.jsonl.
#
# Plan (方案 B):
#   * Cap each of the 4 subsets at PER_SUBSET_CAP (default 1250) => 5000 questions.
#   * Run NUM_PER_QUESTION=4 trials per question => 20k trajectories.
#   * Expected wall-clock: ~1-2 days on 8xH20 (pilot was 100q x 4 trials in ~1h).
#   * Balanced across harmc / harmp / mami / pridemm so mami's raw 2x majority
#     does not dominate the SFT distribution.
#
# Why not 20278 questions x 4 trials = 81k?
#   * Thyme's original 500k SFT is a MULTI-TASK cold start (crop/rotate/contrast
#     /math/multi-round). Ours is a single-task domain adaptation (safety meme
#     classification) starting from an already-strong Qwen2.5-VL-7B. 5k-15k
#     high-quality samples is the standard regime for that.
#   * After filtering (predicted==gt & sandbox_used & format_ok) the expected
#     yield here is ~10-15k usable trajectories, matching the target regime.
#
# Reliability:
#   * --resume: existing (question_id, trial_idx) pairs in the output jsonl are
#     skipped, so this script is safe to re-run after any interrupt.
#   * OUTPUT paths are distinct from pilot's, so pilot artefacts are untouched.
#
# Usage:
#   TEACHER=/path/to/Qwen2.5-VL-72B-Instruct \
#   QUESTIONS_DIR=/path/to/viscot-harmeme/data/questions \
#       bash run_full.sh                            # foreground
#   nohup bash run_full.sh > /dev/null 2>&1 &       # background (recommended)
#   PER_SUBSET_CAP=2500 bash run_full.sh            # override the cap

set -euo pipefail

cd "$(dirname "$0")"

# Activate the environment that has this repo's ms-swift + vLLM installed.
if [[ -n "${CONDA_SH:-}" && -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV:-thyme}"
fi

TEACHER=${TEACHER:-./models/Qwen2.5-VL-72B-Instruct}
QUESTIONS_DIR=${QUESTIONS_DIR:-${VISCOT_ROOT:-../viscot-harmeme}/data/questions}

PER_SUBSET_CAP=${PER_SUBSET_CAP:-1250}
NUM_PER_QUESTION=${NUM_PER_QUESTION:-4}

INPUT_JSONL=./output/merged_train_sub${PER_SUBSET_CAP}.jsonl
OUTPUT_JSONL=./output/full_trajectories.jsonl
TEMP_DIR=./_sandbox_tmp/full
LOG=./logs/full_$(date +%Y%m%d_%H%M%S).log

mkdir -p ./logs "${TEMP_DIR}" ./output

# 1) Build a per-subset-balanced input file. Always rebuild so that changing
#    PER_SUBSET_CAP takes effect immediately; the RNG seed keeps it reproducible.
echo "[launcher] building stratified input at ${INPUT_JSONL} (cap=${PER_SUBSET_CAP} per subset)"
python build_distill_input.py \
    --questions_dir  "${QUESTIONS_DIR}" \
    --out_path       "${INPUT_JSONL}" \
    --per_subset_cap "${PER_SUBSET_CAP}" \
    --seed 42

# 2) Full distillation. We do NOT pass --limit here; the input file itself is
#    already the intended size (4 * PER_SUBSET_CAP).
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_PIXELS=3211264
export FPS_MAX_FRAMES=10
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "[launcher] logging to ${LOG}"

python run_teacher_distill.py \
    --teacher_path           "${TEACHER}" \
    --input_jsonl            "${INPUT_JSONL}" \
    --output_jsonl           "${OUTPUT_JSONL}" \
    --temp_dir               "${TEMP_DIR}" \
    --prompt_path            ./prompt_safety.txt \
    --num_per_question       "${NUM_PER_QUESTION}" \
    --temperature            0.9 \
    --top_p                  0.95 \
    --max_iterations         5 \
    --max_new_tokens         2048 \
    --tensor_parallel_size   8 \
    --max_model_len          16384 \
    --gpu_memory_utilization 0.85 \
    --max_pixels             3211264 \
    --limit                  0 \
    --resume \
    2>&1 | tee "${LOG}"

# 3) Quality report on the full run (uses the same thresholds as pilot).
python pilot_check.py \
    --in_path "${OUTPUT_JSONL}" \
    --out_md  ./output/full_stats.md

echo
echo "=== full distillation done ==="
echo "Inspect ./output/full_stats.md"
