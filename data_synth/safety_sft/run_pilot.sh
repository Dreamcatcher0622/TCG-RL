#!/usr/bin/env bash
# One-shot pilot: 100 questions x 4 trials = 400 trajectories on the
# Qwen2.5-VL-72B teacher.  Should take ~30-60 minutes on 8xH20.
#
#   bash run_pilot.sh
#
# After it finishes, inspect ./output/pilot_stats.md and decide whether
# to launch the full distillation (run_full.sh).

set -euo pipefail

cd "$(dirname "$0")"

# Override from the shell, e.g.
#   TEACHER=/path/to/Qwen2.5-VL-72B-Instruct \
#   QUESTIONS_DIR=/path/to/viscot-harmeme/data/questions bash run_pilot.sh
TEACHER=${TEACHER:-./models/Qwen2.5-VL-72B-Instruct}
QUESTIONS_DIR=${QUESTIONS_DIR:-${VISCOT_ROOT:-../viscot-harmeme}/data/questions}

mkdir -p ./output ./logs ./_sandbox_tmp/pilot

# 1) build merged input (idempotent / fast)
python build_distill_input.py \
    --questions_dir "${QUESTIONS_DIR}" \
    --out_path      ./output/merged_train.jsonl \
    --seed 42

# 2) run teacher distillation on the first 100 questions only
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export MAX_PIXELS=3211264
export FPS_MAX_FRAMES=10
export VLLM_WORKER_MULTIPROC_METHOD=spawn

python run_teacher_distill.py \
    --teacher_path        "${TEACHER}" \
    --input_jsonl         ./output/merged_train.jsonl \
    --output_jsonl        ./output/pilot_trajectories.jsonl \
    --temp_dir            ./_sandbox_tmp/pilot \
    --prompt_path         ./prompt_safety.txt \
    --num_per_question    4 \
    --temperature         0.9 \
    --top_p               0.95 \
    --max_iterations      5 \
    --max_new_tokens      2048 \
    --tensor_parallel_size 8 \
    --max_model_len       16384 \
    --gpu_memory_utilization 0.85 \
    --max_pixels          3211264 \
    --limit               100 \
    --resume \
    2>&1 | tee ./logs/pilot_$(date +%Y%m%d_%H%M%S).log

# 3) summarize
python pilot_check.py \
    --in_path ./output/pilot_trajectories.jsonl \
    --out_md  ./output/pilot_stats.md

echo
echo "=== pilot done ==="
echo "Inspect ./output/pilot_stats.md"
