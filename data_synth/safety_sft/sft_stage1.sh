#!/usr/bin/env bash
# TCG-RL cold-start SFT (supervised fine-tuning stage).
#
# Model  : Qwen2.5-VL-7B-Instruct  (student)
# Data   : distilled + filtered by convert_to_swift_sft.py
#            train/val: produced by the teacher-distillation pipeline
#            each sample already carries its own <system>, so we do NOT pass
#            --system on the CLI (that would override the per-sample system).
# Recipe : mirrors the Thyme SFT recipe: full-parameter FT, freeze_vit,
#          max_length=10240, 3 epochs, ga=16, Deepspeed ZeRO, flash_attn, bf16.
# Compute: single-node 8xGPU (change NPROC_PER_NODE/CUDA_VISIBLE_DEVICES as needed)
#
# Usage:
#   MODEL=./models/Qwen2.5-VL-7B-Instruct bash sft_stage1.sh
#   nohup bash sft_stage1.sh > /dev/null 2>&1 &     # background
#
# The model, output_dir and dataset paths can be overridden via env:
#   MODEL=... OUTPUT_DIR=... DATASET=... bash sft_stage1.sh

set -euo pipefail

cd "$(dirname "$0")"


# ------------ Conda env ------------
# ms-swift launches distributed workers via `sys.executable`, so the python
# that runs *this script* must already be the one with this repo's ms-swift
# on PYTHONPATH. Set CONDA_SH/CONDA_ENV to have the launcher activate it.
#   CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh CONDA_ENV=thyme bash sft_stage1.sh
CONDA_SH=${CONDA_SH:-}
CONDA_ENV=${CONDA_ENV:-thyme}
if [[ -n "${CONDA_SH}" && -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[launcher] conda env : ${CONDA_ENV}"
    echo "[launcher] python    : $(which python)"
    echo "[launcher] version   : $(python --version 2>&1)"
else
    echo "[launcher] conda     : not activated (CONDA_SH unset/not found)"
    echo "[launcher] python    : $(which python)"
fi

# ------------ Paths (override via env if needed) ------------
MODEL=${MODEL:-./models/Qwen2.5-VL-7B-Instruct}
DATASET=${DATASET:-./output/thyme_safety_sft_train.jsonl}
VAL_DATASET=${VAL_DATASET:-./output/thyme_safety_sft_val.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./output/sft_stage1_qwen25vl7b_safety}

if [[ "${MODEL}" == /* || "${MODEL}" == ./* || "${MODEL}" == ../* || "${MODEL}" == "~"* ]] && [[ ! -e "${MODEL}" ]]; then
    echo "[launcher] ERROR: MODEL=${MODEL} does not exist."
    echo "[launcher]        Set MODEL=/path/to/Qwen2.5-VL-7B-Instruct."
    exit 2
fi
if [[ ! -f "${DATASET}" ]]; then
    echo "[launcher] ERROR: DATASET=${DATASET} does not exist."
    echo "[launcher]        Build it first with build_distill_input.py +"
    echo "[launcher]        run_teacher_distill.py + convert_to_swift_sft.py."
    exit 2
fi

mkdir -p ./logs "${OUTPUT_DIR}"
LOG=./logs/sft_stage1_$(date +%Y%m%d_%H%M%S).log
echo "[launcher] logging to ${LOG}"

# ------------ Runtime knobs ------------
# NOTE: we go through the ms-swift shipped with this Thyme repo, not the
# system-installed one. This is important because our sandbox.py fixes and
# the Thyme --O3 flag live inside this in-tree swift package. The PYTHONPATH
# trick makes `swift sft` resolve to Thyme's swift.cli.
export PYTHONPATH=$(cd ../../ && pwd):${PYTHONPATH:-}

# IMPORTANT: some launchers/schedulers pre-export CUDA_VISIBLE_DEVICES with a
# subset of GPUs (e.g. "0,1,2,3"), which would silently override our default
# and produce assertion errors like:
#   AssertionError: n_gpu: 4, local_world_size: 8
# So we FORCE-set CUDA_VISIBLE_DEVICES here (still allow explicit override by
# the caller with `CUDA_VISIBLE_DEVICES=... bash sft_stage1.sh`).
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Auto-derive nproc_per_node from the (possibly overridden) CUDA_VISIBLE_DEVICES
# so world_size == n_gpu, otherwise swift's is_mp() will assert.
n_visible=$(echo "${CUDA_VISIBLE_DEVICES}" | awk -F',' '{print NF}')
nproc_per_node=${NPROC_PER_NODE:-$n_visible}
export NPROC_PER_NODE=$nproc_per_node
export OMP_NUM_THREADS=8
export MAX_PIXELS=${MAX_PIXELS:-3211264}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-10}

echo "[launcher] CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"
echo "[launcher] NPROC_PER_NODE       = ${NPROC_PER_NODE}"

bsz=${BSZ:-1}
ga=${GA:-$(expr 16 / $bsz)}

# ------------ Memory knobs ------------
# Full-parameter FT of a large VL model does not fit in ZeRO-2 on 95GiB
# cards (bf16 params alone are ~64GiB per rank for a 32B model, plus Adam
# states). Parameters are therefore sharded too -> ZeRO-3. If it still OOMs
# during activation, switch to DS_STAGE=zero3_offload to offload optimizer
# state (and params) to CPU, or raise NPROC_PER_NODE / lower MAX_LEN.
DS_STAGE=${DS_STAGE:-zero3}
MAX_LEN=${MAX_LEN:-10240}
SAVE_LIMIT=${SAVE_LIMIT:-2}

# Reduce fragmentation OOMs; recommended by PyTorch for large-model training.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "[launcher] bsz=${bsz}  ga=${ga}  max_len=${MAX_LEN}  deepspeed=${DS_STAGE}"

# ------------ Launch ------------
python -m swift.cli.main sft \
    --model                    "${MODEL}" \
    --dataset                  "${DATASET}" \
    --val_dataset              "${VAL_DATASET}" \
    --train_type               full \
    --torch_dtype              bfloat16 \
    --num_train_epochs         3 \
    --per_device_train_batch_size $bsz \
    --per_device_eval_batch_size  1 \
    --gradient_accumulation_steps $ga \
    --learning_rate            5e-6 \
    --lr_scheduler_type        cosine \
    --warmup_ratio             0.05 \
    --freeze_vit               true \
    --max_length               ${MAX_LEN} \
    --gradient_checkpointing   true \
    --save_strategy            epoch \
    --save_total_limit         ${SAVE_LIMIT} \
    --eval_strategy            epoch \
    --logging_steps            5 \
    --output_dir               "${OUTPUT_DIR}" \
    --dataloader_num_workers   4 \
    --deepspeed                ${DS_STAGE} \
    --attn_impl                flash_attn \
    --report_to                tensorboard \
    2>&1 | tee "${LOG}"

echo
echo "=== SFT stage-1 done ==="
echo "Best checkpoint under: ${OUTPUT_DIR}"
