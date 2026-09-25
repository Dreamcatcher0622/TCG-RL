#!/usr/bin/env bash
# TCG-RL (Tool-Contrastive Grounding RL) GRPO training script.
#
# Model  : default = Qwen2.5-VL-7B-Instruct; point MODEL at your cold-start
#          SFT checkpoint for the paper's recipe.
# Data   : ./output/thyme_safety_rl_train.jsonl (built by build_rl_dataset.py).
# Reward : rule-based fmt / vqa / cst ORMs registered in ./safety_rm.py,
#          plus the optional faithfulness and tool-contrastive ORMs.
# Recipe : mirrors the original Thyme scripts/rl.sh, adapted for single-node 8xGPU
#          (no 72B judge server, no deepspeed multi-host launcher).
#
# Usage:
#   bash rl.sh                                            # foreground
#   nohup bash rl.sh > /dev/null 2>&1 &                   # background
#
# Env overrides (all optional):
#   MODEL=... DATASET=... OUTPUT_DIR=...                  # paths
#   RUN_NAME=faith_v1                                     # tag appended to
#                                                         #   OUTPUT_DIR and LOG
#                                                         #   file name (default
#                                                         #   empty = no tag)
#   FMT_WEIGHT=0.5 VQA_WEIGHT=1.0 CST_WEIGHT=0.5          # reward weights
#   NPROC_PER_NODE=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 # compute
#   BSZ=1 GA=5 NUM_GEN=4 EPOCH=2 LR=5e-7 BETA=0.01        # hparams
#   VLLM_MEM=0.20 VLLM_TP=1                               # vLLM knobs
#   TOOL_MODE=none|cf|tfa|offline_cf F_REF_PATH=...        # reward selection
#   RESUME_FROM=./output/rl_grpo_qwen25vl7b_safety/v8-.../checkpoint-700
#                                                         # full resume (optim+sched+rng+step)

set -euo pipefail

cd "$(dirname "$0")"

# ------------ Conda env ------------
# Activate the environment that provides the in-tree ms-swift + vLLM stack.
# Set CONDA_SH (path to conda.sh) and CONDA_ENV (env name) to have this
# script activate it for you; otherwise the `python` already on PATH is used.
#   CONDA_SH=/opt/miniconda3/etc/profile.d/conda.sh CONDA_ENV=thyme bash rl.sh
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
    echo "[launcher] version   : $(python --version 2>&1)"
fi

# ------------ Pre-flight: probe usable GPUs ------------
# Under multi-tenant / K8s GPU sharing we sometimes see nvidia-smi list N
# GPUs but only M<N are actually usable from this container (the rest hang
# on first CUDA call or return "invalid device ordinal"). Probe each GPU
# with a short timeout and derive the true usable list.
if [[ "${SKIP_GPU_PROBE:-0}" != "1" ]]; then
    # By default we IGNORE any pre-existing CUDA_VISIBLE_DEVICES in the shell
    # (which may have been narrowed manually to a subset earlier) and always
    # probe the full 0..7 range. Set RESPECT_CUDA_VISIBLE_DEVICES=1 to opt in
    # to the old behavior of trusting the parent env.
    if [[ "${RESPECT_CUDA_VISIBLE_DEVICES:-0}" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        _CAND_STR="${CUDA_VISIBLE_DEVICES}"
    else
        _CAND_STR="0,1,2,3,4,5,6,7"
    fi
    export CUDA_VISIBLE_DEVICES="${_CAND_STR}"
    IFS=',' read -ra _CANDIDATE_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
    _USABLE_GPUS=()
    echo "[preflight] probing GPUs: ${CUDA_VISIBLE_DEVICES}"
    for _idx in "${!_CANDIDATE_GPUS[@]}"; do
        _gpu_id="${_CANDIDATE_GPUS[$_idx]}"
        if timeout 20 env CUDA_VISIBLE_DEVICES="${_gpu_id}" \
            python -c "import torch; torch.cuda.set_device(0); _ = torch.zeros(1, device='cuda:0'); print('ok')" \
            >/dev/null 2>&1; then
            _USABLE_GPUS+=("${_gpu_id}")
            echo "[preflight]   gpu ${_gpu_id}  OK"
        else
            echo "[preflight]   gpu ${_gpu_id}  UNUSABLE (timeout or CUDA error)"
        fi
    done
    if (( ${#_USABLE_GPUS[@]} == 0 )); then
        echo "[preflight] ERROR: no usable GPUs, aborting"
        exit 2
    fi
    # Overwrite CUDA_VISIBLE_DEVICES so torchrun / vLLM only see working cards.
    export CUDA_VISIBLE_DEVICES=$(IFS=,; echo "${_USABLE_GPUS[*]}")
    _NPROC_AUTO=${#_USABLE_GPUS[@]}
    echo "[preflight] usable = ${CUDA_VISIBLE_DEVICES}  (count=${_NPROC_AUTO})"
else
    if [[ "${RESPECT_CUDA_VISIBLE_DEVICES:-0}" == "1" && -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        :
    else
        export CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7"
    fi
    IFS=',' read -ra _tmp <<< "${CUDA_VISIBLE_DEVICES}"
    _NPROC_AUTO=${#_tmp[@]}
    echo "[preflight] skipped, assuming ${_NPROC_AUTO} usable GPUs"
fi

# ------------ Paths (override via env if needed) ------------
# NOTOOL toggle (default 0 = original Thyme-with-tool behaviour).
# When NOTOOL=1:
#   * MODEL defaults to the raw Qwen2.5-VL-7B-Instruct weights (no SFT),
#     because the SFT-stage1 checkpoint was distilled on tool-using
#     trajectories and would be biased for the no-tool baselines.
#   * DATASET defaults to the no-tool version produced by
#     ``build_rl_dataset.py --no_tool`` (user text has no sandbox mention).
#   * RESUME_FROM is forced empty (starting from the raw model).
#   * stop_words drop <code>/</code> so the model never learns to emit them.
# Baseline-A (fmt/vqa/cst only) and Baseline-B (+FaithScore) both share
# this switch.
NOTOOL=${NOTOOL:-0}
if [[ "${NOTOOL}" == "1" ]]; then
    MODEL=${MODEL:-./models/Qwen2.5-VL-7B-Instruct}
    DATASET=${DATASET:-./output/thyme_safety_rl_train_notool.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-./output/rl_grpo_qwen25vl7b_safety_notool}
    # No SFT resume in no-tool mode; raw model is the starting point.
    RESUME_FROM=""
    _STOP_WORDS_ARGS=('<|im_end|>' '</answer>')
    echo "[launcher] NOTOOL=1  -> baseline mode, raw model, no <code> tokens"
else
    # TCG-RL / GRPO policy initialisation. Point MODEL at the backbone
    # (Qwen2.5-VL-7B-Instruct) or, preferably, at the cold-start SFT
    # checkpoint produced by ./sft_stage1.sh.
    MODEL=${MODEL:-./models/Qwen2.5-VL-7B-Instruct}
    DATASET=${DATASET:-./output/thyme_safety_rl_train.jsonl}
    OUTPUT_DIR=${OUTPUT_DIR:-./output/rl_grpo_qwen25vl7b_safety}
    _STOP_WORDS_ARGS=('<|im_end|>' '</code>' '</answer>' '<code>')
fi
# Fail fast on a bad local path, but still allow Hub ids such as
# "Qwen/Qwen2.5-VL-7B-Instruct".
if [[ "${MODEL}" == /* || "${MODEL}" == ./* || "${MODEL}" == ../* || "${MODEL}" == "~"* ]] && [[ ! -e "${MODEL}" ]]; then
    echo "[launcher] ERROR: MODEL=${MODEL} does not exist."
    echo "[launcher]        Set MODEL=/path/to/Qwen2.5-VL-7B-Instruct (or your SFT checkpoint)."
    exit 2
fi
PLUGIN=${PLUGIN:-./safety_rm.py}

# Optional short tag appended to both OUTPUT_DIR and LOG file name, e.g.
#     RUN_NAME=faith bash rl.sh
# ->  ./output/rl_grpo_qwen25vl7b_safety_faith/...
# ->  ./logs/rl_20260729_161637_faith.log
# Default empty = original naming (no suffix).
RUN_NAME=${RUN_NAME:-}
if [[ -n "${RUN_NAME}" ]]; then
    OUTPUT_DIR="${OUTPUT_DIR}_${RUN_NAME}"
    _LOG_SUFFIX="_${RUN_NAME}"
else
    _LOG_SUFFIX=""
fi

mkdir -p ./logs "${OUTPUT_DIR}"
LOG=./logs/rl_$(date +%Y%m%d_%H%M%S)${_LOG_SUFFIX}.log
echo "[launcher] run_name    : ${RUN_NAME:-<empty>}"
echo "[launcher] output_dir  : ${OUTPUT_DIR}"
echo "[launcher] logging to  : ${LOG}"

# ------------ Reward weights (mirror Thyme defaults) ------------
export FMT_WEIGHT=${FMT_WEIGHT:-0.5}
export VQA_WEIGHT=${VQA_WEIGHT:-1.0}
export CST_WEIGHT=${CST_WEIGHT:-0.5}
# _NORM=0 keeps rewards in [0, weight] (matches INVALID_REWARD_VALUE=0 semantics).
export FMT_NORM=${FMT_NORM:-0}
export VQA_NORM=${VQA_NORM:-0}
export CST_NORM=${CST_NORM:-0}

# ------------ CF-FaithGRPO: SafeFaithScore reward (optional) ------------
# When FS_TEXT_URL and FS_VEM_URL are set, ``safety_faith_orm`` (defined in
# safety_rm.py -> FaithORM) will score every rollout with:
#     F = FAITH_ALPHA * FaithScore + (1 - FAITH_ALPHA) * CAF
#     reward = FAITH_WEIGHT * F
# Both metrics are computed by two remote vLLM judge servers:
#     FS_TEXT_URL  -> Qwen3-32B     (stages 1/2/4)
#     FS_VEM_URL   -> Qwen3-VL-32B  (stage 3 visual entailment)
# See cf_faith/serve_judge_notes.md for how to start them.
#
# If EITHER url is unset (default), FaithORM prints a single [cf-faith][skip]
# warning at startup and returns F=0 for every sample -> equivalent to
# training without this reward. So this block is safe to leave in place.
export FAITH_WEIGHT=${FAITH_WEIGHT:-0.5}
export FAITH_ALPHA=${FAITH_ALPHA:-0.7}
export FS_TAXONOMY=${FS_TAXONOMY:-harmeme}
export FS_API_WORKERS=${FS_API_WORKERS:-16}
export FS_TIMEOUT=${FS_TIMEOUT:-120}
export FS_MAX_RETRY=${FS_MAX_RETRY:-3}
# FS_TEXT_URL / FS_TEXT_MODEL / FS_VEM_URL / FS_VEM_MODEL are intentionally
# NOT defaulted: leaving them unset is the explicit signal to disable F.
# To enable, set them here (or in the shell) before launching, e.g.:
#     export FS_TEXT_URL=http://<judge-host>:8001/v1
#     export FS_TEXT_MODEL=Qwen3-32B
#     export FS_VEM_URL=http://<judge-host>:8002/v1
#     export FS_VEM_MODEL=Qwen3-VL-32B
if [[ -n "${FS_TEXT_URL:-}" && -n "${FS_VEM_URL:-}" ]]; then
    echo "[launcher] SafeFaithScore ENABLED"
    echo "[launcher]   text: ${FS_TEXT_MODEL:-<unset!>} @ ${FS_TEXT_URL}"
    echo "[launcher]   vem : ${FS_VEM_MODEL:-<unset!>} @ ${FS_VEM_URL}"
    echo "[launcher]   F = ${FAITH_ALPHA} * FaithScore + $(python -c "print(1-${FAITH_ALPHA})") * CAF"
    echo "[launcher]   reward = ${FAITH_WEIGHT} * F   (taxonomy=${FS_TAXONOMY})"
    # Best-effort health probe so the user sees a clear error BEFORE
    # 8-way DDP starts (which would otherwise buries the error).
    if command -v curl >/dev/null 2>&1; then
        if ! curl -sf --max-time 5 "${FS_TEXT_URL}/models" >/dev/null 2>&1; then
            echo "[launcher] WARNING: text judge at ${FS_TEXT_URL}/models is not reachable."
            echo "[launcher]          Training will still start, but FaithORM will fall back to F=0."
        fi
        if ! curl -sf --max-time 5 "${FS_VEM_URL}/models" >/dev/null 2>&1; then
            echo "[launcher] WARNING: vem  judge at ${FS_VEM_URL}/models is not reachable."
            echo "[launcher]          Training will still start, but FaithORM will fall back to F=0."
        fi
    fi
else
    echo "[launcher] SafeFaithScore DISABLED (FS_TEXT_URL / FS_VEM_URL not both set)"
    echo "[launcher]   FaithORM will return 0 for every sample; equivalent to no faith reward."
fi

# ------------ CF-FaithGRPO: tool-level counterfactual reward (optional) ------------
# When CF_SIDECAR_URL and CF_SIDECAR_MODEL are set (AND the FaithScore
# judge is also configured above), ``safety_faith_tool_orm`` will run
# a counterfactual rollout per <code> tool call and score:
#     Δ_t     = F(τ) − F(τ_{-t})
#     bucket  = +1 if Δ_t > TOOL_DELTA_EPSILON,
#               −1 if Δ_t < −TOOL_DELTA_EPSILON, else 0
#     reward  = TOOL_WEIGHT * Σ_t bucket
# The side-car serves the INITIAL SFT checkpoint (fixed, no policy sync)
# on a separate machine; training only talks to it over HTTP.
#
# If EITHER CF_SIDECAR_URL / CF_SIDECAR_MODEL is unset (default), or the
# FaithScore judge is disabled, FaithToolORM returns 0 for every sample
# and prints one [cf-faith][skip] warning -- training is unaffected.
export TOOL_WEIGHT=${TOOL_WEIGHT:-0.3}
export TOOL_DELTA_EPSILON=${TOOL_DELTA_EPSILON:-0.05}
export CF_MAX_TOOLS_PER_SAMPLE=${CF_MAX_TOOLS_PER_SAMPLE:-3}
export CF_ROLLOUT_WORKERS=${CF_ROLLOUT_WORKERS:-8}
export CF_MAX_ROUNDS=${CF_MAX_ROUNDS:-6}
export CF_MAX_NEW_TOKENS=${CF_MAX_NEW_TOKENS:-1536}
export CF_SIDECAR_TIMEOUT=${CF_SIDECAR_TIMEOUT:-180}
export TOOL_MAX_SAMPLES_PER_BATCH=${TOOL_MAX_SAMPLES_PER_BATCH:--1}
# CF_SIDECAR_URL / CF_SIDECAR_MODEL are intentionally NOT defaulted:
# leaving them unset is the explicit signal to disable the tool-level
# faith reward. To enable, set them here (or in the shell), e.g.:
#     export CF_SIDECAR_URL=http://<sidecar-host>:8003/v1
#     export CF_SIDECAR_MODEL=cf_sidecar
if [[ -n "${CF_SIDECAR_URL:-}" && -n "${CF_SIDECAR_MODEL:-}" ]]; then
    echo "[launcher] Tool-level Faith (CF-FaithGRPO) ENABLED"
    echo "[launcher]   sidecar: ${CF_SIDECAR_MODEL} @ ${CF_SIDECAR_URL}"
    echo "[launcher]   reward = ${TOOL_WEIGHT} * Σ_t bucket(F(τ)-F(τ_{-t}))"
    echo "[launcher]   epsilon=${TOOL_DELTA_EPSILON} max_tools=${CF_MAX_TOOLS_PER_SAMPLE} workers=${CF_ROLLOUT_WORKERS}"
    if command -v curl >/dev/null 2>&1; then
        if ! curl -sf --max-time 5 "${CF_SIDECAR_URL}/models" >/dev/null 2>&1; then
            echo "[launcher] WARNING: side-car at ${CF_SIDECAR_URL}/models is not reachable."
            echo "[launcher]          Training will still start, but FaithToolORM will fall back to 0."
        fi
    fi
    # FaithToolORM needs the FaithScore judge too. If the judge is
    # disabled but the sidecar is enabled, F(τ_{-t}) cannot be scored
    # and the tool reward will fall back to 0; warn accordingly.
    if [[ -z "${FS_TEXT_URL:-}" || -z "${FS_VEM_URL:-}" ]]; then
        echo "[launcher] WARNING: CF_SIDECAR_URL is set but FS_TEXT_URL / FS_VEM_URL are not."
        echo "[launcher]          FaithToolORM needs the judge to score F(τ) and F(τ_{-t});"
        echo "[launcher]          without it, tool reward will be 0."
    fi
else
    echo "[launcher] Tool-level Faith DISABLED (CF_SIDECAR_URL / CF_SIDECAR_MODEL not both set)"
    echo "[launcher]   FaithToolORM will return 0 for every sample."
fi

# ------------ TFA: Teacher-Forced Visual Ablation tool reward (optional) ------------
# ALTERNATIVE to the side-car CF-rollout above. Instead of re-generating
# a counterfactual trajectory and scoring it with SafeFaithScore, TFA
# runs TWO teacher-forced forwards of the current policy per sample:
#   - orig : with the original sandbox crop
#   - cf   : with a same-size WHITE image at the t-th crop position
# and computes the per-token log-prob difference on the post-tool
# reasoning window (M_t):
#     D_t     = mean( log π(y | orig) - log π(y | white) )  in nats/token
#     bucket  = +1 if D_t >  TFA_EPSILON, else 0
# The bucketized reward is then quality-gated inside
# ``_score_completions`` using VQAORM (correctness) and FaithORM (F(τ)):
#     +1 -> keep  (+weight) if correct AND F(τ) >= TFA_F_HIGH
#     +1 -> 0                otherwise
# 20260804 change: TFA is now purely POSITIVE reinforcement -- no more
# negative bucket (D_t < -eps) and no more flip-to-negative in the gate.
# Worst case TFA does nothing; it never penalises tool use.
#
# COMPARED TO THE REFERENCE-BASED TOOL-CONTRASTIVE REWARD:
#   * NO side-car server needed (saves 1 GPU host)
#   * NO extra judge calls for τ_{-t} (only orig F is scored)
#   * BUT: measures "evidence dependency" not "faithfulness delta". The
#     semantics are therefore different from the tool-contrastive reward
#     described in the paper.
#
# When TFA_ENABLE!=1 (default), TFAToolORM is a silent no-op even if
# ``safety_tfa_tool_orm`` is in --reward_funcs.
export TFA_ENABLE=${TFA_ENABLE:-0}
export TFA_WEIGHT=${TFA_WEIGHT:-0.5}         # analogue of TOOL_WEIGHT
export TFA_EPSILON=${TFA_EPSILON:-0.01}      # D_t > eps -> +bucket; else 0 (no -1 bucket since 20260804)
export TFA_MAX_TOOLS_PER_SAMPLE=${TFA_MAX_TOOLS_PER_SAMPLE:-1}
export TFA_WINDOW_K=${TFA_WINDOW_K:-128}     # per-token window cap
export TFA_F_HIGH=${TFA_F_HIGH:-0.85}
export TFA_F_LOW=${TFA_F_LOW:-0.65}
# TFA_WHITE_ROOT / TFA_SFT_DIR default to values computed inside the
# trainer if unset; override only if you moved the checkout.

# ------------ Tool-contrastive grounding reward (TCG-RL, optional) ------------
# ALTERNATIVE to the side-car / teacher-forced variants above. Uses a
# PRE-COMPUTED reference FaithScore ``F_ref(qid)`` produced offline by
# running the raw Qwen2.5-VL-7B-Instruct model with the *direct* no-tool
# prompt (no <think>/<answer> tags, no tool calls). At training time:
#     Δ_i     = FaithScore_atomic(τ_i) - F_ref(qid_i)
#     bucket  = +1 if Δ_i >  OFFLINE_CF_EPS, -1 if Δ_i < -OFFLINE_CF_EPS, else 0
#     reward_i= OFFLINE_CF_WEIGHT * bucket
#
# Semantics: reward the policy only when its tool-augmented reasoning
# measurably improves atomic FaithScore over a no-tool reference. Unlike
# the side-car variant this needs no extra inference server, and unlike the
# teacher-forced variant no extra forward pass.
#
# F_REF_PATH points to the json produced by
#   cf_faith/build_f_ref_table.py --inputs <faithscore.direct.jsonl ...>
#
# When F_REF_PATH is unset (default), FaithOfflineCFORM prints one
# [cf-faith][skip] warning and returns reward=0 for every sample.
export OFFLINE_CF_WEIGHT=${OFFLINE_CF_WEIGHT:-0.5}
export OFFLINE_CF_EPS=${OFFLINE_CF_EPS:-0.05}
export OFFLINE_CF_LOG_MISS=${OFFLINE_CF_LOG_MISS:-100}
# F_REF_PATH is intentionally NOT defaulted: leaving it unset disables
# the reward.  To enable, e.g.:
#     export F_REF_PATH=./output/f_ref_qwen25vl7b_direct.json

# Tool-mode selector: which tool-level reward to plug into
# --reward_funcs. Allowed values:
#   cf         (default)   -- FaithToolORM (side-car counterfactual)
#   tfa                    -- TFAToolORM   (teacher-forced ablation)
#   offline_cf             -- FaithOfflineCFORM (reference lookup; TCG-RL)
#   none                   -- disable tool-level ORM entirely
#                             (safety-reward-only / +faithfulness runs)
TOOL_MODE=${TOOL_MODE:-cf}
if [[ "${TOOL_MODE}" == "tfa" ]]; then
    _TOOL_ORM_NAME=safety_tfa_tool_orm
    echo "[launcher] TOOL_MODE=tfa  ->  --reward_funcs uses TFAToolORM"
    echo "[launcher]   TFA_ENABLE=${TFA_ENABLE}  TFA_WEIGHT=${TFA_WEIGHT}"
    echo "[launcher]   TFA_EPSILON=${TFA_EPSILON}  TFA_MAX_TOOLS=${TFA_MAX_TOOLS_PER_SAMPLE}"
    echo "[launcher]   TFA_F_HIGH=${TFA_F_HIGH}  TFA_F_LOW=${TFA_F_LOW}"
elif [[ "${TOOL_MODE}" == "cf" ]]; then
    _TOOL_ORM_NAME=safety_faith_tool_orm
    echo "[launcher] TOOL_MODE=cf   ->  --reward_funcs uses FaithToolORM (side-car)"
elif [[ "${TOOL_MODE}" == "offline_cf" ]]; then
    _TOOL_ORM_NAME=safety_faith_offline_cf_orm
    echo "[launcher] TOOL_MODE=offline_cf -> --reward_funcs uses FaithOfflineCFORM (TCG-RL tool-contrastive reward)"
    echo "[launcher]   F_REF_PATH=${F_REF_PATH:-<unset!>}"
    echo "[launcher]   OFFLINE_CF_WEIGHT=${OFFLINE_CF_WEIGHT}  OFFLINE_CF_EPS=${OFFLINE_CF_EPS}"
    if [[ -z "${F_REF_PATH:-}" ]]; then
        echo "[launcher] WARNING: F_REF_PATH is not set; FaithOfflineCFORM will return 0 for every sample."
    elif [[ ! -f "${F_REF_PATH}" ]]; then
        echo "[launcher] WARNING: F_REF_PATH=${F_REF_PATH} is not a readable file; FaithOfflineCFORM will return 0."
    fi
elif [[ "${TOOL_MODE}" == "none" ]]; then
    _TOOL_ORM_NAME=""
    echo "[launcher] TOOL_MODE=none -> no tool-level ORM in --reward_funcs"
else
    echo "[launcher] ERROR: unknown TOOL_MODE='${TOOL_MODE}' (allowed: cf, tfa, offline_cf, none)"
    exit 2
fi

# ------------ PYTHONPATH -> in-tree ms-swift ------------
export PYTHONPATH=$(cd ../../ && pwd):${PYTHONPATH:-}

# ------------ Runtime knobs ------------
# Auto-adjust NPROC_PER_NODE to match usable GPUs unless the user forced one.
NPROC=${NPROC_PER_NODE:-${_NPROC_AUTO}}
export NPROC_PER_NODE=$NPROC
export OMP_NUM_THREADS=8
export MAX_PIXELS=${MAX_PIXELS:-1204224}
export FPS_MAX_FRAMES=${FPS_MAX_FRAMES:-10}
# Reduce memory fragmentation between vLLM (colocate) and training buffers.
# `expandable_segments:True` is silently ignored on some platforms
# ("expandable_segments not supported on this platform" at startup), so we
# use knobs that are honored on legacy CUDA allocators instead:
#   garbage_collection_threshold:0.8 -> allocator GCs freed blocks earlier
#                                       (default 1.0 = only on OOM), which
#                                       reduces cached-but-unreachable pool.
#   max_split_size_mb:512            -> cap huge cached blocks so a 5 GiB
#                                       backward tensor can be materialized
#                                       from smaller free fragments.
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:512}
# --- NCCL robustness knobs ---
# A single rank hitting a vLLM `limit_mm_per_prompt` violation (too many
# images in one rollout) can silently drop that rollout, mismatch the next
# collective across ranks, and stall everything until the NCCL watchdog
# fires. The knobs below make any such collective deadlock fail fast instead
# of burning GPU time; `limit_mm_per_prompt` itself is set to `image: 12` so
# multi-crop trajectories are not dropped in the first place.
#   TORCH_NCCL_ASYNC_ERROR_HANDLING=1  -> tear down PG on first NCCL error
#   TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC   -> watchdog stall detector, 300s
#   TORCH_NCCL_ENABLE_MONITORING=1     -> keep the heartbeat monitor on
# (dropped deprecated NCCL_ASYNC_ERROR_HANDLING alias, torch was warning
#  "Environment variable NCCL_ASYNC_ERROR_HANDLING is deprecated" for it)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-300}
export TORCH_NCCL_ENABLE_MONITORING=${TORCH_NCCL_ENABLE_MONITORING:-1}
# GRPOTrainer uses this to derive a per-run log tag; keep it stable per launch.
export TIME_STAMP=$(date +%m%d_%H:%M:%S)
echo "[launcher] NPROC_PER_NODE=${NPROC}  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

BSZ=${BSZ:-1}
GA=${GA:-5}
NUM_GEN=${NUM_GEN:-4}
EPOCH=${EPOCH:-2}
LR=${LR:-5e-7}
BETA=${BETA:-0.01}
# Fraction of each GPU permanently held by the colocated vLLM engine.
# The remainder is what the training-side forward/backward can use, so raise
# it if rollouts are slow and lower it if the backward pass OOMs.
VLLM_MEM=${VLLM_MEM:-0.20}
VLLM_TP=${VLLM_TP:-1}
MAX_LEN=${MAX_LEN:-8192}
# Maximum number of tokens generated per rollout. Observed completions peak
# around ~900 tokens, so 1536 leaves headroom while capping the [B,S,H]
# activation buffer materialized during backward. overlong_filter still
# masks any completion that hits the limit so gradients stay clean.
MAX_COMPLETION=${MAX_COMPLETION:-1536}
SAVE_STEPS=${SAVE_STEPS:-100}

# Optional: resume from a previous RL checkpoint (full state, incl. DeepSpeed
# optimizer shards, LR scheduler, RNG, global_step, trainer_state). Leave
# empty (the default) to start fresh from --model. Example:
#   RESUME_FROM=./output/rl_grpo_qwen25vl7b_safety/v8-20260714-144658/checkpoint-700
# NOTE: when NOTOOL=1 (Baseline-A/B) the earlier block has already forced
# RESUME_FROM="" so the following default is inert in that case.
RESUME_FROM=${RESUME_FROM:-}

# ------------ Pre-flight: kill stale RL processes + pick a free port ------------
# A previous launch that died mid-way (OOM / Ctrl-C) can leave a torchrun /
# swift rlhf process holding the master_port, so the next launch fails with
#   RuntimeError: ... port: 29500 ... EADDRINUSE
# Kill any leftovers first, then choose a random free port to avoid clashes.
if [[ "${NO_KILL_STALE:-0}" != "1" ]]; then
    _stale=$(pgrep -f "swift.cli.main rlhf" 2>/dev/null || true)
    if [[ -n "${_stale}" ]]; then
        echo "[launcher] killing stale rlhf processes: ${_stale}"
        # shellcheck disable=SC2086
        kill ${_stale} 2>/dev/null || true
        sleep 5
        # force-kill anything still alive
        _stale2=$(pgrep -f "swift.cli.main rlhf" 2>/dev/null || true)
        if [[ -n "${_stale2}" ]]; then
            # shellcheck disable=SC2086
            kill -9 ${_stale2} 2>/dev/null || true
            sleep 2
        fi
    fi
fi

# Choose a master_port: default 29500 + random offset in [0, 99] so repeated
# launches don't collide on the same fixed port. Override with MASTER_PORT=xxx.
if [[ -z "${MASTER_PORT:-}" ]]; then
    MASTER_PORT=$(( 29500 + (RANDOM % 100) ))
    # ensure it is actually free
    for _try in $(seq 1 20); do
        if python -c "import socket,sys; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
try:
    s.bind(('0.0.0.0',${MASTER_PORT})); s.close(); sys.exit(0)
except OSError:
    sys.exit(1)" 2>/dev/null; then
            break
        fi
        MASTER_PORT=$(( 29500 + (RANDOM % 100) ))
    done
fi
export MASTER_PORT
echo "[launcher] master_port = ${MASTER_PORT}"

# ------------ Launch ------------
# NOTE: do NOT pass --master_port to swift; swift's `rlhf` CLI does not
# recognize that argument and raises `ValueError: remaining_argv`. swift
# reads the MASTER_PORT env var internally (swift/utils/torch_utils.py:
# os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29500')) and
# forwards it to torchrun. We already `export MASTER_PORT` above.
python -m swift.cli.main rlhf \
    --rlhf_type                    grpo \
    --model                        "${MODEL}" \
    --external_plugins             "${PLUGIN}" \
    --reward_funcs                 safety_fmt_orm safety_vqa_orm safety_cst_orm safety_faith_orm ${_TOOL_ORM_NAME:+${_TOOL_ORM_NAME}} \
    --use_vllm                     true \
    --vllm_mode                    colocate \
    --vllm_max_model_len           ${MAX_LEN} \
    --vllm_tensor_parallel_size    ${VLLM_TP} \
    --vllm_limit_mm_per_prompt     '{"image": 12, "video": 0}' \
    --vllm_device                  auto \
    --vllm_gpu_memory_utilization  ${VLLM_MEM} \
    --vllm_enforce_eager           true \
    --train_type                   full \
    --torch_dtype                  bfloat16 \
    --dataset                      "${DATASET}" \
    --dataset_shuffle              true \
    --train_dataloader_shuffle     true \
    --max_pixels                   ${MAX_PIXELS} \
    --max_length                   ${MAX_LEN} \
    --max_completion_length        ${MAX_COMPLETION} \
    --freeze_aligner               false \
    --stop_words                   "${_STOP_WORDS_ARGS[@]}" \
    --num_train_epochs             ${EPOCH} \
    --per_device_train_batch_size  ${BSZ} \
    --per_device_eval_batch_size   1 \
    --padding_side                 left \
    --learning_rate                ${LR} \
    --lr_scheduler_type            cosine_with_min_lr \
    --lr_scheduler_kwargs          '{"min_lr_rate": 0.1, "num_cycles": 0.5}' \
    --gradient_accumulation_steps  ${GA} \
    --save_strategy                'steps' \
    --eval_strategy                'no' \
    --split_dataset_ratio          0 \
    --save_steps                   ${SAVE_STEPS} \
    --save_total_limit             100000 \
    --logging_steps                1 \
    --output_dir                   "${OUTPUT_DIR}" \
    --warmup_ratio                 0.03 \
    --dataloader_num_workers       8 \
    --num_generations              ${NUM_GEN} \
    --temperature                  1.0 \
    --beta                         ${BETA} \
    --top_p                        0.9 \
    --top_k                        50 \
    --repetition_penalty           1.05 \
    --deepspeed                    zero3 \
    --O3                           true \
    --log_completions              true \
    --report_to                    tensorboard \
    --async_generate               false \
    --num_iterations               1 \
    --overlong_filter              true \
    --offload_optimizer            true \
    --offload_model                true \
    --gc_collect_after_offload     false \
    --attn_impl                    flash_attn \
    ${RESUME_FROM:+--resume_from_checkpoint "${RESUME_FROM}"} \
    2>&1 | tee "${LOG}"

echo
echo "=== RL (GRPO) done ==="
echo "Checkpoints under: ${OUTPUT_DIR}"
