#!/usr/bin/env bash
# =============================================================================
# 使用 viscot-harmeme 流程，对 Thyme-safety RL/SFT checkpoint 进行端到端评估，
# 覆盖 HarMeme（harmc / harmp）、MAMI 和 PrideMM。
#
# 脚本执行内容：
#   1. 为指定的 Thyme checkpoint 启动 OpenAI 兼容的 vLLM 服务（Thyme 保存的是
#      原生 HF safetensors，无需合并 checkpoint）。
#   2. 对 ${DATASETS} 中的每个数据集：
#        - 运行 eval/eval_safety_meme.py（多轮 <code>-sandbox rollout，Thyme 的
#          system/user prompt 与 build_rl_dataset.py 保持一致），并写入
#          <VISCOT_ROOT>/results/<subset>/<TAG>/stage2_cot.jsonl。
#        - 使用 viscot-harmeme 的 eval_harmeme.py 计算 Acc / macro-F1。
#        - 可选运行 viscot-harmeme 的 scripts/run_faithscore.sh，计算 FaithScore
#          和 CAF（Taiji 文本后端 + 本地 vLLM VEM）。
#
# 用法：
#   bash eval/run_safety_meme.sh <CKPT_PATH> [<TP_SIZE>]
#
#   CKPT_PATH  Thyme HF 格式 checkpoint 目录
#              （例如 output/rl_grpo_qwen25vl7b_safety/v21-.../checkpoint-2800）
#   TP_SIZE    vLLM 的 --tensor-parallel-size（默认 1）
#
# 环境变量覆盖（全部可选）：
#   VISCOT_ROOT      viscot-harmeme 项目根目录
#   DATA_ROOT        原始数据根目录（默认 ${VISCOT_ROOT}/data）
#   REBUILD_QUESTIONS 设为 1 时用 DATA_ROOT 路径重建测试 JSONL（默认 1）
#   BASE_HF_MODEL    Qwen2.5-VL-7B-Instruct 路径（仅 FaithScore 的 VEM 后端使用；
#                    默认 $MODEL 或 ./models/Qwen2.5-VL-7B-Instruct）
#   TAG              输出子目录名称（默认：<checkpoint 名称>）
#   DATASETS         以空格分隔的数据集列表（默认 "harmc harmp mami pridemm"）
#   NUM_WORKERS      客户端 rollout 并发数（默认 4）
#   MAX_ROUNDS       每个样本的 Thyme agent 最大轮数（默认 6）
#   MAX_NEW_TOKENS   每轮生成的最大 token 数（默认 2048）
#   GPU_MEM_UTIL     vLLM 的 --gpu-memory-utilization（默认 0.80）
#   MAX_MODEL_LEN    vLLM 的 --max-model-len（默认 32768）
#   MAX_PIXELS       Qwen2.5-VL 处理器的 max_pixels（默认 1204224，即
#                    rl.sh 的训练配置；除非明确调整视觉 token 预算，否则保持一致）
#   VLLM_PORT        端口（默认 18902）
#   VLLM_HOST        主机（默认 127.0.0.1）
#   RUN_FAITHSCORE   设为 1 运行 FaithScore（默认）/ 设为 0 跳过
#   FS_TEXT_BACKEND  文本 FaithScore 后端（默认 "vllm"）
#   FS_TEXT_MODEL    文本 FaithScore 模型（默认 ./models/Qwen3-32B）
#   FS_VEM_BACKEND   VEM 后端（默认 "vllm"，本地视觉模型，避免自验证）
#   FS_VEM_MODEL     VEM 模型（默认 ./models/Qwen3-VL-32B-Instruct）
#   FS_API_WORKERS   FaithScore API 并发数（默认 8）
#   SKIP_SERVE       设为 1 复用已启动的 vLLM（地址为 $API_URL，默认 0）
#   CUDA_VISIBLE_DEVICES  使用的 GPU（默认继承 shell 设置，未设置时为 "0"）
# =============================================================================
set -euo pipefail

# ---- Conda 环境 ----
# 设置 CONDA_SH / CONDA_ENV 让脚本自动激活评估环境；否则使用当前 PATH 里的 python。
CONDA_SH=${CONDA_SH:-}
CONDA_ENV=${CONDA_ENV:-thyme_eval}
if [[ -z "${CONDA_SH}" ]]; then
    echo "[env] CONDA_SH not set; using current PATH's python: $(which python)"
elif [[ -f "${CONDA_SH}" ]]; then
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
    echo "[env] conda env : ${CONDA_ENV}"
    echo "[env] python    : $(which python)"
else
    echo "[env] ERROR: conda setup not found: ${CONDA_SH}" >&2
    echo "[env] Set CONDA_SH to an existing conda.sh, or leave it unset." >&2
    exit 1
fi

# ---- 命令行参数 ----
CKPT_PATH=${1:?"missing arg 1: Thyme checkpoint directory"}
TP_SIZE=${2:-1}

# ---- 路径 ----
SFT_DIR="$(cd "$(dirname "$0")/.." && pwd)"                     # data_synth/safety_sft
THYME_ROOT="$(cd "${SFT_DIR}/../.." && pwd)"                    # repo root
VISCOT_ROOT="${VISCOT_ROOT:-$(cd "${THYME_ROOT}/../.." && pwd)/viscot-harmeme}"
# Root of the raw benchmark data used to (re)build the question jsonl files.
DATA_ROOT="${DATA_ROOT:-${VISCOT_ROOT}/data}"
# Backbone / VEM weights; point these at your local copies.
BASE_HF_MODEL="${BASE_HF_MODEL:-${MODEL:-./models/Qwen2.5-VL-7B-Instruct}}"
REBUILD_QUESTIONS="${REBUILD_QUESTIONS:-1}"

# ---- 参数 ----
CKPT_TAG="$(basename "${CKPT_PATH}")"                            # e.g. checkpoint-2800
CKPT_PARENT="$(basename "$(dirname "${CKPT_PATH}")")"            # e.g. v21-20260722-193456
TAG="${TAG:-${CKPT_PARENT}_${CKPT_TAG}}"

DATASETS="${DATASETS:-harmc harmp mami pridemm}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_ROUNDS="${MAX_ROUNDS:-6}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.80}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
# 与 rl.sh 训练时的 MAX_PIXELS 保持一致（1204224 = 1.2M 像素，约每图 1.5k
# 个视觉 token），传给 Qwen2.5-VL 处理器，使评估图片分辨率与 RL rollout 一致。
MAX_PIXELS="${MAX_PIXELS:-1204224}"
RUN_FAITHSCORE="${RUN_FAITHSCORE:-1}"

# ---- vLLM 服务 ----
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-18902}"
API_URL="${API_URL:-http://${VLLM_HOST}:${VLLM_PORT}/v1}"
SERVED_NAME="thyme_safety_${CKPT_TAG}"

# ---- FaithScore 后端 ----
FS_TEXT_BACKEND="${FS_TEXT_BACKEND:-vllm}"
FS_TEXT_MODEL="${FS_TEXT_MODEL:-./models/Qwen3-32B}"
FS_VEM_BACKEND="${FS_VEM_BACKEND:-vllm}"
FS_VEM_MODEL="${FS_VEM_MODEL:-./models/Qwen3-VL-32B-Instruct}"
FS_API_WORKERS="${FS_API_WORKERS:-8}"

# ---- 开关 ----
SKIP_SERVE="${SKIP_SERVE:-0}"

echo "======================================================================"
echo "[thyme-eval] ckpt         : ${CKPT_PATH}"
echo "[thyme-eval] viscot_root  : ${VISCOT_ROOT}"
echo "[thyme-eval] data_root    : ${DATA_ROOT}"
echo "[thyme-eval] base_hf      : ${BASE_HF_MODEL}"
echo "[thyme-eval] tag          : ${TAG}"
echo "[thyme-eval] datasets     : ${DATASETS}"
echo "[thyme-eval] api_url      : ${API_URL}"
echo "[thyme-eval] served_name  : ${SERVED_NAME}"
echo "[thyme-eval] tp_size      : ${TP_SIZE}"
echo "[thyme-eval] gpu_mem_util : ${GPU_MEM_UTIL}"
echo "[thyme-eval] max_model_len: ${MAX_MODEL_LEN}"
echo "[thyme-eval] max_pixels   : ${MAX_PIXELS}  (== training MAX_PIXELS)"
echo "[thyme-eval] max_rounds   : ${MAX_ROUNDS}"
echo "[thyme-eval] run_faith    : ${RUN_FAITHSCORE}"
echo "[thyme-eval] cuda_devices : ${CUDA_VISIBLE_DEVICES:-<inherit>}"
echo "======================================================================"

for required_path in "${CKPT_PATH}" "${VISCOT_ROOT}" "${DATA_ROOT}"; do
    if [[ ! -e "${required_path}" ]]; then
        echo "[thyme-eval] ERROR: required path not found: ${required_path}" >&2
        exit 1
    fi
done
if [[ ! -d "${BASE_HF_MODEL}" ]]; then
    echo "[thyme-eval] ERROR: BASE_HF_MODEL not found: ${BASE_HF_MODEL}" >&2
    exit 1
fi

# 现有 question JSONL 中保存的是旧服务器的绝对图片路径。在当前服务器重建，
# 确保 rollout 和 FaithScore 都使用 DATA_ROOT。
if [[ "${REBUILD_QUESTIONS}" == "1" ]]; then
    echo "[thyme-eval] === rebuilding question files from ${DATA_ROOT} ==="
    (
        cd "${VISCOT_ROOT}"
        MM_SAFETY_DATA_ROOT="${DATA_ROOT}" python -m data.build_harmeme_questions \
            --subsets harmc harmp --splits test
        MM_SAFETY_DATA_ROOT="${DATA_ROOT}" python -m data.build_mami_questions \
            --splits test
        MM_SAFETY_DATA_ROOT="${DATA_ROOT}" python -m data.build_pridemm_questions \
            --splits test
    )
fi

# ---- 第 1 步：启动 vLLM（SKIP_SERVE=1 时跳过） ----
VLLM_PID=""
_cleanup() {
    if [[ -n "${VLLM_PID}" ]] && kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
        echo "[thyme-eval] stopping vLLM (pid=${VLLM_PID})"
        kill "${VLLM_PID}" || true
        wait "${VLLM_PID}" 2>/dev/null || true
    fi
}
trap _cleanup EXIT

if [[ "${SKIP_SERVE}" != "1" ]]; then
    echo "[thyme-eval] === starting vLLM ==="
    LOG_DIR="${SFT_DIR}/logs/eval"
    mkdir -p "${LOG_DIR}"
    VLLM_LOG="${LOG_DIR}/vllm_${TAG}_$(date +%Y%m%d_%H%M%S).log"

    # --limit-mm-per-prompt image=12 与 rl.sh 训练配置一致；每个样本可能累积
    # 1 张原图和每轮最多 3 张 sandbox 裁剪图，因此预留一定余量。
    # Qwen2.5-VL 需要开启 trust remote code。
    # --mm-processor-kwargs 固定训练时的 max_pixels，使视觉 token 数与训练一致。
    (
        vllm serve "${CKPT_PATH}" \
            --host "${VLLM_HOST}" \
            --port "${VLLM_PORT}" \
            --served-model-name "${SERVED_NAME}" \
            --tensor-parallel-size "${TP_SIZE}" \
            --limit-mm-per-prompt '{"image": 12}' \
            --gpu-memory-utilization "${GPU_MEM_UTIL}" \
            --max-model-len "${MAX_MODEL_LEN}" \
            --mm-processor-kwargs "{\"max_pixels\": ${MAX_PIXELS}}" \
            --trust-remote-code 
    ) >"${VLLM_LOG}" 2>&1 &
    VLLM_PID=$!
    echo "[thyme-eval] vLLM pid=${VLLM_PID} log=${VLLM_LOG}"

    # Poll /v1/models until ready (up to ~10 min).
    for i in $(seq 1 120); do
        if curl -sf "${API_URL}/models" >/dev/null 2>&1; then
            echo "[thyme-eval] vLLM ready after ${i} probes"
            break
        fi
        if ! kill -0 "${VLLM_PID}" >/dev/null 2>&1; then
            echo "[thyme-eval] ERROR: vLLM died early. Log tail:" >&2
            tail -n 80 "${VLLM_LOG}" >&2 || true
            exit 1
        fi
        sleep 5
    done

    if ! curl -sf "${API_URL}/models" >/dev/null 2>&1; then
        echo "[thyme-eval] ERROR: vLLM did not become ready in 10 min" >&2
        tail -n 80 "${VLLM_LOG}" >&2 || true
        exit 1
    fi
else
    echo "[thyme-eval] === reusing existing vLLM at ${API_URL} ==="
fi

# ---- 第 2 步：逐数据集运行 rollout、分类评估和（可选）FaithScore ----
for DS in ${DATASETS}; do
    QUESTION_FILE="${VISCOT_ROOT}/data/questions/${DS}_test.jsonl"
    SAVE_DIR="${VISCOT_ROOT}/results/${DS}/${TAG}"
    STAGE2_JSONL="${SAVE_DIR}/stage2_cot.jsonl"

    if [[ ! -f "${QUESTION_FILE}" ]]; then
        echo "[thyme-eval] skip ${DS}: not found ${QUESTION_FILE}"
        continue
    fi

    echo "======================================================================"
    echo "[thyme-eval] === ${DS} ==="
    echo "[thyme-eval] question_file : ${QUESTION_FILE}"
    echo "[thyme-eval] save_dir      : ${SAVE_DIR}"
    echo "======================================================================"

    mkdir -p "${SAVE_DIR}"

    # 2.1 Rollout inference (Thyme code-sandbox agent loop).
    python "${SFT_DIR}/eval/eval_safety_meme.py" \
        --api_url         "${API_URL}" \
        --eval_model_name "${SERVED_NAME}" \
        --question_file   "${QUESTION_FILE}" \
        --save_path       "${SAVE_DIR}" \
        --num_workers     "${NUM_WORKERS}" \
        --max_new_tokens  "${MAX_NEW_TOKENS}" \
        --max_rounds      "${MAX_ROUNDS}" \
        --sandbox_root    "${SFT_DIR}/_sandbox_eval/${TAG}/${DS}"

    # 2.2 通过 viscot-harmeme 计算分类指标。
    (
        cd "${VISCOT_ROOT}"
        python -m eval.eval_harmeme --answers-file "${STAGE2_JSONL}"
    )

    # 2.3 FaithScore + CAF（可选）。
    if [[ "${RUN_FAITHSCORE}" == "1" ]]; then
        case "${DS}" in
            harmc|harmp) TAX=harmeme ;;
            mami)        TAX=mami    ;;
            pridemm)     TAX=pridemm ;;
            *)           TAX="${DS}" ;;
        esac

        (
            cd "${VISCOT_ROOT}"
            TAXONOMY="${TAX}" \
            VEM_BACKEND="${FS_VEM_BACKEND}" \
            VEM_MODEL="${FS_VEM_MODEL}" \
            API_WORKERS="${FS_API_WORKERS}" \
                bash scripts/run_faithscore.sh \
                    "${STAGE2_JSONL}" \
                    "${FS_TEXT_BACKEND}" "${FS_TEXT_MODEL}" \
                    cot
        )
    fi
done

echo "======================================================================"
echo "[thyme-eval] all done"
echo "  outputs under: ${VISCOT_ROOT}/results/<subset>/${TAG}/"
echo "======================================================================"
