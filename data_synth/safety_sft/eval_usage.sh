#!/usr/bin/env bash
# Thyme 迁移服务器上的评估命令。
# 如果硬件配置不同，可以在 shell 中覆盖 CUDA_VISIBLE_DEVICES、TP_SIZE、CKPT、
# DATASETS 或运行脚本支持的其他变量。
set -euo pipefail

# Adjust these to your local checkout before running (or export them in the shell).
SFT_DIR="${SFT_DIR:-$(cd "$(dirname "$0")" && pwd)}"
VISCOT_ROOT="${VISCOT_ROOT:-$(cd "${SFT_DIR}/../../.." && pwd)/viscot-harmeme}"
DATA_ROOT="${DATA_ROOT:-${VISCOT_ROOT}/data}"
BASE_HF_MODEL="${BASE_HF_MODEL:-${MODEL:-./models/Qwen2.5-VL-7B-Instruct}}"
CONDA_SH="${CONDA_SH:-}"
CONDA_ENV="${CONDA_ENV:-thyme_eval}"

# 要评估的 checkpoint（HF safetensors 目录），例如
#   CKPT=./output/rl_grpo_qwen25vl7b_safety_rl_tcg/<run>/checkpoint-1600
CKPT="${CKPT:?please set CKPT=/path/to/checkpoint-dir}"
GPU_IDS="${CUDA_VISIBLE_DEVICES:-0}"
TP_SIZE="${TP_SIZE:-1}"

cd "${SFT_DIR}"

run_eval() {
    env DATA_ROOT="${DATA_ROOT}" VISCOT_ROOT="${VISCOT_ROOT}" \
    BASE_HF_MODEL="${BASE_HF_MODEL}" CONDA_SH="${CONDA_SH}" \
    CONDA_ENV="${CONDA_ENV}" "$@" \
        bash eval/run_safety_meme.sh "${CKPT}" "${TP_SIZE}"
}

# MODE=smoke（默认）：只评估一个数据集的分类结果，不依赖远程 Taiji 服务。
# MODE=full：评估四个测试子集并运行 FaithScore/CAF。FaithScore 文本阶段和 VEM
# 默认使用本地 vLLM 模型；不需要 FaithScore 时设置 RUN_FAITHSCORE=0。
case "${MODE:-smoke}" in
    smoke)
        run_eval DATASETS=harmc RUN_FAITHSCORE=0 NUM_WORKERS=1 \
            CUDA_VISIBLE_DEVICES="${GPU_IDS}"
        ;;
    full)
        run_eval CUDA_VISIBLE_DEVICES="${GPU_IDS}"
        ;;
    *)
        echo "用法：MODE=smoke|full CKPT=/path/to/checkpoint bash $0" >&2
        exit 2
        ;;
esac

# --------------------------- 常用变体 ---------------------------------------
# DATASETS="harmc harmp" RUN_FAITHSCORE=0 \
#     CUDA_VISIBLE_DEVICES=0 bash eval/run_safety_meme.sh "${CKPT}" 1
#
# 复用已经启动的 vLLM 服务：
# SKIP_SERVE=1 API_URL=http://127.0.0.1:18902/v1 RUN_FAITHSCORE=0 \
#     CUDA_VISIBLE_DEVICES=0 bash eval/run_safety_meme.sh "${CKPT}" 1
#
# 所有阶段使用本地 FaithScore（第二个模型需要足够显存；run_faithscore.sh
# 默认会拆分阶段运行）：
# FS_TEXT_BACKEND=vllm FS_TEXT_MODEL="${BASE_HF_MODEL}" \
# FS_VEM_BACKEND=vllm FS_VEM_MODEL="${BASE_HF_MODEL}" \
#     CUDA_VISIBLE_DEVICES=0 bash eval/run_safety_meme.sh "${CKPT}" 1

# 结果写入：
#   ${VISCOT_ROOT}/results/<subset>/<checkpoint-parent>_<checkpoint>/
