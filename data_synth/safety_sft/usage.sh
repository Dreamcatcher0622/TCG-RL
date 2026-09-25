# =========================================================================
# TCG-RL (Tool-Contrastive Grounding RL) 启动命令合集
# =========================================================================
# 对应论文里的完整配方（奖励逐层叠加）：
#   规则奖励       只用 fmt / vqa / cst 三项规则奖励
#   + 忠实度奖励   SafeFaithScore 整轨迹忠实度 F(tau)
#   + 工具对比奖励 离线反事实奖励 FaithOfflineCFORM（论文的 TCG-RL 核心）
#
# 另外保留两条无工具基线（NOTOOL=1，原生 7B 直接 GRPO），
# 用来隔离「是否使用工具」这一变量的影响。
#
# 本文件是命令速查表，不是可直接执行的脚本：请先把下面的
# 占位路径/主机名替换成你自己的，或在 shell 里 export 好再逐段执行。
# =========================================================================
#
# 需要预先设置的变量（示例）：
#   REPO=/path/to/Thyme_code
#   VISCOT_ROOT=/path/to/viscot-harmeme        # 提供 faithscore 包（原子事实忠实度）
#   MODELS=/path/to/models                     # Qwen2.5-VL-7B/32B/72B、Qwen3-32B 等
#   CONDA_SH=/path/to/miniconda3/etc/profile.d/conda.sh
#   JUDGE_HOST=<judge-host>                    # 运行 FaithScore judge 的机器
#
# 奖励阶梯与论文的对应关系：
#   规则奖励             R = λ_fmt·r_fmt + λ_vqa·r_vqa + λ_cst·r_cst
#   + 忠实度奖励         R += λ_faith·F(τ)
#   + 工具对比奖励(TCG)  R += λ_cf·bucket(F_atomic(τ) - F_ref(q))


# -------------------------------------------------------------------------
# [公共] 在另一台机器启动两个 FaithScore judge（忠实度奖励/工具对比奖励都需要）
# -------------------------------------------------------------------------
# text judge: 服务 stage1/2/4（拆句、抽原子事实、归因）
CUDA_VISIBLE_DEVICES=0,1 vllm serve "${MODELS}/Qwen3-32B" \
    --host 0.0.0.0 --port 8001 --served-model-name Qwen3-32B \
    --tensor-parallel-size 2 --gpu-memory-utilization 0.90 \
    --max-model-len 16384 --trust-remote-code --disable-log-requests

# vem judge: 服务 stage3（视觉蕴含）
CUDA_VISIBLE_DEVICES=2,3 vllm serve "${MODELS}/Qwen3-VL-32B-Instruct" \
    --host 0.0.0.0 --port 8002 --served-model-name Qwen3-VL-32B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels": 1003520}' \
    --trust-remote-code --disable-log-requests


# -------------------------------------------------------------------------
# [可选] 工具级反事实奖励（FaithToolORM）需要的 side-car
# -------------------------------------------------------------------------
# 用 --served-model-name 时需要跟训练侧的 CF_SIDECAR_MODEL 一致
CUDA_VISIBLE_DEVICES=4,5 vllm serve "${REPO}/output/sft_stage1_qwen25vl7b_safety/checkpoint-XXXX" \
    --host 0.0.0.0 --port 8003 --served-model-name cf_sidecar \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 --max-model-len 16384 \
    --mm-processor-kwargs '{"max_pixels": 1204224}' \
    --trust-remote-code --disable-log-requests


# =========================================================================
# 训练侧
# =========================================================================
cd "${REPO}/data_synth/safety_sft"
source "${CONDA_SH}"
conda activate thyme

export MODEL=${REPO}/output/sft_stage1_qwen25vl7b_safety/checkpoint-XXXX
export RESUME_FROM=

# -------------------------------------------------------------------------
# 配置一：只用规则奖励（基线，fmt/vqa/cst）
# -------------------------------------------------------------------------
RUN_NAME=baseline bash rl.sh


# -------------------------------------------------------------------------
# 配置二：规则奖励 + 整轨迹忠实度奖励 F(tau)
# -------------------------------------------------------------------------
export FS_TEXT_URL=http://${JUDGE_HOST}:8001/v1
export FS_TEXT_MODEL=Qwen3-32B
export FS_VEM_URL=http://${JUDGE_HOST}:8002/v1
export FS_VEM_MODEL=Qwen3-VL-32B
export FAITH_WEIGHT=0.5    # F 的权重
export FAITH_ALPHA=0.7     # F = 0.7*FaithScore + 0.3*CAF
# (CF_SIDECAR_URL 不设置 -> FaithToolORM 自动返回 0)
RUN_NAME=faith_v2_alpha07 bash rl.sh


# -------------------------------------------------------------------------
# 配置三（论文主方法）：规则奖励 + 忠实度奖励 + 工具对比奖励
# =========================================================================
# [前置] 离线预算 F_ref：一次性完成，可以在别的机器/卡上跑，独立于训练，
#        产出 ./output/f_ref_qwen25vl7b_direct.json 供训练加载。
# =========================================================================
cd "${VISCOT_ROOT}"
source "${CONDA_SH}"
conda activate thyme
export CUDA_VISIBLE_DEVICES=4,5,6,7
MODEL_TAG="Qwen25VL7B_direct"
MODEL_PATH="${MODELS}/Qwen2.5-VL-7B-Instruct"
for SUBSET_FILE in harmc_train harmp_train mami_training pridemm_train; do
    RES_DIR="results/${SUBSET_FILE}/${MODEL_TAG}"
    mkdir -p "${RES_DIR}"
    python -m pipelines.run_stage2_answer \
        --question-file "data/questions/${SUBSET_FILE}.jsonl" \
        --answers-file  "${RES_DIR}/stage2_cot.jsonl" \
        --model-path    "${MODEL_PATH}" \
        --mode direct \
        --backend qwen-vl \
        --tensor-parallel-size 4 \
        --gpu-mem-util 0.85 \
        --max-new-tokens 512 \
        --temperature 0.0
done

# ---- (2) 用同一套 FaithScore judge (Qwen3-32B + Qwen3-VL-32B) 打分 ----
export FS_BACKEND=vllm
export FS_MODEL="${MODELS}/Qwen3-32B"
export FS_VEM_BACKEND=vllm
export FS_VEM_MODEL="${MODELS}/Qwen3-VL-32B-Instruct"
export GPU_MEM_UTIL=0.85
export TP_SIZE=4
for SUBSET_FILE in harmc_train harmp_train mami_training pridemm_train; do
    RES_DIR="results/${SUBSET_FILE}/${MODEL_TAG}"
    case "$SUBSET_FILE" in
        harmc_train)     TAX="harmeme" ;;
        harmp_train)     TAX="harmeme" ;;
        mami_training)   TAX="mami" ;;
        pridemm_train)   TAX="pridemm" ;;
    esac
    QUESTION_FILE="data/questions/${SUBSET_FILE}.jsonl" \
    TAXONOMY="${TAX}" \
    VEM_BACKEND="$FS_VEM_BACKEND" VEM_MODEL="$FS_VEM_MODEL" \
        bash scripts/run_faithscore.sh \
            "${RES_DIR}/stage2_cot.jsonl" \
            "$FS_BACKEND" "$FS_MODEL" cot
done

# ---- (3) 聚合成一个 {qid: F_ref} json 供训练侧加载 ----
# ⚠️ 这一步必须先于 RL 训练完成（否则训练侧 F_REF_PATH 文件不存在，
#    FaithOfflineCFORM 会一直返回 0）。如果训练已经先启动了也没关系：
#    FaithOfflineCFORM 在文件缺失时是「每 batch 重试」而非永久禁用，所以
#    这边生成好文件后，正在跑的训练下一个 batch 会自动加载并生效，无需重启。
cd "${REPO}/data_synth/safety_sft"
conda activate thyme
python cf_faith/build_f_ref_table.py \
    --inputs \
        "${VISCOT_ROOT}/results/harmc_train/${MODEL_TAG}/faithscore.cot.jsonl" \
        "${VISCOT_ROOT}/results/harmp_train/${MODEL_TAG}/faithscore.cot.jsonl" \
        "${VISCOT_ROOT}/results/mami_training/${MODEL_TAG}/faithscore.cot.jsonl" \
        "${VISCOT_ROOT}/results/pridemm_train/${MODEL_TAG}/faithscore.cot.jsonl" \
    --out ./output/f_ref_qwen25vl7b_direct.json


# =========================================================================
# [配置三 训练] 规则奖励 + 忠实度奖励 + 工具对比奖励（论文主方法）
# =========================================================================
cd "${REPO}/data_synth/safety_sft"
source "${CONDA_SH}"
conda activate thyme

unset NOTOOL
unset CF_SIDECAR_URL CF_SIDECAR_MODEL
unset TFA_ENABLE
export MODEL=${REPO}/output/sft_stage1_qwen25vl7b_safety/checkpoint-XXXX
export RESUME_FROM=

# FaithScore judge（同配置二）
export FS_TEXT_URL=http://${JUDGE_HOST}:8001/v1
export FS_TEXT_MODEL=Qwen3-32B
export FS_VEM_URL=http://${JUDGE_HOST}:8002/v1
export FS_VEM_MODEL=Qwen3-VL-32B
export FAITH_WEIGHT=0.5
export FAITH_ALPHA=0.7

# 挂工具对比奖励（离线反事实 ORM）
export TOOL_MODE=offline_cf
export F_REF_PATH=./output/f_ref_qwen25vl7b_direct.json
export OFFLINE_CF_WEIGHT=0.5   # λ
export OFFLINE_CF_EPS=0.05     # 分桶阈值
export OFFLINE_CF_LOG_MISS=100 # 最多打印 100 个不同的缺失 qid warning

RUN_NAME=offline_cf_v1_alpha07 bash rl.sh


# =========================================================================
# 可选：无工具基线（NOTOOL=1）
# =========================================================================
# 无工具 + 只跑规则奖励
unset FS_TEXT_URL FS_TEXT_MODEL FS_VEM_URL FS_VEM_MODEL
unset CF_SIDECAR_URL CF_SIDECAR_MODEL
unset TFA_ENABLE TOOL_MODE
export MODEL="${MODELS}/Qwen2.5-VL-7B-Instruct"
export RESUME_FROM=
export NOTOOL=1
RUN_NAME=baseline_notool bash rl.sh

# 无工具 + 忠实度奖励
export FS_TEXT_URL=http://${JUDGE_HOST}:8001/v1
export FS_TEXT_MODEL=Qwen3-32B
export FS_VEM_URL=http://${JUDGE_HOST}:8002/v1
export FS_VEM_MODEL=Qwen3-VL-32B
export FAITH_WEIGHT=0.5
export FAITH_ALPHA=0.7
RUN_NAME=baseline_notool_faith_alpha07 bash rl.sh


# =========================================================================
# 环境变量速查
# =========================================================================
# --- 忠实度奖励的 judge ---
# FS_TEXT_URL / FS_TEXT_MODEL : Qwen3-32B judge (stage1/2/4)
# FS_VEM_URL  / FS_VEM_MODEL  : Qwen3-VL-32B judge (stage3 VEM)
# VISCOT_ROOT (无默认)        : 提供 faithscore 包的仓库路径；未设置 = 忠实度奖励关闭
# FS_API_WORKERS  (16)        : judge HTTP 并发
# FS_TIMEOUT      (120)       : 单请求超时
# FS_MAX_RETRY    (3)         : 失败重试
# FS_TAXONOMY     (harmeme)   : stage4 归因 taxonomy
# FAITH_WEIGHT    (0.5)       : reward = FAITH_WEIGHT * F
# FAITH_ALPHA     (0.7)       : F = α*FaithScore + (1-α)*CAF
# FAITH_CACHE_MAX (4096)      : batch 内 F 缓存容量
#
# --- 工具对比奖励 / 离线反事实 (TOOL_MODE=offline_cf) ---
# F_REF_PATH                  : build_f_ref_table.py 产出的 json；未设置 = 该 ORM 静默返回 0
# OFFLINE_CF_WEIGHT (0.5)     : reward_i = OFFLINE_CF_WEIGHT * bucket(Δ_i, eps)
# OFFLINE_CF_EPS    (0.05)    : |Δ_i| <= eps -> 分桶 0
# OFFLINE_CF_LOG_MISS (100)   : 训练日志中最多打印多少个不同 qid miss 警告
#
# --- 工具级反事实奖励 side-car (TOOL_MODE=cf，可选) ---
# CF_SIDECAR_URL / CF_SIDECAR_MODEL : side-car endpoint
# TOOL_WEIGHT     (0.3)       : reward_i = TOOL_WEIGHT * Σ_t bucket(Δ_t)
# TOOL_DELTA_EPSILON (0.05)   : 分桶阈值
# CF_MAX_TOOLS_PER_SAMPLE (3) : 每条样本最多做 K 次反事实
# CF_ROLLOUT_WORKERS (8)      : 反事实并发
# CF_MAX_ROUNDS   (6)         : 反事实 agent 最多轮数
# CF_MAX_NEW_TOKENS (1536)    : 反事实单次生成最大 token
# CF_SIDECAR_TIMEOUT (180)    : 反事实单请求超时
#
# --- 教师强制视觉消融奖励 (TOOL_MODE=tfa，可选) ---
# TFA_ENABLE (0)              : 1=在 trainer 内做两次前向算 D_t
# TFA_WEIGHT (0.5)            : reward_i = TFA_WEIGHT * Σ_t bucket(D_t)
# TFA_EPSILON (0.01)          : D_t>eps -> +分桶；否则 0
# TFA_MAX_TOOLS_PER_SAMPLE (1): 每条样本最多考察前 K 次工具
# TFA_WINDOW_K (128)          : post-tool token 窗口上限
# TFA_F_HIGH (0.85)           : F>=high 且答对 -> +weight
# TFA_F_LOW  (0.65)           : F<=low 或答错 -> -weight (对 +bucket 生效)
#
# --- 无工具基线 (NOTOOL=1) ---
# NOTOOL   (0)                : 1 = 无工具基线；rl.sh 自动切换 MODEL /
#                               DATASET / stop_words / RESUME_FROM
# 默认 MODEL   : 原生 Qwen2.5-VL-7B-Instruct (无 SFT)
# 默认 DATASET : ./output/thyme_safety_rl_train_notool.jsonl
# 默认 OUTPUT  : ./output/rl_grpo_qwen25vl7b_safety_notool[_${RUN_NAME}]
# 需要 FaithScore judge? 规则奖励基线：否 / +忠实度奖励：是
