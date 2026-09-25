# FaithScore Judge Servers — 部署说明

CF-FaithGRPO 训练时会通过 HTTP 调用**两个独立的 vLLM 服务**来打 F 分：

| Server | 用途 | 模型 | 训练时对应的 stages |
|---|---|---|---|
| **Text judge** | 文本 chat | Qwen3-32B | Stage 1（拆句） / Stage 2（原子事实） / Stage 4（CAF 归因） |
| **VEM judge**  | 视觉 chat | Qwen3-VL-32B | Stage 3（视觉蕴含） |

⚠️ **这两个服务必须部署在训练机之外**（例如另一台 8×H20），因为训练机上 8 张
卡已经跑满了 policy + colocate vLLM，没有余量再加载 32B。

---

## 一、Text judge (Qwen3-32B)

推荐启动命令（例如另一台机器的 GPU 0-1，TP=2）：

```bash
CUDA_VISIBLE_DEVICES=0,1 vllm serve /path/to/Qwen3-32B \
    --host 0.0.0.0 --port 8001 \
    --served-model-name Qwen3-32B \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 16384 \
    --trust-remote-code \
    --disable-log-requests
```

说明：
- Qwen3-32B fp16 ≈ 65 GB，H20 96GB 单卡也能塞下但 KV cache 紧张；建议 TP=2。
- Stage 1/2/4 的 prompt 通常 < 4k tokens，`max-model-len=16384` 已经很充足。
- `--served-model-name` 需要与 `rl.sh` 里 `FS_TEXT_MODEL` 保持完全一致。

启动后自检：
```bash
curl -sf http://<host>:8001/v1/models    # 应返回 JSON 含 Qwen3-32B
```

---

## 二、VEM judge (Qwen3-VL-32B)

```bash
CUDA_VISIBLE_DEVICES=2,3 vllm serve /path/to/Qwen3-VL-32B \
    --host 0.0.0.0 --port 8002 \
    --served-model-name Qwen3-VL-32B \
    --tensor-parallel-size 2 \
    --limit-mm-per-prompt image=2 \
    --gpu-memory-utilization 0.90 \
    --max-model-len 32768 \
    --mm-processor-kwargs '{"max_pixels": 1003520}' \
    --trust-remote-code \
    --disable-log-requests
```

说明：
- 视觉模型 KV cache 需要预留图像 token 空间，`max-model-len` 至少 32k。
- `max_pixels=1003520`（≈1280 tokens/图）与 meme 数据集图像大小匹配；更大会
  显存不足，更小会 caption OCR 精度下降。
- `--limit-mm-per-prompt image=2` 就够（stage3 一次只喂 1 张图）。

启动后自检：
```bash
curl -sf http://<host>:8002/v1/models
```

---

## 三、训练侧配置

在 `rl.sh` 顶部（或 shell 里）设置：

```bash
export FS_TEXT_URL=http://<host>:8001/v1
export FS_TEXT_MODEL=Qwen3-32B
export FS_TEXT_API_KEY=EMPTY

export FS_VEM_URL=http://<host>:8002/v1
export FS_VEM_MODEL=Qwen3-VL-32B
export FS_VEM_API_KEY=EMPTY

export FS_API_WORKERS=16       # 每个 backend 的并发 HTTP 连接数
export FS_TIMEOUT=120          # 单请求超时 (秒)
export FS_MAX_RETRY=3          # 失败重试次数
export FAITH_ALPHA=0.7         # F = alpha * FaithScore + (1-alpha) * CAF
export FAITH_WEIGHT=0.5        # reward 权重 (乘在 F 上)
export FS_TAXONOMY=harmeme     # stage4 归因 taxonomy: harmeme / mami / pridemm
```

如果**未设置** `FS_TEXT_URL` 或 `FS_VEM_URL`：
- `FaithORM` 会打印一次 `[cf-faith][skip]` 警告并对所有样本返回 F=0
- 训练照常进行，等同于关闭 FaithORM
- 只要不给 `safety_faith_orm` 一个非零权重，就完全没有副作用

---

## 四、验证 judge 服务可用

在训练机上（无需启动训练）快速跑：

```bash
# <your checkout>/data_synth/safety_sft
cd /path/to/Thyme_code/data_synth/safety_sft

python -c "
import os
os.environ['FS_TEXT_URL']='http://<host>:8001/v1'
os.environ['FS_TEXT_MODEL']='Qwen3-32B'
os.environ['FS_VEM_URL']='http://<host>:8002/v1'
os.environ['FS_VEM_MODEL']='Qwen3-VL-32B'
from cf_faith.faith_judge import SafeFaithScoreJudge
j = SafeFaithScoreJudge()
# 用一张真实的 meme 图跑单条打分
result = j.score_batch([{
    'image_path': '/absolute/path/to/some/meme.png',
    'cot': 'This meme shows a cat with a caption. It is not harmful.',
    'pred': 'not_harmful',
    'label': 'not_harmful',
    'question_id': 'sanity-1',
}])
print(result)
"
```

预期输出：
```
[cf-faith][init] health check OK (text: ...)
[{'F': 0.xx, 'faithscore_atomic': 0.xx, 'caf_composite': 0.xx, 'n_facts': N, 'status': 'ok'}]
```

如果 `status != 'ok'`，看前面的 `[cf-faith][error]` 提示定位问题。
