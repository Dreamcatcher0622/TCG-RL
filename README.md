# TCG-RL: Tool-Contrastive Grounding Reinforcement Learning

Training code for **TCG-RL** (Tool-Contrastive Grounding Reinforcement Learning),
the method proposed in *"When Safety Rationales Hallucinate: Diagnosing and
Grounding Multimodal Safety Guards"* (paper under review).

The policy (Qwen2.5-VL-7B) analyses a meme, may write Python that is executed in a
sandbox (crop / zoom / rotate / contrast-enhance), and produces a `harmful` /
`not_harmful` verdict. GRPO optimises it with the reward

```
R(tau) = l_fmt*r_fmt + l_vqa*r_vqa + l_cst*r_cst + l_faith*F(tau) + l_cf*bucket(delta)
```

- `r_fmt`, `r_vqa`, `r_cst` — format validity, safety-label correctness, rationale-answer consistency
- `F(tau) = alpha*FaithScore_atomic(tau) + (1-alpha)*CAF(tau)` — trajectory-level faithfulness
- `delta = FaithScore_atomic(tau) - F_ref(q)`, `bucket(delta)` in `{+1, 0, -1}` — the **tool-contrastive** term: tool use is rewarded only when the rationale becomes more grounded than a fixed no-tool reference for the same question

Three reward configurations share one entry point
(`data_synth/safety_sft/rl.sh`); the last one is the paper's full method:

1. safety-prediction rewards — `r_fmt + r_vqa + r_cst`
2. + hallucination-aware faithfulness — `l_faith*F(tau)`
3. + tool-contrastive grounding — `l_cf*bucket(delta)`   **(full TCG-RL)**

The code builds on the Thyme Think-with-Image framework (an ms-swift fork); the
upstream README is kept as [README_Thyme.md](README_Thyme.md).

## Layout

```
swift/                     in-tree ms-swift fork: GRPO trainer, sandbox, plugin registry
data_synth/safety_sft/     TCG-RL pipeline
  rl.sh                    GRPO entry point (all three reward configurations)
  safety_rm.py             reward plugin (fmt / vqa / cst / faith / tool-contrastive ORMs)
  sft_stage1.sh            cold-start supervised fine-tuning
  build_*.py, run_teacher_distill.py, convert_to_swift_sft.py   data construction
  cf_faith/                faithfulness judge, counterfactual helpers, F_ref table builder
  eval/                    evaluation on the four safety benchmarks
  usage.sh                 full command cheat sheet (judge servers, data, all configurations)
```

## Setup

The paper's configuration uses 8 GPUs comparable to 8xH20 (96 GB).

```bash
conda create -n thyme python=3.10 -y && conda activate thyme
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/framework.txt "vllm>=0.5.1" deepspeed timm \
            qwen_vl_utils opencv-python timeout_decorator
pip install flash-attn --no-build-isolation
```

The grounding rewards need two OpenAI-compatible vLLM judge servers (text judge
`Qwen3-32B` and visual-entailment judge `Qwen3-VL-32B-Instruct`) plus an external
checkout of the atomic-faithfulness pipeline:

```bash
export VISCOT_ROOT=/path/to/viscot-harmeme          # provides `faithscore` + question files
export FS_TEXT_URL=http://<judge-host>:8001/v1  FS_TEXT_MODEL=Qwen3-32B
export FS_VEM_URL=http://<judge-host>:8002/v1   FS_VEM_MODEL=Qwen3-VL-32B
```

## Data

```bash
cd data_synth/safety_sft
python build_rl_dataset.py --questions_dir "${VISCOT_ROOT}/data/questions" \
    --out_path ./output/thyme_safety_rl_train.jsonl --seed 42
```

The tool-contrastive reward additionally needs the offline reference table: run the
raw Qwen2.5-VL-7B with the direct no-tool prompt on every RL question, score those
rationales with the same atomic-faithfulness pipeline, then aggregate them with
`cf_faith/build_f_ref_table.py` into `./output/f_ref_qwen25vl7b_direct.json`
(the exact commands are in `usage.sh`).

A cold-start SFT stage is recommended to teach the tool-use output format
(`run_full.sh` -> `convert_to_swift_sft.py` -> `sft_stage1.sh`); skip it if you
already have a suitable checkpoint.

## Training

```bash
cd data_synth/safety_sft
export MODEL=/path/to/cold-start-sft-checkpoint

# hyperparameters used in the paper (8xH20, 96 GB each); lower them if your
# GPUs have less memory, they are plain environment variables
export LR=1e-6 BSZ=8 GA=64 NUM_GEN=8 EPOCH=150 MAX_LEN=32768 \
       MAX_COMPLETION=512 BETA=0.01

# 1) safety-prediction rewards
TOOL_MODE=none RUN_NAME=baseline bash rl.sh

# 2) + hallucination-aware faithfulness reward
TOOL_MODE=none FAITH_WEIGHT=0.5 FAITH_ALPHA=0.7 RUN_NAME=faith bash rl.sh

# 3) full TCG-RL: + tool-contrastive grounding reward
TOOL_MODE=offline_cf F_REF_PATH=./output/f_ref_qwen25vl7b_direct.json \
    OFFLINE_CF_WEIGHT=0.5 OFFLINE_CF_EPS=0.05 RUN_NAME=tcg_rl bash rl.sh
```

Checkpoints are written under `./output/<OUTPUT_DIR>_<RUN_NAME>/` and launch logs
under `./logs/`. Any reward whose dependencies are unset returns 0 instead of
breaking the run, and `rl.sh` prints a pre-flight summary of the active rewards and
paths.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `[cf-faith][skip] ... F_REF_PATH not set` | the tool-contrastive term is 0 for every sample; export `F_REF_PATH` |
| `could not import ... faithscore` | `VISCOT_ROOT` is unset or does not point at the faithfulness pipeline |
| `[miss]` warnings on every batch | the dataset's `meta.question_id` values do not match the reference-table keys |
| grounding rewards always 0 | judge URLs unreachable (`curl $FS_TEXT_URL/models`) or `VISCOT_ROOT` unset |
| CUDA OOM during the backward pass | lower `VLLM_MEM`, `MAX_PIXELS`, `MAX_LEN` or `MAX_COMPLETION` |

## Citation

```bibtex
@misc{tcgrl2026,
  title = {When Safety Rationales Hallucinate: Diagnosing and Grounding Multimodal Safety Guards},
  note  = {TCG-RL: Tool-Contrastive Grounding Reinforcement Learning. Under review},
  year  = {2026}
}
```

Built on [Thyme](https://github.com/yfzhang114/Thyme),
[ms-swift](https://github.com/modelscope/ms-swift) and the FaithScore atomic-fact
faithfulness pipeline.
