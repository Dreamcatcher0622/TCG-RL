"""Build the RL training dataset for Thyme-safety GRPO.

The GRPO trainer in this Thyme fork reads a Swift-style jsonl where each row
looks like:

    {
      "messages": [
        {"role": "system", "content": <RL system prompt>},
        {"role": "user",   "content": "<image>\\nIs this meme harmful..."}
      ],
      "images":   ["/abs/path/to/original.png"],
      "solution": "harmful"        // or "not_harmful"
      "question": "Is this meme harmful or not? ..."
    }

* ``messages`` + ``images`` -> given to the student for rollout.
* ``solution``               -> passed as a positional arg to every reward ORM.
* ``question``               -> passed via **kwargs to every reward ORM
                                (agent_rm.py uses this for the 72B judge; our
                                rule ORMs ignore it, but we keep it for
                                schema parity with the Thyme baseline).

System prompt is loaded verbatim from ``prompt_safety_rl.txt`` (which is
already the RL-facing prompt agreed with the SFT phase). The first user text
mirrors ``_build_initial_user_text`` in ``run_teacher_distill.py`` so that
SFT and RL see the exact same input distribution.

Usage
-----
    python build_rl_dataset.py \\
        --questions_dir /path/to/viscot-harmeme/data/questions \\
        --system_prompt ./prompt_safety_rl.txt \\
        --out_path      ./output/thyme_safety_rl_train.jsonl \\
        [--per_subset_cap 0]      # 0 = use everything from each subset
        [--exclude_sft ./output/thyme_safety_sft_train.jsonl]
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter
from typing import Iterable

from PIL import Image


SUBSET_TO_FILE = {
    "harmc":   "harmc_train.jsonl",
    "harmp":   "harmp_train.jsonl",
    "mami":    "mami_training.jsonl",
    "pridemm": "pridemm_train.jsonl",
}


# --------------------------------------------------------------------------- #
# Prompt construction (kept in sync with run_teacher_distill.py).
# --------------------------------------------------------------------------- #

def _load_system_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip() + "\n"


def _build_initial_user_text(image_path: str, no_tool: bool = False) -> str:
    """Mirrors ``run_teacher_distill._build_initial_user_text`` verbatim.

    When ``no_tool=True`` all mentions of the Python sandbox / ``<code>`` /
    ``<sandbox_output>`` are stripped so that the user turn is compatible
    with the no-tool baseline (Baseline-A / Baseline-B).
    """
    try:
        with Image.open(image_path) as img:
            sz = f"{img.width}x{img.height}"
    except Exception:
        sz = "Unable to determine"

    if no_tool:
        return (
            "<image>\n"
            "Is this meme harmful or not? Reason carefully using the image and "
            "any overlaid text, and finally output `<answer>harmful</answer>` or "
            "`<answer>not_harmful</answer>`.\n\n"
            f"### User Image Path: \"{image_path}\"\n"
            f"### User Image Size: \"{sz}\"\n\n"
            "### Output Format (strict adherence required):\n"
            "<think>Your detailed reasoning process about what is visible in the "
            "image and why it is or is not harmful, grounded in concrete visual "
            "and textual evidence.</think>\n"
            "<answer>harmful</answer>  OR  <answer>not_harmful</answer>\n"
        )

    return (
        "<image>\n"
        "Is this meme harmful or not? Reason carefully using the image and "
        "any overlaid text, optionally calling the sandbox to crop/zoom into "
        "fine details, and finally output `<answer>harmful</answer>` or "
        "`<answer>not_harmful</answer>`.\n\n"
        f"### User Image Path: \"{image_path}\"\n"
        f"### User Image Size: \"{sz}\"\n\n"
        "### Output Format (strict adherence required):\n"
        "<think>Your detailed reasoning process, including any <code> blocks "
        "and the <sandbox_output> blocks returned to you, goes here.</think>\n"
        "<answer>harmful</answer>  OR  <answer>not_harmful</answer>\n"
    )


# --------------------------------------------------------------------------- #
# I/O helpers.
# --------------------------------------------------------------------------- #

def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{ln_idx+1} bad json: {e}", file=sys.stderr)


def _normalize(rec: dict, subset: str) -> dict | None:
    image_path = rec.get("image_path")
    label = rec.get("label")
    qid = rec.get("question_id")
    if not image_path or not label or not qid:
        return None
    if label not in ("harmful", "not_harmful"):
        return None
    if not os.path.isfile(image_path):
        return None
    return {
        "question_id": qid,
        "subset": subset,
        "image_path": image_path,
        "label": label,
    }


def _load_sft_qids(path: str) -> set[str]:
    """Extract question_ids already used in SFT so RL can skip them."""
    if not path:
        return set()
    if not os.path.isfile(path):
        print(f"[WARN] --exclude_sft file not found: {path}", file=sys.stderr)
        return set()
    seen: set[str] = set()
    for rec in _iter_jsonl(path):
        meta = rec.get("meta") or {}
        qid = meta.get("question_id") or rec.get("question_id")
        if qid:
            seen.add(qid)
    print(f"[exclude_sft] loaded {len(seen)} question_ids to skip")
    return seen


# --------------------------------------------------------------------------- #
# Main.
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", required=True,
                    help="dir with harmc_train.jsonl / harmp_train.jsonl / "
                         "mami_training.jsonl / pridemm_train.jsonl")
    ap.add_argument("--system_prompt", default="./prompt_safety_rl.txt",
                    help="path to the RL system prompt file")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--per_subset_cap", type=int, default=0,
                    help="cap each subset to at most N (0 = use all)")
    ap.add_argument("--exclude_sft", default="",
                    help="jsonl of SFT training data; question_ids found "
                         "there will be excluded from the RL set")
    ap.add_argument("--no_tool", action="store_true",
                    help="strip all sandbox / <code> mentions from the user "
                         "turn (Baseline-A / Baseline-B compatible).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".",
                exist_ok=True)

    system_text = _load_system_prompt(args.system_prompt)
    print(f"[system] loaded {len(system_text)} chars from {args.system_prompt}")

    skip_qids = _load_sft_qids(args.exclude_sft)

    pooled: list[dict] = []
    per_subset_stats: dict[str, Counter] = {}

    for subset, fn in SUBSET_TO_FILE.items():
        src = os.path.join(args.questions_dir, fn)
        if not os.path.isfile(src):
            print(f"[ERR] missing {src}", file=sys.stderr)
            sys.exit(2)

        bucket: list[dict] = []
        skipped_by_sft = 0
        bad_imgpath = 0
        for rec in _iter_jsonl(src):
            n = _normalize(rec, subset)
            if n is None:
                bad_imgpath += 1
                continue
            if n["question_id"] in skip_qids:
                skipped_by_sft += 1
                continue
            bucket.append(n)

        rng.shuffle(bucket)
        if args.per_subset_cap > 0 and len(bucket) > args.per_subset_cap:
            bucket = bucket[: args.per_subset_cap]

        cnt = Counter(r["label"] for r in bucket)
        per_subset_stats[subset] = cnt
        print(f"[{subset:7s}] kept={len(bucket):6d}  "
              f"harmful={cnt['harmful']:5d}  "
              f"not_harmful={cnt['not_harmful']:5d}  "
              f"(skipped {skipped_by_sft} sft-overlap, "
              f"{bad_imgpath} bad-record)")

        pooled.extend(bucket)

    rng.shuffle(pooled)

    # Build Swift-format rows.
    n_written = 0
    with open(args.out_path, "w", encoding="utf-8") as fout:
        for r in pooled:
            user_text = _build_initial_user_text(r["image_path"], no_tool=args.no_tool)
            row = {
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user",   "content": user_text},
                ],
                "images":   [r["image_path"]],
                "solution": r["label"],
                # `question` is what agent_rm.py-style ORMs read via kwargs.
                "question": user_text,
                # Metadata kept for debugging / dataset ablation.
                "meta": {
                    "question_id": r["question_id"],
                    "subset": r["subset"],
                },
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1

    overall_label = Counter(r["label"] for r in pooled)
    overall_subset = Counter(r["subset"] for r in pooled)
    print("-" * 60)
    print(f"TOTAL RL examples: {n_written}  ->  {args.out_path}")
    print(f"  harmful     = {overall_label['harmful']}")
    print(f"  not_harmful = {overall_label['not_harmful']}")
    print(f"  per-subset  = {dict(overall_subset)}")

    # Also drop a small markdown-friendly stats file next to the jsonl.
    stats_path = re.sub(r"\.jsonl$", "_stats.md", args.out_path)
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("# Thyme-safety RL dataset stats\n\n")
        f.write(f"- source dir: `{args.questions_dir}`\n")
        f.write(f"- output    : `{args.out_path}`\n")
        f.write(f"- excluded  : {len(skip_qids)} SFT question_ids from "
                f"`{args.exclude_sft}`\n\n")
        f.write("## Per-subset\n\n")
        f.write("| subset | total | harmful | not_harmful |\n")
        f.write("| --- | ---: | ---: | ---: |\n")
        for subset in SUBSET_TO_FILE:
            cnt = per_subset_stats.get(subset, Counter())
            tot = cnt["harmful"] + cnt["not_harmful"]
            f.write(f"| {subset} | {tot} | {cnt['harmful']} | "
                    f"{cnt['not_harmful']} |\n")
        f.write(f"\n**Total: {n_written} "
                f"(harmful={overall_label['harmful']}, "
                f"not_harmful={overall_label['not_harmful']})**\n")
    print(f"[stats] wrote {stats_path}")


if __name__ == "__main__":
    main()
