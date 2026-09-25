"""
Step 1 of Thyme safety SFT data synthesis.

Merges the four train splits already prepared by viscot-harmeme:
    - harmc_train.jsonl     (HarMeme-Covid)
    - harmp_train.jsonl     (HarMeme-USPolitics)
    - mami_training.jsonl   (MAMI misogynous binary)
    - pridemm_train.jsonl   (PrideMM hate vs not-hate)

into a single shuffled "distillation input" jsonl whose schema is the
minimum needed by ``run_teacher_distill.py``:

    {
      "question_id": "harmc_train_000123",
      "subset":      "harmc" | "harmp" | "mami" | "pridemm",
      "image_path":  "/abs/path/to/img.png",
      "label":       "harmful" | "not_harmful",
      "ocr_text":    "...",                     # for logging only
    }

The 4 source files already use a unified ``label`` field
(``harmful`` / ``not_harmful``) and an absolute ``image_path``, so this
script is essentially a concatenate + shuffle + sanity-check + (optional)
per-subset cap.

Usage:
    python build_distill_input.py \\
        --questions_dir /path/to/viscot-harmeme/data/questions \\
        --out_path     ./output/merged_train.jsonl \\
        [--per_subset_cap 99999] \\
        [--limit 0] \\
        [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from typing import Iterable


SUBSET_TO_FILE = {
    "harmc":   "harmc_train.jsonl",
    "harmp":   "harmp_train.jsonl",
    "mami":    "mami_training.jsonl",
    "pridemm": "pridemm_train.jsonl",
}


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
    """Pick only the fields we need; drop records that are unusable."""
    image_path = rec.get("image_path")
    label = rec.get("label")
    qid = rec.get("question_id")
    if not image_path or not label or not qid:
        return None
    if label not in ("harmful", "not_harmful"):
        return None
    if not os.path.isfile(image_path):
        # Will be a hard failure in the sandbox; skip early.
        return None
    return {
        "question_id": qid,
        "subset": subset,
        "image_path": image_path,
        "label": label,
        "ocr_text": rec.get("ocr_text", ""),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions_dir", required=True,
                    help="dir containing harmc_train.jsonl etc.")
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--per_subset_cap", type=int, default=99999,
                    help="cap each subset to at most N examples (after shuffle)")
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, additionally cap the merged total to N (debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)) or ".", exist_ok=True)

    pooled: list[dict] = []
    per_subset_stats: dict[str, Counter] = {}

    for subset, fn in SUBSET_TO_FILE.items():
        src = os.path.join(args.questions_dir, fn)
        if not os.path.isfile(src):
            print(f"[ERR] missing {src}", file=sys.stderr)
            sys.exit(2)

        bucket: list[dict] = []
        bad_imgpath = bad_label = 0
        for rec in _iter_jsonl(src):
            n = _normalize(rec, subset)
            if n is None:
                if not rec.get("image_path") or not os.path.isfile(rec.get("image_path", "")):
                    bad_imgpath += 1
                else:
                    bad_label += 1
                continue
            bucket.append(n)

        rng.shuffle(bucket)
        if len(bucket) > args.per_subset_cap:
            bucket = bucket[: args.per_subset_cap]

        cnt = Counter(r["label"] for r in bucket)
        per_subset_stats[subset] = cnt
        print(f"[{subset:7s}] kept={len(bucket):6d}  "
              f"harmful={cnt['harmful']:5d}  "
              f"not_harmful={cnt['not_harmful']:5d}  "
              f"(skipped {bad_imgpath} missing-img, {bad_label} other)")

        pooled.extend(bucket)

    # Final shuffle of the pooled set so that during distillation we round-robin
    # across subsets — this also makes the throughput more uniform if some
    # subsets have systematically larger images.
    rng.shuffle(pooled)
    if args.limit > 0:
        pooled = pooled[: args.limit]

    with open(args.out_path, "w", encoding="utf-8") as f:
        for r in pooled:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    overall = Counter(r["label"] for r in pooled)
    overall_subset = Counter(r["subset"] for r in pooled)
    print("-" * 60)
    print(f"TOTAL written: {len(pooled)}  ->  {args.out_path}")
    print(f"  harmful     = {overall['harmful']}")
    print(f"  not_harmful = {overall['not_harmful']}")
    print(f"  per-subset  = {dict(overall_subset)}")


if __name__ == "__main__":
    main()
