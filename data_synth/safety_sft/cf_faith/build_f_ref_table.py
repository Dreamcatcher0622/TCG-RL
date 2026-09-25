"""Build the F_ref table for the TCG-RL tool-contrastive reward.

Reads one or more ``faithscore.*.jsonl`` files produced by
the ``faithscore`` pipeline (each line has a ``question_id`` and a
``faithscore_atomic``) and aggregates them into a single JSON:

    {
        "<question_id>": {
            "F_ref":  <float in [0,1]>,   # == faithscore_atomic
            "n_facts": <int>,
            "subset":  "<harmc|harmp|mami|pridemm>"
        },
        ...
    }

The saved dict is what ``FaithOfflineCFORM`` (safety_faith_offline_cf_orm)
loads at training start via ``F_REF_PATH``. Missing question_ids in the
table will produce reward = 0 during training (safe fallback).

Usage
-----
    python build_f_ref_table.py \\
        --inputs \\
            /path/to/results/harmc_train/direct/faithscore.direct.jsonl \\
            /path/to/results/harmp_train/direct/faithscore.direct.jsonl \\
            /path/to/results/mami_training/direct/faithscore.direct.jsonl \\
            /path/to/results/pridemm_train/direct/faithscore.direct.jsonl \\
        --out ./output/f_ref_qwen25vl7b_direct.json

The subset name for each row is inferred from the ``question_id`` prefix
(e.g. ``harmc_train_000123`` -> ``harmc``). Users can override that with
``--subset-from-path`` to derive the subset from the input path stem
instead.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Iterable


_QID_SUBSET_RE = re.compile(r"^(harmc|harmp|mami|pridemm)_")


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for ln_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{ln_idx + 1} bad json: {e}",
                      file=sys.stderr)


def _infer_subset(qid: str, path_stem_subset: str = "") -> str:
    m = _QID_SUBSET_RE.match(qid or "")
    if m:
        return m.group(1)
    return path_stem_subset or "unknown"


def _path_subset(path: str) -> str:
    stem = os.path.basename(os.path.dirname(os.path.abspath(path))) or ""
    for k in ("harmc", "harmp", "mami", "pridemm"):
        if k in stem.lower():
            return k
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="one or more faithscore.*.jsonl files "
                         "(direct-mode preferred; any faithscore.*.jsonl works)")
    ap.add_argument("--out", required=True,
                    help="output json path (dict {qid -> {F_ref, n_facts, subset}})")
    ap.add_argument("--subset-from-path", action="store_true",
                    help="derive subset from input path stem instead of qid prefix")
    ap.add_argument("--clip", action="store_true", default=True,
                    help="clip F_ref to [0,1] (default True)")
    ap.add_argument("--overwrite-on-collision", action="store_true",
                    help="if the same qid appears in multiple inputs, "
                         "the LATER one wins (default: WARN + keep first)")
    args = ap.parse_args()

    table: dict[str, dict] = {}
    per_subset: Counter = Counter()
    n_seen = 0
    n_skipped_no_qid = 0
    n_skipped_no_score = 0
    n_collisions = 0

    for path in args.inputs:
        if not os.path.isfile(path):
            print(f"[ERR] missing input: {path}", file=sys.stderr)
            sys.exit(2)
        path_stem_subset = _path_subset(path) if args.subset_from_path else ""
        print(f"[read] {path}  (subset hint={path_stem_subset or '<auto>'})")

        n_this = 0
        for rec in _iter_jsonl(path):
            n_seen += 1
            qid = rec.get("question_id") or ""
            if not qid:
                n_skipped_no_qid += 1
                continue
            fs = rec.get("faithscore_atomic", None)
            if fs is None:
                n_skipped_no_score += 1
                continue
            try:
                fs = float(fs)
            except (TypeError, ValueError):
                n_skipped_no_score += 1
                continue
            if args.clip:
                fs = max(0.0, min(1.0, fs))
            subset = _infer_subset(qid, path_stem_subset)
            n_facts = int(rec.get("n_facts", 0) or 0)

            if qid in table:
                n_collisions += 1
                if args.overwrite_on_collision:
                    table[qid] = {
                        "F_ref": fs,
                        "n_facts": n_facts,
                        "subset": subset,
                    }
                else:
                    # keep first-seen; warn once for every 100 collisions
                    if n_collisions <= 5 or n_collisions % 100 == 0:
                        print(f"[collision] qid={qid} appears again in {path}; "
                              f"keeping first-seen value "
                              f"(pass --overwrite-on-collision to flip).",
                              file=sys.stderr)
                continue

            table[qid] = {
                "F_ref": fs,
                "n_facts": n_facts,
                "subset": subset,
            }
            per_subset[subset] += 1
            n_this += 1

        print(f"[read] {path}  ->  {n_this} rows added")

    # ---- Persist ----------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    payload = {
        "_meta": {
            "n_rows": len(table),
            "per_subset": dict(per_subset),
            "inputs": [os.path.abspath(p) for p in args.inputs],
            "n_seen": n_seen,
            "n_skipped_no_qid": n_skipped_no_qid,
            "n_skipped_no_score": n_skipped_no_score,
            "n_collisions": n_collisions,
            "overwrite_on_collision": bool(args.overwrite_on_collision),
        },
        "table": table,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---- Report -----------------------------------------------------------
    print("-" * 60)
    print(f"F_ref table saved -> {args.out}")
    print(f"  rows            = {len(table)}")
    print(f"  seen input rows = {n_seen}")
    print(f"  skipped no qid  = {n_skipped_no_qid}")
    print(f"  skipped no atom = {n_skipped_no_score}")
    print(f"  collisions      = {n_collisions}")
    print(f"  per subset      = {dict(per_subset)}")
    if table:
        atoms = [v["F_ref"] for v in table.values()]
        print(f"  F_ref mean      = {sum(atoms) / len(atoms):.4f}")
        print(f"  F_ref min/max   = {min(atoms):.4f} / {max(atoms):.4f}")


if __name__ == "__main__":
    main()
