"""TFA offline probe — step 1b: batch D_t scan over an eval jsonl.

Goal
----
Extend the single-sample probe (``tfa_offline_probe.py``) to every rollout
row in a ``stage2_cot.jsonl`` with ``n_code_calls == 1``:

  * Load the policy checkpoint ONCE.
  * For each qid: build orig / cf batches, run two teacher-forced
    forwards, compute D_t across K ∈ {64, 128, 256, full}.
  * Dump per-sample results to a jsonl for downstream analysis (histogram,
    correlation with correctness, etc.).

Success criterion for step 1b
-----------------------------
  * ``D_t`` distribution is not all-zero and not all-blown-up (mean should
    land in the O(0.01–0.1) nats/token range, matching what 1a showed).
  * Failure rate < 5% (crashes / window-empty / etc.).
  * By-pred / by-correctness breakdown looks reasonable (correct
    predictions should not systematically have D_t ≈ 0).

Usage
-----
    python cf_faith/tfa_offline_scan.py \
        --model /path/to/Qwen2.5-VL-7B-Instruct \
        --rollout_jsonl /tmp/step1_harmc.jsonl \
        --sandbox_root ./_sandbox_eval/<run_tag>/harmc \
        --question_file "${VISCOT_ROOT}/data/questions/harmc_test.jsonl" \
        --out_jsonl ./_tfa/dt_scan_v21_harmc.jsonl \
        [--limit 500]
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import List, Optional

# Path bootstrap so we can import both ms-swift and our 1a probe helpers.
_THIS_DIR = Path(__file__).resolve().parent
_SFT_DIR = _THIS_DIR.parent
_THYME_ROOT = _SFT_DIR.parent.parent
if str(_THYME_ROOT) not in sys.path:
    sys.path.insert(0, str(_THYME_ROOT))
if str(_THIS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR.parent))

import torch

# Reuse the 1a helpers verbatim so 1b and 1a stay in lock-step.
from cf_faith.tfa_offline_probe import (
    parse_trajectory, build_messages, get_white_image_of_same_size,
    encode_via_swift_template, locate_post_tool_windows,
    per_token_logp_of_input_ids, compute_D_t_over_K,
    _load_question_index,
)


def iter_candidate_rows(rollout_jsonl: str, want_ncode: int = 1):
    """Yield only rows where ``n_code_calls == want_ncode``.

    Step 1b restricts to single-tool rollouts to remove ambiguity in
    which crop corresponds to which tool call (see 1a rationale).
    """
    with open(rollout_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if int(r.get("n_code_calls") or 0) != want_ncode:
                continue
            yield r


def make_result_row(
    qid: str,
    pred: str,
    label: str,
    D_by_K: dict,
    window: tuple,
    n_input_tokens: int,
    cot_len: int,
    fwd_orig_ms: float,
    fwd_cf_ms: float,
    status: str = "success",
    error: Optional[str] = None,
) -> dict:
    return {
        "question_id": qid,
        "pred": pred,
        "label": label,
        "correct": (pred == label),
        "D_t_K64":   D_by_K.get(64),
        "D_t_K128":  D_by_K.get(128),
        "D_t_K256":  D_by_K.get(256),
        "D_t_Kfull": D_by_K.get(-1),
        "window_start": window[0],
        "window_end":   window[1],
        "window_size":  window[1] - window[0],
        "n_input_tokens": n_input_tokens,
        "cot_len_chars": cot_len,
        "fwd_orig_ms": fwd_orig_ms,
        "fwd_cf_ms":   fwd_cf_ms,
        "status": status,
        "error": error,
    }


def summarize(rows: List[dict], keys=("D_t_K64", "D_t_K128", "D_t_K256", "D_t_Kfull")) -> str:
    lines = []
    ok = [r for r in rows if r["status"] == "success"]
    lines.append(f"N total    : {len(rows)}")
    lines.append(f"N success  : {len(ok)}")
    lines.append(f"N failure  : {len(rows) - len(ok)}")
    if not ok:
        return "\n".join(lines)
    for K in keys:
        vals = [r[K] for r in ok if r[K] is not None and r[K] == r[K]]  # drop NaN
        if not vals:
            lines.append(f"  {K}: n=0")
            continue
        vals_sorted = sorted(vals)
        n = len(vals_sorted)
        mean = sum(vals_sorted) / n
        median = vals_sorted[n // 2]
        p10 = vals_sorted[max(0, int(0.10 * n) - 1)]
        p90 = vals_sorted[min(n - 1, int(0.90 * n))]
        vmin, vmax = vals_sorted[0], vals_sorted[-1]
        pos = sum(1 for v in vals if v > 0.01)
        neg = sum(1 for v in vals if v < -0.01)
        near0 = n - pos - neg
        lines.append(
            f"  {K}: mean={mean:+.4f} med={median:+.4f} "
            f"p10={p10:+.4f} p90={p90:+.4f} min={vmin:+.4f} max={vmax:+.4f}  "
            f"|  >+0.01: {pos}  |Δ|<0.01: {near0}  <-0.01: {neg}"
        )
    # Correctness split on D_t_K128 (our primary K).
    lines.append("")
    lines.append("Correctness split (D_t_K128):")
    corr = [r["D_t_K128"] for r in ok if r["correct"]]
    wrng = [r["D_t_K128"] for r in ok if not r["correct"]]
    for tag, arr in (("correct", corr), ("wrong", wrng)):
        arr = [v for v in arr if v is not None and v == v]
        if arr:
            lines.append(f"  {tag:8s} n={len(arr):4d}  mean={sum(arr)/len(arr):+.4f}  "
                         f"median={sorted(arr)[len(arr)//2]:+.4f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.environ.get("MODEL", "./models/Qwen2.5-VL-7B-Instruct"),
    )
    ap.add_argument("--rollout_jsonl", required=True)
    ap.add_argument("--sandbox_root", required=True)
    ap.add_argument("--question_file", default="")
    ap.add_argument("--system_prompt",
                    default=str(_SFT_DIR / "prompt_safety_rl.txt"))
    ap.add_argument("--white_cache_dir",
                    default=str(_SFT_DIR / "_tfa_white_cache"))
    ap.add_argument("--out_jsonl", required=True,
                    help="Where to dump per-sample D_t records.")
    ap.add_argument("--summary_txt", default="",
                    help="Optional aggregate summary output (auto-derived "
                         "from out_jsonl if empty).")
    ap.add_argument("--limit", type=int, default=-1,
                    help="Cap on number of samples (-1 = all).")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K_list", default="64,128,256,-1")
    ap.add_argument("--flush_every", type=int, default=10,
                    help="Print running summary every N successful samples.")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    K_list = [int(x) for x in args.K_list.split(",")]

    # --- I/O setup --------------------------------------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.out_jsonl)), exist_ok=True)
    summary_txt = args.summary_txt or (
        os.path.splitext(args.out_jsonl)[0] + ".summary.txt"
    )
    os.makedirs(args.white_cache_dir, exist_ok=True)

    # --- Load question_id -> orig image path index (fallback) -------------
    q_index = _load_question_index(args.question_file)

    # --- Load system prompt ----------------------------------------------
    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read().rstrip() + "\n"
    print(f"[tfa-1b] system: {len(system_prompt)} chars from {args.system_prompt}",
          flush=True)

    # --- Load model once --------------------------------------------------
    print(f"[tfa-1b] loading model {args.model} in {dtype} ...", flush=True)
    t0 = time.time()
    from swift.llm import get_model_tokenizer, get_template
    model, tokenizer = get_model_tokenizer(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_impl="flash_attn",
    )
    model.eval()
    print(f"[tfa-1b] model loaded in {time.time() - t0:.1f}s", flush=True)

    template = get_template(
        model.model_meta.template,
        tokenizer,
        default_system=None,
        max_length=8192,
    )
    template.set_mode("train")     # ensure assistant content is encoded
    template.model = model         # required for _get_position_ids

    # --- Enumerate candidates ---------------------------------------------
    candidates = list(iter_candidate_rows(args.rollout_jsonl, want_ncode=1))
    if args.limit > 0:
        candidates = candidates[: args.limit]
    print(f"[tfa-1b] {len(candidates)} candidate rows "
          f"(n_code_calls==1) from {args.rollout_jsonl}", flush=True)

    # --- Iterate ----------------------------------------------------------
    results: List[dict] = []
    err_ct = Counter()
    t_start = time.time()
    with open(args.out_jsonl, "w", encoding="utf-8") as fout:
        for i, row in enumerate(candidates):
            qid = row.get("question_id", f"idx_{i}")
            try:
                traj = parse_trajectory(
                    row=row,
                    system_prompt=system_prompt,
                    sandbox_root=args.sandbox_root,
                    question_file_index=q_index,
                )
            except Exception as e:
                err_ct[type(e).__name__] += 1
                rec = make_result_row(
                    qid, row.get("pred", ""), row.get("label", ""),
                    {}, (0, 0), 0, len(row.get("cot") or ""),
                    0.0, 0.0, status="parse_error", error=str(e),
                )
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                results.append(rec)
                continue

            try:
                white_path = get_white_image_of_same_size(
                    traj.tool_turns[0].crop_image_path,
                    args.white_cache_dir,
                )
                msgs_o, imgs_o = build_messages(traj, crop_override_paths=None)
                msgs_c, imgs_c = build_messages(traj, crop_override_paths=[white_path])

                _enc_o, batch_o = encode_via_swift_template(msgs_o, imgs_o, template, tokenizer)
                _enc_c, batch_c = encode_via_swift_template(msgs_c, imgs_c, template, tokenizer)

                ids_o = batch_o["input_ids"]
                ids_c = batch_c["input_ids"]
                if ids_o.shape != ids_c.shape or not torch.equal(ids_o, ids_c):
                    raise RuntimeError(
                        f"input_ids mismatch: {ids_o.shape} vs {ids_c.shape}; "
                        f"crop-vs-white pixel dims must be equal"
                    )
                if "image_grid_thw" in batch_o and not torch.equal(
                    batch_o["image_grid_thw"], batch_c["image_grid_thw"]
                ):
                    raise RuntimeError("image_grid_thw differ")

                windows = locate_post_tool_windows(
                    input_ids=ids_o[0],
                    tokenizer=tokenizer,
                    n_tools=len(traj.tool_turns),
                )
                (ws, we) = windows[0]

                ta = time.time()
                logps_o = per_token_logp_of_input_ids(model, batch_o, device)
                fwd_o_ms = (time.time() - ta) * 1e3
                tb = time.time()
                logps_c = per_token_logp_of_input_ids(model, batch_c, device)
                fwd_c_ms = (time.time() - tb) * 1e3

                D_by_K = compute_D_t_over_K(logps_o, logps_c, (ws, we), K_list)

                rec = make_result_row(
                    qid=qid,
                    pred=row.get("pred", ""),
                    label=row.get("label", ""),
                    D_by_K=D_by_K,
                    window=(ws, we),
                    n_input_tokens=ids_o.shape[1],
                    cot_len=len(row.get("cot") or ""),
                    fwd_orig_ms=fwd_o_ms,
                    fwd_cf_ms=fwd_c_ms,
                    status="success",
                )
            except Exception as e:
                err_ct[type(e).__name__] += 1
                tb_str = traceback.format_exc()
                # Only keep first two lines of trace to keep jsonl small.
                short_tb = "; ".join(tb_str.strip().splitlines()[-2:])
                rec = make_result_row(
                    qid, row.get("pred", ""), row.get("label", ""),
                    {}, (0, 0), 0, len(row.get("cot") or ""),
                    0.0, 0.0, status="forward_error",
                    error=f"{type(e).__name__}: {str(e)[:200]}  ({short_tb[:200]})",
                )

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            results.append(rec)

            # Free the batch tensors before next iter.
            del batch_o, batch_c
            if 'logps_o' in locals():
                del logps_o, logps_c
            torch.cuda.empty_cache()

            if (i + 1) % args.flush_every == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(candidates) - i - 1) / max(rate, 1e-6)
                ok_ct = sum(1 for r in results if r["status"] == "success")
                d128 = [r["D_t_K128"] for r in results if r["status"] == "success"
                        and r["D_t_K128"] is not None]
                d128_mean = (sum(d128) / len(d128)) if d128 else float("nan")
                print(
                    f"[tfa-1b] {i+1}/{len(candidates)}  ok={ok_ct}  "
                    f"D128_mean={d128_mean:+.4f}  "
                    f"elapsed={elapsed:.1f}s  ETA={eta:.1f}s  "
                    f"errs={dict(err_ct)}",
                    flush=True,
                )

    total = time.time() - t_start
    ok_ct = sum(1 for r in results if r["status"] == "success")
    print(
        f"[tfa-1b] DONE: {ok_ct}/{len(candidates)} success in {total:.1f}s "
        f"({total / max(1, len(candidates)):.2f}s/sample); errors={dict(err_ct)}",
        flush=True,
    )
    print(f"[tfa-1b] wrote {args.out_jsonl}", flush=True)

    # --- Summary ---------------------------------------------------------
    summary = summarize(results)
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"# TFA step 1b summary\n")
        f.write(f"# model       : {args.model}\n")
        f.write(f"# rollout     : {args.rollout_jsonl}\n")
        f.write(f"# sandbox_root: {args.sandbox_root}\n")
        f.write(f"# out_jsonl   : {args.out_jsonl}\n")
        f.write(f"# n_candidates: {len(candidates)}  n_success: {ok_ct}\n")
        f.write(f"# errors      : {dict(err_ct)}\n")
        f.write(f"# total time  : {total:.1f}s\n\n")
        f.write(summary + "\n")
    print("\n===== SUMMARY =====")
    print(summary)
    print(f"\n[tfa-1b] summary -> {summary_txt}")


if __name__ == "__main__":
    main()
