"""
Pilot diagnostics for the teacher distillation step.

Reads ``trajectories_raw.jsonl`` (output of ``run_teacher_distill.py``)
and prints a quality report so we can decide whether to scale up.

Pass criteria (suggested):
  * verdict_acc       >= 0.50    (else teacher is unreliable on memes)
  * sandbox_use_rate  >= 0.60    (else system prompt fails to push tool use)
  * format_ok_rate    >= 0.85    (else the schema is broken)

Usage:
    python pilot_check.py --in_path ./output/trajectories_raw.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict


CODE_RE = re.compile(
    r"<code>\s*(?:```\s*)?(?:python\s*)?[\s\S]*?\s*(?:```\s*)?</code>",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)


def _format_ok(text: str) -> bool:
    if len(THINK_RE.findall(text)) != 1:
        return False
    if len(ANSWER_RE.findall(text)) != 1:
        return False
    # answer must come AFTER think
    t_end = text.rfind("</think>")
    a_start = text.rfind("<answer>")
    return t_end != -1 and a_start != -1 and a_start > t_end


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_md", default=None,
                    help="optional: write summary to this markdown file")
    args = ap.parse_args()

    if not os.path.isfile(args.in_path):
        print(f"[ERR] not found: {args.in_path}", file=sys.stderr)
        sys.exit(2)

    n_total = 0
    n_correct = 0
    n_unparsed = 0
    n_format_ok = 0
    n_with_sandbox = 0
    n_error = 0
    iters_hist: Counter = Counter()
    code_blocks_hist: Counter = Counter()
    thinking_chars_buckets: Counter = Counter()
    per_subset_correct: dict = defaultdict(lambda: [0, 0])  # [correct, total]
    per_subset_sandbox: dict = defaultdict(lambda: [0, 0])
    per_label_correct: dict = defaultdict(lambda: [0, 0])
    confusion: Counter = Counter()  # (gt, pred) -> n
    qid_seen: set = set()

    with open(args.in_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            n_total += 1
            qid_seen.add(r["question_id"])

            text = r.get("raw_assistant_text", "") or ""
            pred = r.get("predicted", "<unparsed>")
            gt = r.get("gt_label")
            subset = r.get("subset", "unknown")

            if r.get("error"):
                n_error += 1
            if pred == "<unparsed>":
                n_unparsed += 1

            if pred == gt:
                n_correct += 1
                per_subset_correct[subset][0] += 1
                per_label_correct[gt][0] += 1
            per_subset_correct[subset][1] += 1
            per_label_correct[gt][1] += 1
            confusion[(gt, pred)] += 1

            if _format_ok(text):
                n_format_ok += 1
            if r.get("n_sandbox_images", 0) > 0:
                n_with_sandbox += 1
                per_subset_sandbox[subset][0] += 1
            per_subset_sandbox[subset][1] += 1

            iters_hist[r.get("n_iterations", 0)] += 1
            code_blocks_hist[r.get("n_code_blocks", 0)] += 1

            tc = r.get("thinking_chars", 0)
            if tc < 100:
                thinking_chars_buckets["<100"] += 1
            elif tc < 300:
                thinking_chars_buckets["100-300"] += 1
            elif tc < 800:
                thinking_chars_buckets["300-800"] += 1
            elif tc < 2000:
                thinking_chars_buckets["800-2000"] += 1
            else:
                thinking_chars_buckets[">=2000"] += 1

    if n_total == 0:
        print("no trajectories found")
        sys.exit(1)

    # ---- print report ----
    def line(s: str = ""):
        print(s)
        if md_lines is not None:
            md_lines.append(s)

    md_lines: list[str] | None = [] if args.out_md else None

    line("# Pilot quality report")
    line("")
    line(f"- file: `{args.in_path}`")
    line(f"- distinct questions: **{len(qid_seen)}**")
    line(f"- total trajectories: **{n_total}**")
    line(f"- crashed / errored:  {n_error} ({n_error/n_total:.1%})")
    line("")
    line("## Hard metrics (the ones that decide whether to scale up)")
    line("")
    line(f"- **verdict_acc**     = {n_correct/n_total:.3f}    "
         f"(correct {n_correct}/{n_total}; unparsed answers {n_unparsed})")
    line(f"- **format_ok_rate**  = {n_format_ok/n_total:.3f}  "
         "(exactly one <think>...</think> followed by one <answer>...</answer>)")
    line(f"- **sandbox_use_rate**= {n_with_sandbox/n_total:.3f}  "
         f"({n_with_sandbox}/{n_total} trajectories returned at least one sandbox image)")
    line("")
    line("## Per-subset verdict accuracy")
    line("")
    for subset, (c, t) in sorted(per_subset_correct.items()):
        sb_c, sb_t = per_subset_sandbox.get(subset, (0, 1))
        line(f"- `{subset:7s}`  acc={c/max(t,1):.3f}  ({c}/{t})   "
             f"sandbox_use={sb_c/max(sb_t,1):.3f}")
    line("")
    line("## Per-label verdict accuracy (recall)")
    line("")
    for lbl, (c, t) in sorted(per_label_correct.items()):
        line(f"- gt={lbl}: recall={c/max(t,1):.3f}  ({c}/{t})")
    line("")
    line("## Confusion matrix  (gt -> pred -> count)")
    line("")
    for (gt, pred), n in sorted(confusion.items(),
                                key=lambda x: (-x[1])):
        line(f"  - {gt:11s} -> {pred:11s} : {n}")
    line("")
    line("## Iteration count histogram")
    line("")
    for k, v in sorted(iters_hist.items()):
        line(f"  - n_iterations={k}: {v}")
    line("")
    line("## Code-block count histogram")
    line("")
    for k, v in sorted(code_blocks_hist.items()):
        line(f"  - n_code_blocks={k}: {v}")
    line("")
    line("## Thinking length distribution (chars)")
    line("")
    for k in ("<100", "100-300", "300-800", "800-2000", ">=2000"):
        line(f"  - {k:8s}: {thinking_chars_buckets.get(k, 0)}")
    line("")
    line("## Suggested pass thresholds")
    line("")
    line("  - verdict_acc        >= 0.50")
    line("  - format_ok_rate     >= 0.85")
    line("  - sandbox_use_rate   >= 0.60")
    line("")

    pass_v = (n_correct / n_total) >= 0.50
    pass_f = (n_format_ok / n_total) >= 0.85
    pass_s = (n_with_sandbox / n_total) >= 0.60
    overall = pass_v and pass_f and pass_s
    line(f"PASS={overall}  (verdict={pass_v}, format={pass_f}, sandbox={pass_s})")

    if md_lines is not None and args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)) or ".", exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[md] wrote {args.out_md}")


if __name__ == "__main__":
    main()
