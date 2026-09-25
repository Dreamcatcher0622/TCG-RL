"""Step 3 of Thyme safety SFT data synthesis.

Converts ``full_trajectories.jsonl`` (the raw teacher rollouts written by
``run_teacher_distill.py``) into ms-swift's standard multi-modal SFT format::

    {
        "messages": [
            {"role": "system",    "content": "<safety system prompt>"},
            {"role": "user",      "content": "<image> Is this meme harmful..."},
            {"role": "assistant", "content": "<think>reasoning ... <code>```python...```</code><sandbox_output><image></sandbox_output> more reasoning ...</think><answer>harmful</answer>"}
        ],
        "images": ["/abs/orig.png", "/abs/_sandbox_tmp/full/crop_caption.jpg", ...]
    }

Design choices
--------------
1. **Filtering** — we keep only trajectories that meet ALL of:
     * ``error`` is null   (no crash)
     * ``predicted == gt_label``   (teacher got the right answer)
     * ``n_sandbox_images >= 1``   (the trajectory actually used the sandbox;
        this is the whole point of Thyme, and > 90% of our accepted samples
        naturally satisfy this)
     * format_ok               (exactly one <think>...</think> followed by
                                exactly one <answer>...</answer>, with
                                <answer> AFTER </think>)
     * every referenced image file actually exists on disk
   These 5 filters give us a clean SFT corpus.

2. **Message flattening** — the teacher's raw ``conversation`` is
   already ``[system, user, assistant]``. We keep that structure verbatim
   and simply flatten the assistant's ``content[]`` array (which
   interleaves ``text`` and ``image`` items) into a single string where
   each image is replaced by the ms-swift ``<image>`` placeholder. The
   images list is collected in strict left-to-right order across all
   messages (user first, then assistant), which is what ms-swift expects.

3. **One trajectory per (question_id) by default** — the pilot ran 4
   trials per question; keeping all 4 accepted trials would over-weight
   easy questions. We rank the accepted trials per question by (fewest
   iterations first, then longest thinking text) and keep just the top-1
   unless ``--keep_all_trials`` is passed.

4. **Absolute image paths** — the teacher wrote sandbox crops under
   ``./_sandbox_tmp/full/*.jpg`` (relative to the safety_sft/ dir);
   we prefix them with the absolute path so ms-swift's dataloader can
   read them regardless of the training script's CWD.

5. **Optional train/val split** — ``--val_ratio 0.02`` (default) carves a
   small held-out slice, useful for eyeballing intermediate checkpoints.

Usage
-----
    python convert_to_swift_sft.py \\
        --in_path      ./output/full_trajectories.jsonl \\
        --out_train    ./output/thyme_safety_sft_train.jsonl \\
        --out_val      ./output/thyme_safety_sft_val.jsonl \\
        --stats_md     ./output/thyme_safety_sft_stats.md \\
        --val_ratio    0.02 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Any


_HERE = os.path.dirname(os.path.abspath(__file__))

THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


def _format_ok(text: str) -> bool:
    """Same rule as pilot_check.py — exactly one <think>...</think> followed
    by exactly one <answer>...</answer>."""
    if len(THINK_RE.findall(text)) != 1:
        return False
    if len(ANSWER_RE.findall(text)) != 1:
        return False
    t_end = text.rfind("</think>")
    a_start = text.rfind("<answer>")
    return t_end != -1 and a_start != -1 and a_start > t_end


def _abs_image_path(p: str) -> str:
    """Sandbox crops in the raw jsonl look like './_sandbox_tmp/full/xxx.jpg'.
    Anchor them to the safety_sft/ directory so the swift dataloader can find
    them regardless of the training script's CWD.
    """
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(_HERE, p))


def _flatten_assistant_content(content_items: list[dict], images_out: list[str]) -> str | None:
    """Concatenate the interleaved text/image content items of the assistant
    turn into a single string with ``<image>`` placeholders in place of the
    image items, and append the absolute image paths to ``images_out`` in
    order.

    Note: any literal ``<image>`` that already appears inside a text item is
    stripped, so the final count of placeholders is exactly one per image
    item. This matters because the initial user text produced by Thyme's
    ``_build_initial_user_text`` already contains a literal ``<image>`` for
    the original meme; if we did not strip it we would double-count and end
    up with more placeholders than actual images.

    Returns None if any image file is missing on disk.
    """
    parts: list[str] = []
    for it in content_items:
        if it["type"] == "text":
            parts.append(it["text"].replace("<image>", ""))
        elif it["type"] == "image":
            abs_p = _abs_image_path(it["image"])
            if not os.path.isfile(abs_p):
                return None
            images_out.append(abs_p)
            parts.append("<image>")
        else:
            return None  # unknown modality; be strict
    return "".join(parts)


def _flatten_user_content(content_items: list[dict], images_out: list[str]) -> str | None:
    """Same idea as ``_flatten_assistant_content`` but for the initial user
    turn, which is exactly one image followed by the question text. Any
    literal ``<image>`` in the text is stripped for the same reason (see
    ``_flatten_assistant_content``'s docstring)."""
    parts: list[str] = []
    for it in content_items:
        if it["type"] == "image":
            abs_p = _abs_image_path(it["image"])
            if not os.path.isfile(abs_p):
                return None
            images_out.append(abs_p)
            parts.append("<image>")
        elif it["type"] == "text":
            parts.append(it["text"].replace("<image>", ""))
        else:
            return None
    return "".join(parts)


def _convert_one(r: dict, override_system: str | None = None) -> dict | None:
    """Convert a single trajectory record. Returns None if it fails any
    filter or has broken content.

    Args:
        r: one raw trajectory dict as written by run_teacher_distill.py.
        override_system: if not None, replace the sample's ``system`` message
            content with this string (still stripped of any literal
            ``<image>``). We use this to swap the STRONG safety prompt used
            for teacher rollouts with the RELAXED prompt that RL will use,
            so the SFT stage does not train the student to depend on prompt
            details that RL will not provide.
    """
    # Filter 1: no crash.
    if r.get("error"):
        return None

    # Filter 2: teacher got the right answer.
    if r.get("predicted") != r.get("gt_label"):
        return None

    # Filter 3: at least one sandbox image was actually returned.
    if int(r.get("n_sandbox_images", 0)) < 1:
        return None

    # Filter 4: strict <think>/<answer> format.
    if not _format_ok(r.get("raw_assistant_text", "") or ""):
        return None

    conv = r.get("conversation", [])
    if not conv or len(conv) < 3:
        return None
    if conv[0]["role"] != "system" or conv[1]["role"] != "user" or conv[2]["role"] != "assistant":
        return None

    # Extract system text. Strip any literal ``<image>`` that appears purely
    # as documentation inside the system prompt (our safety prompt has one at
    # line ~26 describing the sandbox return format). System never carries an
    # actual image, so removing the literal token is safe and keeps the total
    # placeholder count consistent with len(images).
    sys_items = conv[0].get("content", [])
    if len(sys_items) != 1 or sys_items[0]["type"] != "text":
        return None
    # If the caller provided an override, use it; otherwise keep the prompt
    # the teacher actually saw. Either way, strip literal ``<image>`` tokens.
    raw_system = override_system if override_system is not None else sys_items[0]["text"]
    system_text = raw_system.replace("<image>", "")

    # Extract user + assistant, collecting images in order.
    images: list[str] = []
    user_text = _flatten_user_content(conv[1].get("content", []), images)
    if user_text is None:
        return None
    asst_text = _flatten_assistant_content(conv[2].get("content", []), images)
    if asst_text is None:
        return None

    if len(images) < 1:
        return None  # must have at least the original image

    # Sanity: number of <image> tags in messages must match len(images).
    n_placeholders = user_text.count("<image>") + asst_text.count("<image>")
    if n_placeholders != len(images):
        return None

    return {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": asst_text},
        ],
        "images": images,
        # metadata we can filter/inspect on later; ms-swift ignores unknown keys
        "meta": {
            "question_id": r.get("question_id"),
            "subset": r.get("subset"),
            "gt_label": r.get("gt_label"),
            "trial_idx": r.get("trial_idx"),
            "n_iterations": r.get("n_iterations"),
            "n_code_blocks": r.get("n_code_blocks"),
            "n_sandbox_images": r.get("n_sandbox_images"),
            "thinking_chars": r.get("thinking_chars"),
        },
    }


def _rank_trials(records: list[dict]) -> list[dict]:
    """When multiple trials for a single question pass the filters, prefer:
      1. Fewer iterations (simpler is better; less risk of noise in later turns)
      2. Longer thinking text (more informative)
    """
    return sorted(
        records,
        key=lambda x: (x["meta"]["n_iterations"], -(x["meta"]["thinking_chars"] or 0)),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--out_train", required=True)
    ap.add_argument("--out_val", default=None,
                    help="if set, write a small held-out slice here")
    ap.add_argument("--stats_md", default=None,
                    help="write a markdown summary here")
    ap.add_argument("--keep_all_trials", action="store_true",
                    help="keep every accepted trial (default: 1 per question)")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--system_prompt_path", default=None,
                    help="if set, override every sample's system message with the "
                         "contents of this file. Use this to align the SFT data's "
                         "system prompt with the (relaxed) RL prompt, so the "
                         "student is not trained to depend on the strong prompt "
                         "the teacher was rolled out with.")
    args = ap.parse_args()

    override_system: str | None = None
    if args.system_prompt_path:
        with open(args.system_prompt_path, "r", encoding="utf-8") as f:
            override_system = f.read().rstrip()
        print(f"[system] overriding all samples' system with {args.system_prompt_path} "
              f"({len(override_system)} chars)")

    rng = random.Random(args.seed)

    n_total = 0
    n_error = 0
    n_wrong = 0
    n_no_sandbox = 0
    n_bad_format = 0
    n_bad_content = 0
    accepted: dict[str, list[dict]] = defaultdict(list)  # qid -> [records]
    subset_counts_kept: Counter = Counter()
    subset_counts_seen: Counter = Counter()

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
            subset_counts_seen[r.get("subset", "unknown")] += 1

            if r.get("error"):
                n_error += 1
                continue
            if r.get("predicted") != r.get("gt_label"):
                n_wrong += 1
                continue
            if int(r.get("n_sandbox_images", 0)) < 1:
                n_no_sandbox += 1
                continue
            if not _format_ok(r.get("raw_assistant_text", "") or ""):
                n_bad_format += 1
                continue

            conv_rec = _convert_one(r, override_system=override_system)
            if conv_rec is None:
                n_bad_content += 1
                continue

            accepted[r["question_id"]].append(conv_rec)

    # Deduplicate to 1 trial per question unless --keep_all_trials.
    final: list[dict] = []
    if args.keep_all_trials:
        for recs in accepted.values():
            final.extend(recs)
    else:
        for recs in accepted.values():
            best = _rank_trials(recs)[0]
            final.append(best)

    for rec in final:
        subset_counts_kept[rec["meta"]["subset"]] += 1

    rng.shuffle(final)

    # Optional train/val split.
    if args.out_val and args.val_ratio > 0:
        n_val = max(1, int(round(len(final) * args.val_ratio)))
        val_set = final[:n_val]
        train_set = final[n_val:]
    else:
        val_set = []
        train_set = final

    os.makedirs(os.path.dirname(os.path.abspath(args.out_train)) or ".", exist_ok=True)
    with open(args.out_train, "w", encoding="utf-8") as f:
        for rec in train_set:
            # ms-swift ignores 'meta' but we keep it for downstream inspection
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if val_set and args.out_val:
        with open(args.out_val, "w", encoding="utf-8") as f:
            for rec in val_set:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ---------------- Report ----------------
    def line(s: str = "") -> None:
        print(s)
        if md_lines is not None:
            md_lines.append(s)

    md_lines: list[str] | None = [] if args.stats_md else None

    line("# Convert-to-swift-SFT report")
    line("")
    line(f"- input:  `{args.in_path}`")
    line(f"- train:  `{args.out_train}` ({len(train_set)} examples)")
    if val_set and args.out_val:
        line(f"- val:    `{args.out_val}` ({len(val_set)} examples)")
    line(f"- one_trial_per_question: {not args.keep_all_trials}")
    line("")
    line("## Filter funnel")
    line("")
    line(f"- total trajectories read : **{n_total}**")
    line(f"- dropped: error / crash   : {n_error}")
    line(f"- dropped: wrong verdict   : {n_wrong}")
    line(f"- dropped: no sandbox img  : {n_no_sandbox}")
    line(f"- dropped: bad format tags : {n_bad_format}")
    line(f"- dropped: bad content     : {n_bad_content}")
    line(f"- questions with >=1 pass  : **{len(accepted)}**")
    line(f"- final examples written   : **{len(final)}**")
    line("")
    line("## Per-subset final counts")
    line("")
    for k in sorted(subset_counts_seen):
        seen = subset_counts_seen[k]
        kept = subset_counts_kept[k]
        line(f"- `{k:7s}`  seen={seen:5d}  kept={kept:5d}  ({(kept/max(seen,1))*100:.1f}%)")
    line("")

    if args.stats_md and md_lines is not None:
        os.makedirs(os.path.dirname(os.path.abspath(args.stats_md)) or ".", exist_ok=True)
        with open(args.stats_md, "w", encoding="utf-8") as f:
            f.write("\n".join(md_lines))
        print(f"[md] wrote {args.stats_md}")


if __name__ == "__main__":
    main()
