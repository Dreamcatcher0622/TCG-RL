"""TFA offline probe — step 1a: single-sample end-to-end sanity check.

Goal
----
Verify the *plumbing* of the Teacher-Forced Visual Ablation (TFA) reward
proposed in ``教师强制视觉消融奖励_实现文档.md`` on a single rollout:

  1. Load a Qwen2.5-VL policy checkpoint (defaults to the base
     Qwen2.5-VL-7B-Instruct so we do NOT pay for the full RL ckpt).
  2. Take ONE trajectory from an existing eval jsonl
     (``stage2_cot.jsonl``) whose ``n_code_calls >= 1``, and locate its
     sandbox crop under ``_sandbox_eval/<TAG>/<subset>/qid_<qid>/``.
  3. Build the multi-turn chat exactly like the RL rollout did:
        system + user(orig_img + text)
        + assistant(<think>...<code>...</code>)
        + user(<sandbox_output> + CROP_IMG + </sandbox_output>)
        + assistant(<think>...</think><answer>...</answer>)
  4. Encode twice through the SAME Qwen2.5-VL template:
        (a) orig  : with the real crop
        (b) cf    : with a white image of the SAME (W,H) as the real crop
     Assert ``input_ids`` / ``attention_mask`` / ``image_grid_thw`` are
     IDENTICAL between (a) and (b). Only ``pixel_values`` differ.
  5. Run TWO no-grad forwards, compute per-token log-prob of the ORIGINAL
     ``y_j`` under each condition, subtract:
        D_j = orig_logp[j] - cf_logp[j]
  6. Locate the assistant-turn-2 (the "post-tool" reasoning) token span in
     the sequence, then for each K ∈ {64, 128, 256, full}, report
        D_t^(K) = mean(D_j) over j in that window with K cap.
  7. Print timings, shape assertions, and the four D_t values.

This script is deliberately *self-contained*:
  - No training-loop coupling. Does NOT import ``grpo_trainer``.
  - CPU-side sandbox is not called (we already have the crop on disk).
  - Judge / side-car are NOT invoked (that's step 1c).

Success criterion for step 1a
-----------------------------
  * Shapes match between orig and cf encoded inputs (image token count
    invariance).
  * ``D_j`` tensor is finite (no NaN/Inf).
  * At least one of the K windows produces a non-zero D_t (i.e. the crop
    actually affects some post-tool tokens).

Usage
-----
    python cf_faith/tfa_offline_probe.py \
        --model /path/to/Qwen2.5-VL-7B-Instruct \
        --rollout_jsonl /tmp/step1_harmc.jsonl \
        --sandbox_root ./_sandbox_eval/<run_tag>/harmc \
        --question_file "${VISCOT_ROOT}/data/questions/harmc_test.jsonl" \
        --system_prompt ./prompt_safety_rl.txt \
        [--sample_qid harmc_test_000000]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path bootstrap: make Thyme's in-repo ms-swift importable.
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent                # .../cf_faith
_SFT_DIR = _THIS_DIR.parent                                # .../safety_sft
_THYME_ROOT = _SFT_DIR.parent.parent                       # repo root
if str(_THYME_ROOT) not in sys.path:
    sys.path.insert(0, str(_THYME_ROOT))
# Also expose the safety_sft dir so ``cf_faith.tfa_core`` is importable
# as a package (probe / scan share the same TFA core primitives).
if str(_SFT_DIR) not in sys.path:
    sys.path.insert(0, str(_SFT_DIR))

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Regex for parsing rollout raw_text.
# ---------------------------------------------------------------------------
_CODE_RE = re.compile(
    r"<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>",
    re.IGNORECASE,
)
_SBOX_RE = re.compile(r"<sandbox_output>([\s\S]*?)</sandbox_output>", re.IGNORECASE)
_THINK_RE = re.compile(r"<think>([\s\S]*?)</think>", re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>([\s\S]*?)</answer>", re.IGNORECASE)
_USER_IMG_PATH_RE = re.compile(
    r'### User Image Path:\s*"([^"]+)"', re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Rollout structure that the probe needs.
# ---------------------------------------------------------------------------
@dataclass
class ToolTurn:
    """One <code>...</code> + <sandbox_output>[img] slice."""
    code_body: str
    crop_image_path: str          # actual crop on disk


@dataclass
class ParsedTrajectory:
    """Everything the probe needs to rebuild the multi-turn chat."""
    qid: str
    orig_image_path: str          # original meme
    system_prompt: str
    user_text: str                # first user turn, with `<image>\n...`
    tool_turns: List[ToolTurn]    # each with code + crop path
    assistant_texts: List[str]    # length == len(tool_turns) + 1
    # assistant_texts[i] is what the model emitted at round i.
    # For the terminal round (index -1), it contains <answer>...</answer>.
    # For pre-tool rounds (index 0..T-1) it ends with </code>.


# ---------------------------------------------------------------------------
# Trajectory parsing.
# ---------------------------------------------------------------------------
def _split_assistant_rounds(raw_text: str) -> Tuple[List[str], List[str]]:
    """Split ``raw_text`` into (assistant_texts, sandbox_texts).

    In eval_safety_meme.py the final ``raw_text`` is
        (assistant_0)(<sandbox_output>[sandbox image]</sandbox_output>)
        (assistant_1)(<sandbox_output>...</sandbox_output>) ...
        (assistant_final)
    We split by ``<sandbox_output>...</sandbox_output>`` boundaries so the
    N sandbox blocks separate N+1 assistant chunks.
    """
    sandbox_matches = list(_SBOX_RE.finditer(raw_text))
    assistant_texts: List[str] = []
    sandbox_bodies: List[str] = []
    cursor = 0
    for m in sandbox_matches:
        assistant_texts.append(raw_text[cursor : m.start()])
        sandbox_bodies.append(m.group(1))
        cursor = m.end()
    assistant_texts.append(raw_text[cursor:])
    return assistant_texts, sandbox_bodies


def parse_trajectory(
    row: dict,
    system_prompt: str,
    sandbox_root: str,
    question_file_index: Optional[dict] = None,
) -> ParsedTrajectory:
    """Recover images + assistant chunks from a stage2_cot.jsonl row.

    Parameters
    ----------
    row
        one dict from stage2_cot.jsonl (must have ``raw_text``,
        ``question_id``, ``n_code_calls`` >= 1).
    system_prompt
        Content of ``prompt_safety_rl.txt`` (the SFT/RL system).
    sandbox_root
        Directory containing ``qid_<qid>/`` folders with the crop images
        that this rollout produced. We resolve
        ``sandbox_root/qid_<qid>/crop_*.jpg``.
    question_file_index
        Optional {qid -> image_path} map from the test set (harmc_test.jsonl).
        Used as a fallback if the raw_text has no ``### User Image Path:``.
    """
    qid = row["question_id"]
    raw_text = row.get("raw_text", "") or ""
    n_code = int(row.get("n_code_calls") or 0)
    if n_code <= 0:
        raise ValueError(f"[tfa] qid={qid} has no <code> — skip.")

    # -- 1. locate the original meme image path ----------------------------
    m = _USER_IMG_PATH_RE.search(raw_text)
    orig_image_path = m.group(1) if m else ""
    if not orig_image_path and question_file_index is not None:
        orig_image_path = question_file_index.get(qid, "")
    if not orig_image_path or not os.path.isfile(orig_image_path):
        raise FileNotFoundError(
            f"[tfa] qid={qid}: cannot resolve orig image path "
            f"(from raw_text or question_file_index).  raw={orig_image_path!r}"
        )

    # -- 2. split raw_text into assistant / sandbox turns -------------------
    assistant_texts, _sandbox_bodies = _split_assistant_rounds(raw_text)
    n_sandbox = len(assistant_texts) - 1
    if n_sandbox <= 0:
        raise ValueError(
            f"[tfa] qid={qid}: raw_text has {n_sandbox} sandbox turns "
            f"but n_code_calls={n_code}; corrupt jsonl?"
        )

    # -- 3. locate the crops on disk ---------------------------------------
    case_dir = os.path.join(sandbox_root, f"qid_{qid}")
    if not os.path.isdir(case_dir):
        raise FileNotFoundError(f"[tfa] no sandbox dir: {case_dir}")

    # eval_safety_meme.py writes to <SANDBOX_ROOT>/qid_<qid>/*.jpg without
    # any per-round index. If the model calls the sandbox multiple times we
    # would need to reconstruct the order; for step 1a we require n_code=1
    # and pick the only file.
    crop_files = sorted(
        p for p in Path(case_dir).iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not crop_files:
        raise FileNotFoundError(f"[tfa] no crop files under {case_dir}")

    # For step 1a: constrain to n_code == 1 to remove ambiguity.
    if n_code > 1:
        raise NotImplementedError(
            f"[tfa][step1a] qid={qid} has n_code_calls={n_code} > 1. "
            f"Step 1a requires exactly 1 tool call. Skip this sample or "
            f"pick another. (Multi-call ordering is a step 1b concern.)"
        )
    if len(crop_files) > 1:
        # Just take the first one; log so the user knows.
        print(
            f"[tfa][warn] {case_dir} has {len(crop_files)} crops for a "
            f"single-tool rollout; using {crop_files[0].name}.",
            flush=True,
        )
    crop_path = str(crop_files[0])

    # -- 4. extract the code body (best-effort; we don't need it to run) --
    code_bodies = _CODE_RE.findall(raw_text)
    code_body = code_bodies[0].strip() if code_bodies else ""

    tool_turns = [ToolTurn(code_body=code_body, crop_image_path=crop_path)]

    # -- 5. reconstruct the first user text ---------------------------------
    # We use the *stored* user path so the encoded text matches training.
    user_text = _build_user_text(orig_image_path)

    return ParsedTrajectory(
        qid=qid,
        orig_image_path=orig_image_path,
        system_prompt=system_prompt,
        user_text=user_text,
        tool_turns=tool_turns,
        assistant_texts=assistant_texts,
    )


def _build_user_text(image_path: str) -> str:
    """VERBATIM copy of build_rl_dataset._build_initial_user_text."""
    try:
        with Image.open(image_path) as img:
            sz = f"{img.width}x{img.height}"
    except Exception:
        sz = "Unable to determine"

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


# ---------------------------------------------------------------------------
# Message construction (Thyme-style multi-turn with sandbox tool response).
# ---------------------------------------------------------------------------
def build_messages(
    traj: ParsedTrajectory,
    crop_override_paths: Optional[List[str]] = None,
) -> Tuple[List[dict], List[str]]:
    """Rebuild the chat exactly like the RL trainer produced it.

    IMPORTANT: In Thyme's RL rollout (see grpo_trainer.py L1114/L1176) the
    sandbox output is **concatenated into the same assistant message**,
    NOT emitted as an independent user turn. So the chat has EXACTLY:

        system : prompt_safety_rl.txt
        user   : "<image>\n..."   (original meme)
        assistant : "<think>[reasoning]<code>...</code>
                     <sandbox_output><image></sandbox_output>
                     [more reasoning]</think><answer>...</answer>"

    The order of ``<image>`` placeholders in the flattened text is
    [original_meme, crop_1, crop_2, ...]; ms-swift's Template consumes
    them lazily from ``images`` in that same order.

    ``crop_override_paths`` (list of length ``len(tool_turns)``) lets us
    swap the crop image path per tool turn; used to inject a white image
    of the same (W,H) for the counterfactual condition.
    """
    crops = crop_override_paths
    if crops is None:
        crops = [tt.crop_image_path for tt in traj.tool_turns]
    assert len(crops) == len(traj.tool_turns), (
        f"crop_override_paths length {len(crops)} != {len(traj.tool_turns)}"
    )

    # The full assistant message = concatenation of:
    #   assistant_texts[0] (ends with </code>)
    #   + <sandbox_output><image></sandbox_output>   (for tool 0)
    #   + assistant_texts[1] (ends with </code>)     (if any)
    #   + <sandbox_output><image></sandbox_output>   (for tool 1)
    #   + ...
    #   + assistant_texts[-1] (final segment with </answer>)
    #
    # We rebuild it exactly as the RL trainer would have written it.
    parts: List[str] = []
    for t in range(len(traj.tool_turns)):
        a_text = traj.assistant_texts[t]
        # vLLM stopped at </code> so the raw_text may lack it; ensure it's
        # present so the final assistant string is a coherent trajectory.
        if not a_text.rstrip().endswith("</code>") and "<code>" in a_text:
            a_text = a_text + "</code>"
        parts.append(a_text)
        parts.append("<sandbox_output><image></sandbox_output>")
    parts.append(traj.assistant_texts[-1])
    full_assistant_content = "".join(parts)

    messages: List[dict] = [
        {"role": "system", "content": traj.system_prompt},
        {"role": "user", "content": traj.user_text},
        {"role": "assistant", "content": full_assistant_content},
    ]

    # Images in order: [orig_meme, crop_1, crop_2, ...].
    images = [traj.orig_image_path] + list(crops)
    return messages, images


# ---------------------------------------------------------------------------
# White-image cache, window localization, teacher-forced logp, D_t —
# all shared with the training-time TFA path. See cf_faith/tfa_core.py.
# ---------------------------------------------------------------------------
from cf_faith.tfa_core import (  # noqa: E402
    get_white_image_of_same_size,
    locate_post_tool_windows,
    per_token_logp_of_input_ids,
    compute_D_t_over_K,
)


# ---------------------------------------------------------------------------
# Template encoding (offline-probe specific: rebuilds the InferRequest from
# scratch. Training does not need this because ``inputs[i]['messages']`` is
# already prepared by the rollout loop.)
# ---------------------------------------------------------------------------
def encode_via_swift_template(
    messages: List[dict],
    image_paths: List[str],
    template,
    tokenizer,
):
    """Turn (messages, image_paths) into a batch dict ready for model.forward.

    Uses ms-swift's ``Template`` API which is what the trainer uses too.
    """
    from swift.llm.template.template_inputs import InferRequest

    req = InferRequest(messages=messages, images=image_paths)
    encoded = template.encode(req)                              # dict per-sample
    batch = template.data_collator([encoded])                   # list -> batch
    return encoded, batch


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def _load_question_index(path: str) -> dict:
    if not path or not os.path.isfile(path):
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            qid = r.get("question_id")
            ip = r.get("image_path")
            if qid and ip:
                out[qid] = ip
    return out


def pick_sample(
    rollout_jsonl: str, want_qid: Optional[str] = None
) -> dict:
    """Return the first row with n_code_calls == 1 (and matching qid if given)."""
    with open(rollout_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("n_code_calls") != 1:
                continue
            if want_qid and r.get("question_id") != want_qid:
                continue
            return r
    raise RuntimeError(
        f"No row with n_code_calls==1"
        + (f" and qid={want_qid}" if want_qid else "")
        + f" in {rollout_jsonl}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.environ.get("MODEL", "./models/Qwen2.5-VL-7B-Instruct"),
        help="Policy checkpoint to score under. Step 1a defaults to the "
             "base Qwen2.5-VL-7B-Instruct so we don't pay for the full "
             "RL ckpt during plumbing verification.",
    )
    ap.add_argument("--rollout_jsonl", required=True,
                    help="Path to a stage2_cot.jsonl produced by "
                         "eval_safety_meme.py (needs raw_text + "
                         "question_id + n_code_calls).")
    ap.add_argument("--sandbox_root", required=True,
                    help="Directory containing qid_<qid>/ subfolders with "
                         "the sandbox crop images that this rollout "
                         "produced (e.g. _sandbox_eval/v21-*/harmc).")
    ap.add_argument("--question_file", default="",
                    help="viscot-harmeme unified questions jsonl; used as "
                         "fallback for image_path when raw_text lacks the "
                         "### User Image Path line.")
    ap.add_argument("--system_prompt", default=str(_SFT_DIR / "prompt_safety_rl.txt"))
    ap.add_argument("--sample_qid", default="",
                    help="Optional specific qid to probe.")
    ap.add_argument("--white_cache_dir",
                    default=str(_SFT_DIR / "_tfa_white_cache"))
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--K_list", default="64,128,256,-1",
                    help="Comma-separated K values (nats/token windows). "
                         "-1 means 'full window'.")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    K_list = [int(x) for x in args.K_list.split(",")]

    # ------------------------------------------------------------------
    # 1. Load rollout row + question index.
    # ------------------------------------------------------------------
    q_index = _load_question_index(args.question_file)
    row = pick_sample(args.rollout_jsonl, args.sample_qid or None)
    qid = row["question_id"]
    print(f"[tfa] picked qid = {qid}", flush=True)

    system_prompt = open(args.system_prompt, "r", encoding="utf-8").read().rstrip() + "\n"

    traj = parse_trajectory(
        row=row,
        system_prompt=system_prompt,
        sandbox_root=args.sandbox_root,
        question_file_index=q_index,
    )
    print(f"[tfa] orig meme  : {traj.orig_image_path}")
    print(f"[tfa] crop image : {traj.tool_turns[0].crop_image_path}")
    print(f"[tfa] # tools    : {len(traj.tool_turns)}")
    print(f"[tfa] # asst chunks : {len(traj.assistant_texts)}")

    # ------------------------------------------------------------------
    # 2. Build white image of the same size.
    # ------------------------------------------------------------------
    white_path = get_white_image_of_same_size(
        traj.tool_turns[0].crop_image_path, args.white_cache_dir
    )
    with Image.open(traj.tool_turns[0].crop_image_path) as im:
        orig_wh = im.size
    with Image.open(white_path) as im:
        white_wh = im.size
    assert orig_wh == white_wh, f"{orig_wh} vs {white_wh}"
    print(f"[tfa] white img  : {white_path}  (same size {white_wh})")

    # ------------------------------------------------------------------
    # 3. Load model + template.
    # ------------------------------------------------------------------
    print(f"[tfa] loading model {args.model} in {dtype} ...", flush=True)
    t0 = time.time()
    from swift.llm import get_model_tokenizer, get_template
    model, tokenizer = get_model_tokenizer(
        args.model,
        torch_dtype=dtype,
        device_map={"": device},
        attn_impl="flash_attn",
    )
    model.eval()
    print(f"[tfa] model loaded in {time.time()-t0:.1f}s", flush=True)

    # Template.  For Qwen2.5-VL the meta lookup uses model_meta from the
    # tokenizer that get_model_tokenizer just attached.
    template = get_template(
        model.model_meta.template,
        tokenizer,
        default_system=None,
        max_length=8192,
    )
    # We need mode='train' (is_training=True) so that the FINAL assistant
    # message content (which contains our <think>...<code>...<sandbox_output>
    # ...</think><answer>...) is actually tokenized into ``input_ids``.
    # In 'pt' mode the template treats the last assistant as an empty
    # generation prompt and stops right at '<|im_start|>assistant\n'.
    template.set_mode("train")
    # ``_get_position_ids`` needs ``self.model`` to reach the base model's
    # ``get_rope_index``. The trainer sets this via a hook; we set it
    # directly for offline use.
    template.model = model

    # ------------------------------------------------------------------
    # 4. Encode ORIG and CF batches.
    # ------------------------------------------------------------------
    msgs_orig, imgs_orig = build_messages(traj, crop_override_paths=None)
    msgs_cf,   imgs_cf   = build_messages(traj, crop_override_paths=[white_path])

    t1 = time.time()
    _enc_orig, batch_orig = encode_via_swift_template(msgs_orig, imgs_orig, template, tokenizer)
    _enc_cf,   batch_cf   = encode_via_swift_template(msgs_cf,   imgs_cf,   template, tokenizer)
    print(f"[tfa] template.encode took {time.time()-t1:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # 5. Assert equal token-level structure.
    # ------------------------------------------------------------------
    ids_o = batch_orig["input_ids"]
    ids_c = batch_cf["input_ids"]
    print(f"[tfa] input_ids: orig={tuple(ids_o.shape)}  cf={tuple(ids_c.shape)}")
    assert ids_o.shape == ids_c.shape, (
        f"input_ids shape mismatch: {ids_o.shape} vs {ids_c.shape} "
        f"(white image must preserve token count — check crop size)"
    )
    if not torch.equal(ids_o, ids_c):
        n_diff = (ids_o != ids_c).sum().item()
        raise AssertionError(
            f"input_ids differ at {n_diff} positions — likely because "
            f"the two crops had different sizes, or the template chose "
            f"different image-token counts. Aborting."
        )
    if "image_grid_thw" in batch_orig:
        assert torch.equal(
            batch_orig["image_grid_thw"], batch_cf["image_grid_thw"]
        ), "image_grid_thw differ — pixel dims must match"
    print("[tfa] ✔ input_ids / attention_mask / image_grid_thw all equal.")

    # ---- Diagnostic dumps (help debug D_t ~ 0) ----
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    def _decode_range(ids: torch.Tensor, s: int, e: int) -> str:
        return text_tok.decode(
            ids[s:e].tolist(), skip_special_tokens=False
        )

    print(f"[tfa][diag] image_grid_thw orig = {batch_orig.get('image_grid_thw')}")
    print(f"[tfa][diag] image_grid_thw cf   = {batch_cf.get('image_grid_thw')}")
    pv_o = batch_orig.get("pixel_values")
    pv_c = batch_cf.get("pixel_values")
    if pv_o is not None and pv_c is not None:
        print(f"[tfa][diag] pixel_values shape: orig={tuple(pv_o.shape)}  cf={tuple(pv_c.shape)}")
        # If white image is truly white, cf pixel_values should have a
        # very different distribution from orig.
        diff = (pv_o.float() - pv_c.float()).abs()
        print(f"[tfa][diag] |Δ pixel_values|: mean={diff.mean().item():.4f} "
              f"max={diff.max().item():.4f} nonzero_frac="
              f"{(diff > 1e-3).float().mean().item():.4f}")
    print(f"[tfa][diag] full assistant tail decoded  = "
          f"{_decode_range(ids_o[0], max(0, ids_o.shape[1]-120), ids_o.shape[1])!r}")
    # find sandbox_output_close in tokens
    ids_list = ids_o[0].tolist()
    # locate </sandbox_output> by decoding piece-by-piece
    dec_pieces = [text_tok.decode([tid], skip_special_tokens=False) for tid in ids_list]
    running = ""
    close_tok_end = None
    for i, p in enumerate(dec_pieces):
        running += p
        if running.rstrip().endswith("</sandbox_output>") and close_tok_end is None:
            close_tok_end = i + 1
            break
    print(f"[tfa][diag] first </sandbox_output> ends at token index = {close_tok_end}")
    if close_tok_end is not None:
        print(f"[tfa][diag] tokens[{close_tok_end}:{close_tok_end+60}] decoded = "
              f"{_decode_range(ids_o[0], close_tok_end, close_tok_end+60)!r}")

    # ------------------------------------------------------------------
    # 6. Locate the post-tool window(s) in tokens.
    # ------------------------------------------------------------------
    windows = locate_post_tool_windows(
        input_ids=ids_o[0], tokenizer=tokenizer,
        n_tools=len(traj.tool_turns),
    )
    for t, (s, e) in enumerate(windows):
        n_win_tok = e - s
        print(f"[tfa] window t={t}: tokens[{s}:{e}]  size={n_win_tok}")

    # ------------------------------------------------------------------
    # 7. Two teacher-forced forwards.
    # ------------------------------------------------------------------
    print("[tfa] forward-orig ...", flush=True)
    ta = time.time()
    logps_orig = per_token_logp_of_input_ids(model, batch_orig, device)
    print(f"[tfa]   done in {time.time()-ta:.2f}s  shape={tuple(logps_orig.shape)}")
    print("[tfa] forward-cf   ...", flush=True)
    tb = time.time()
    logps_cf = per_token_logp_of_input_ids(model, batch_cf, device)
    print(f"[tfa]   done in {time.time()-tb:.2f}s  shape={tuple(logps_cf.shape)}")

    assert logps_orig.shape == logps_cf.shape

    # ---- Global logp diff diagnostic ----
    diff_all = (logps_orig - logps_cf).abs()
    print(f"[tfa][diag] |Δ logp| across ALL {logps_orig.shape[0]} positions: "
          f"mean={diff_all.mean().item():.6f}  max={diff_all.max().item():.6f}  "
          f"n_nonzero>1e-4: {(diff_all > 1e-4).sum().item()}")

    # ------------------------------------------------------------------
    # 8. D_t across K windows.
    # ------------------------------------------------------------------
    print("\n[tfa] ==== D_t results (nats/token, higher = more visual-dependent) ====")
    for t, win in enumerate(windows):
        row_D = compute_D_t_over_K(logps_orig, logps_cf, win, K_list)
        row_str = "  ".join(
            f"K={K if K > 0 else 'full'}:{v:+.4f}"
            for K, v in row_D.items()
        )
        print(f"  tool t={t}   window=({win[0]},{win[1]}, n={win[1]-win[0]})")
        print(f"    {row_str}")

    # Sanity: at least one non-zero
    all_zero = all(
        abs(compute_D_t_over_K(logps_orig, logps_cf, win, K_list)[K_list[0]]) < 1e-9
        for win in windows
    )
    if all_zero:
        print("[tfa][warn] D_t is ~zero everywhere. "
              "Either the crop was uninformative for this sample, or "
              "the ablation had no effect. Consider trying another qid.")
    else:
        print("[tfa] ✔ D_t has non-trivial magnitude — plumbing looks OK.")

    print("[tfa] done.")


if __name__ == "__main__":
    main()
