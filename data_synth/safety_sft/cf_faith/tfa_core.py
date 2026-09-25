"""Teacher-Forced Visual Ablation (TFA) — shared core primitives.

All the pure functions below are used by BOTH:
  * ``cf_faith/tfa_offline_probe.py`` / ``tfa_offline_scan.py``
    (single-sample and batch offline probes verified in steps 1a / 1b).
  * ``swift/trainers/rlhf_trainer/grpo_trainer.py``
    (online per-step D_t computation inside the RL loop; see the
    ``TFA_ENABLE=1`` code path).

Rationale
---------
Keeping the offline probe and the training-time TFA reward on the SAME
implementation of:
  * white-image ablation (``get_white_image_of_same_size``)
  * post-tool token-window localization (``locate_post_tool_windows``)
  * teacher-forced per-token log-prob (``per_token_logp_of_input_ids``)
  * D_t computation across K windows (``compute_D_t_over_K``)

is what lets us claim: **the D_t observed offline (1b, mean≈0 nats/token,
std≈0.03) is the same statistic being optimized in training**.

None of these functions depend on ms-swift Trainer internals; they only
depend on the tokenizer / model / a batch dict shaped like what
``template.data_collator`` produces.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import torch
from PIL import Image


# ---------------------------------------------------------------------------
# White-image cache (thread-safe / multi-process-safe write).
# ---------------------------------------------------------------------------
_WHITE_CACHE: dict[Tuple[int, int], str] = {}


def get_white_image_of_same_size(crop_path: str, cache_dir: str) -> str:
    """Return a path to a pure-white RGB image with the same (W, H) as
    ``crop_path``.

    Cache is keyed by (W, H) — different tool crops with the same shape
    share the same white file, so a training run producing thousands of
    tool calls generates at most a few hundred distinct white PNGs.

    Robustness: the sandbox-produced crop may be a corrupt / empty file
    (we saw ``libpng error: bad parameters to zlib`` on some crops). We
    ``verify()`` the decode and fall back to a small default canvas so a
    single bad crop never crashes the whole TFA step.
    """
    w, h = 16, 16  # safe default
    try:
        with Image.open(crop_path) as im:
            im.verify()  # raises on corrupt files
        with Image.open(crop_path) as im:
            w, h = im.size
    except Exception as _e:
        print(f"[tfa][warn] get_white_image_of_same_size: cannot read "
              f"{crop_path!r} ({type(_e).__name__}: {_e}); falling back to "
              f"16x16 white.", flush=True)
    key = (w, h)
    if key in _WHITE_CACHE and os.path.isfile(_WHITE_CACHE[key]):
        return _WHITE_CACHE[key]
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"white_{w}x{h}.png")
    if not os.path.isfile(out):
        # Atomic write: create under a .tmp suffix, then rename. Two ranks
        # trying to create the same white file at once is fine — the
        # loser's rename is a no-op (dst already exists on POSIX-atomic FS)
        # or overwrites with an identical byte string.
        tmp = out + f".tmp.{os.getpid()}"
        Image.new("RGB", (w, h), (255, 255, 255)).save(tmp, format="PNG")
        try:
            os.replace(tmp, out)
        except Exception:
            # Rare: another process finished first; drop the tmp file.
            try:
                os.remove(tmp)
            except Exception:
                pass
    _WHITE_CACHE[key] = out
    return out


# ---------------------------------------------------------------------------
# Post-tool token window localization.
# ---------------------------------------------------------------------------
def locate_post_tool_windows(
    input_ids: torch.Tensor,          # (L,) int64/int32
    tokenizer,
    n_tools: int,
) -> List[Tuple[int, int]]:
    """Return the [start, end) TOKEN offsets of the "post-tool" reasoning
    window for each of the ``n_tools`` sandbox tool calls in ``input_ids``.

    Window semantics (see 教师强制视觉消融奖励_实现文档.md §3.2):
      start = first token AFTER the t-th ``</sandbox_output>``
      end   = smallest token position (> start) among any of
              {next ``<code>``, ``</think>``, ``<answer>``,
              end-of-sequence}.

    Important robustness detail: the system prompt AND the initial user
    prompt both contain literal ``<sandbox_output></sandbox_output>`` /
    ``<code>`` phrasing (as part of the tool-use instructions). We MUST
    NOT match those; only occurrences INSIDE the final assistant turn
    (after the last ``<|im_start|>assistant`` marker) are real tool
    calls. This was verified in step-1a debugging.

    Implementation: we build a char-level index by decoding each token
    individually. Qwen2.5-VL's ``</sandbox_output>``, ``<code>``,
    ``</think>``, ``<answer>`` are multi-token spans in BPE, so we can't
    rely on single-token id matching; substring search over the joined
    decoded text is the simplest correct approach.
    """
    ids = input_ids.tolist()

    # For Qwen2.5-VL, ``tokenizer`` returned by ms-swift may be a
    # Processor. The underlying fast text tokenizer sits at
    # ``tokenizer.tokenizer``.
    text_tok = getattr(tokenizer, "tokenizer", tokenizer)

    # cum_char_offsets[i] = char position where token i STARTS in the
    # joined decoded string.
    char_pieces = [
        text_tok.decode([tid], skip_special_tokens=False) for tid in ids
    ]
    joined = "".join(char_pieces)
    cum_char_offsets: List[int] = []
    c = 0
    for piece in char_pieces:
        cum_char_offsets.append(c)
        c += len(piece)
    cum_char_offsets.append(c)  # sentinel

    def char_to_first_token_after(char_pos: int) -> int:
        for i, off in enumerate(cum_char_offsets):
            if off >= char_pos:
                return i
        return len(cum_char_offsets) - 1

    def find_all(needle: str) -> List[int]:
        out: List[int] = []
        j = 0
        while True:
            k = joined.find(needle, j)
            if k < 0:
                break
            out.append(k)
            j = k + 1
        return out

    sbox_close = find_all("</sandbox_output>")
    code_open = find_all("<code>")
    think_close = find_all("</think>")
    ans_open = find_all("<answer>")

    # Restrict to matches INSIDE the final assistant turn so we don't
    # match the system prompt's example of "<sandbox_output>...".
    asst_header = "<|im_start|>assistant"
    asst_starts = find_all(asst_header)
    asst_head_char = asst_starts[-1] if asst_starts else 0

    def _after_header(positions: List[int]) -> List[int]:
        return [p for p in positions if p >= asst_head_char]

    sbox_close = _after_header(sbox_close)
    code_open = _after_header(code_open)
    think_close = _after_header(think_close)
    ans_open = _after_header(ans_open)

    if len(sbox_close) < n_tools:
        raise ValueError(
            f"[tfa] only found {len(sbox_close)} </sandbox_output> "
            f"delimiters inside the assistant turn but n_tools={n_tools}"
        )

    windows: List[Tuple[int, int]] = []
    for t in range(n_tools):
        start_char = sbox_close[t] + len("</sandbox_output>")
        candidates: List[int] = []
        for pos in code_open:
            if pos > start_char:
                candidates.append(pos); break
        for pos in think_close:
            if pos > start_char:
                candidates.append(pos); break
        for pos in ans_open:
            if pos > start_char:
                candidates.append(pos); break
        end_char = min(candidates) if candidates else len(joined)

        s_tok = char_to_first_token_after(start_char)
        e_tok = char_to_first_token_after(end_char)
        if e_tok <= s_tok:
            e_tok = min(len(ids), s_tok + 1)
        windows.append((s_tok, e_tok))

    return windows


# ---------------------------------------------------------------------------
# Teacher-forced per-token log-prob.
# ---------------------------------------------------------------------------
@torch.no_grad()
def per_token_logp_of_input_ids(
    model,
    batch: dict,
    device: torch.device,
) -> torch.Tensor:
    """Return shape (L-1,) float32-on-CPU tensor: logp of input_ids[1:L]
    under ``model``, using teacher forcing on ``input_ids``.

    Uses ``trl.trainer.utils.selective_log_softmax`` to avoid materializing
    a full (L, V) softmax; this matches what grpo_trainer's
    ``_get_per_token_logps`` does internally.

    All non-tensor entries in ``batch`` are passed through verbatim.
    """
    from trl.trainer.utils import selective_log_softmax

    to_dev: dict = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            to_dev[k] = v.to(device)
        else:
            to_dev[k] = v

    input_ids = to_dev["input_ids"]                    # (1, L)
    attention_mask = to_dev.get("attention_mask")
    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        pixel_values=to_dev.get("pixel_values"),
        image_grid_thw=to_dev.get("image_grid_thw"),
        use_cache=False,
    )
    logits = out.logits                                # (1, L, V)
    shifted_logits = logits[:, :-1, :]                 # (1, L-1, V)
    shifted_targets = input_ids[:, 1:]                 # (1, L-1)
    logps = selective_log_softmax(shifted_logits, shifted_targets)  # (1, L-1)
    return logps[0].detach().to(torch.float32).cpu()


# ---------------------------------------------------------------------------
# D_t over K windows.
# ---------------------------------------------------------------------------
def compute_D_t_over_K(
    orig_logps: torch.Tensor,        # (L-1,)
    cf_logps: torch.Tensor,          # (L-1,)
    window: Tuple[int, int],
    K_list: List[int],
) -> dict:
    """Return ``{K -> D_t (nats/token)}`` for a single tool call.

    NOTE on off-by-one: the token-level logp array has length ``L-1``
    (index j predicts ``input_ids[j+1]``). A window ``(s, e)`` given in
    ``input_ids`` coordinates therefore maps to logp indices
    ``(s - 1, e - 1)``, clipped at 0. The window elements correspond to
    positions in ``input_ids[s:e]`` which are the FIRST ``e - s`` tokens
    the model is predicting AFTER the tool observation.

    ``K < 0`` (or ``K == 0``, or ``K >= (e - s)``) means "use the full
    window".
    """
    s, e = window
    s = max(0, s - 1)
    e = max(s, e - 1)
    if e <= s:
        return {K: float("nan") for K in K_list}
    delta = (orig_logps[s:e] - cf_logps[s:e]).to(torch.float64)
    out: dict = {}
    n = delta.numel()
    for K in K_list:
        if K < 0 or K == 0 or K >= (e - s):
            out[K] = float(delta.mean().item()) if n > 0 else float("nan")
        else:
            out[K] = float(delta[:K].mean().item())
    return out


# ---------------------------------------------------------------------------
# Small helper for the training-time bucket + gate.
# ---------------------------------------------------------------------------
def bucket_D_t(D_t: float, eps: float) -> int:
    """Discretize a raw D_t (nats/token) into {0, +1}.

    New policy (20260804): TFA is a purely POSITIVE-reinforcement signal.
    A negative D_t (white-image ablation raised the model's log-prob)
    could mean either (a) the tool call was distractive, or (b) the CoT
    is already tool-agnostic (Step-2 hacking pattern). We do NOT want to
    penalise tool USE per se on the basis of that -- it collapses tool
    usage to zero, which is the very hack we're trying to prevent. So
    only ``D_t > eps`` (evidence the tool observation matters) gets +1;
    everything else -- including ``D_t < -eps`` -- is 0.

    See 教师强制视觉消融奖励_实现文档.md §4.2 for the original
    (three-bucket) semantics; the negative bucket is now retired.
    """
    if D_t != D_t:  # NaN
        return 0
    if D_t > eps:
        return 1
    return 0


def gated_tool_reward(
    D_t: float,
    eps: float,
    correct: bool,
    F_tau: float,
    F_high: float,
    F_low: float,
) -> float:
    """Full gated per-tool reward in [-1, 0, +1].

    Rule (文档 §4.2):
      +1  if  D_t >  eps  AND correct AND F_tau >= F_high
      -1  if  D_t >  eps  AND (NOT correct OR F_tau <= F_low)
       0  otherwise (D_t <= eps  OR  quality in the dead zone)

    Notes:
      * A negative D_t (tool observation was distractive) currently maps
        to 0, matching the doc's "does not represent value" caveat: we do
        NOT penalise a benign no-op crop that happens to reduce the
        model's certainty. The -1 case is reserved for HIGH D_t on a bad
        trajectory, which is the "confidently used a bad tool" hacking
        pattern we want to punish.
    """
    b = bucket_D_t(D_t, eps)
    if b <= 0:
        return 0.0
    # b == 1  ->  D_t > eps
    if correct and F_tau >= F_high:
        return 1.0
    if (not correct) or F_tau <= F_low:
        return -1.0
    return 0.0
