"""Counterfactual rollout client for CF-FaithGRPO tool-level reward.

Given an ORIGINAL trajectory produced by the training-time Thyme agent loop
(a sequence of ``<code>`` / ``<sandbox_output>`` round-trips ending in
``<answer>...</answer>``), this module produces N counterfactual trajectories
by "skipping" each of the N tool calls in turn.

For each ``skip_code_idx = t`` in ``{0, 1, ..., N-1}``:
    1. Rewind the assistant text to the point just BEFORE the t-th
       ``<code>`` opening tag.
    2. Re-run the agent loop from that point against a **side-car** vLLM
       server (Qwen2.5-VL-7B loaded with a FIXED base SFT checkpoint --
       no policy sync during training) until it emits ``</answer>``.
    3. Return the counterfactual CoT + pred for downstream FaithScore.

Design constraints
------------------
* **Reuse eval_safety_meme.py verbatim.** ``_openai_chat_once``,
  ``_extract_code_block``, ``_execute_code_in_sandbox``, verdict parsing,
  and ``extract_think_cot`` are all imported from that module. This
  guarantees the counterfactual trajectory has the same on-wire format as
  training and eval, so FaithScore stage-1 parsing is identical.

* **Fixed side-car.** The side-car serves a *base* SFT checkpoint, NOT
  the current-policy checkpoint. This is the "reference-policy
  counterfactual" simplification agreed with the user; the tool-level
  reward measures "did this tool help vs. what a REFERENCE policy would
  produce without it", which is still a valid faithfulness signal.

* **Zero training-side GPU.** All calls to the side-car are HTTP.

* **Batched.** ``rollout_counterfactuals_for_batch`` fires all N * B
  counterfactuals across the batch via a ThreadPoolExecutor so the
  side-car's continuous batcher stays saturated.

* **Defensive.** Every failure mode returns a neutral placeholder
  ``{"cot": "", "pred": "unknown", "status": "..."}`` and logs with the
  ``[cf-rollout]`` prefix; the caller can decide how to translate that
  into a reward.
"""
from __future__ import annotations

import base64
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from typing import Any, Optional

from PIL import Image

# --------------------------------------------------------------------------- #
# Make eval_safety_meme.py importable (single source of truth for the agent   #
# loop).                                                                       #
# --------------------------------------------------------------------------- #
_THIS_DIR = Path(__file__).resolve().parent
_SFT_DIR = _THIS_DIR.parent                            # data_synth/safety_sft
_THYME_ROOT = _SFT_DIR.parent.parent                   # repo root
_EVAL_DIR = _SFT_DIR / "eval"                          # data_synth/safety_sft/eval

# NOTE: ``eval/`` has no ``__init__.py`` (and adding one is risky because
# ``eval`` shadows the python builtin in some contexts). So we add the eval
# directory itself to sys.path and import ``eval_safety_meme`` as a
# top-level module.
for _p in (str(_SFT_DIR), str(_THYME_ROOT), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _log(tag: str, msg: str) -> None:
    print(f"[cf-rollout][{tag}] {msg}", flush=True)


_WARNED: set[str] = set()


def _log_once(tag: str, key: str, msg: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    _log(tag, msg)


# --------------------------------------------------------------------------- #
# Regex to find each <code>...</code> block in the assistant text.            #
# Identical to the training-time / eval-time regex.                            #
# --------------------------------------------------------------------------- #
_CODE_BLOCK_RE = re.compile(
    r"<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>",
    re.IGNORECASE,
)


def count_code_calls(assistant_text: str) -> int:
    """Return N = number of <code>...</code> blocks in the trajectory."""
    if not assistant_text:
        return 0
    return len(_CODE_BLOCK_RE.findall(assistant_text))


def find_code_starts(assistant_text: str) -> list[int]:
    """Return the char offset of every ``<code>`` opening tag.

    Used to slice the trajectory at "the point just before the t-th code
    call". We find the OPEN tag because the counterfactual should not have
    even seen the model choose to call the tool.
    """
    starts: list[int] = []
    if not assistant_text:
        return starts
    idx = 0
    while True:
        found = assistant_text.find("<code>", idx)
        if found < 0:
            break
        starts.append(found)
        idx = found + 6
    return starts


# --------------------------------------------------------------------------- #
# Image encoding                                                               #
# --------------------------------------------------------------------------- #
def _encode_image_path(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        _log("warn", f"cannot read image {image_path}: {e}")
        return None


# --------------------------------------------------------------------------- #
# Fixed message shown in place of a real sandbox output when the tool is      #
# "unavailable" in a counterfactual rollout. Kept short + assertive so the    #
# model does not waste tokens re-trying.                                      #
# --------------------------------------------------------------------------- #
CF_TOOL_UNAVAILABLE_MSG = "Tool unavailable. Do not call the tool again."


# --------------------------------------------------------------------------- #
# Prefix -> multi-turn messages replay.                                        #
# The Thyme agent-loop trajectory looks like:                                  #
#     <think>intro                                                             #
#     <code>...</code>                     (assistant turn ends here)          #
#     <sandbox_output>...</sandbox_output> (user turn: sandbox result)         #
#     ...intra-round think...              (next assistant turn resumes)       #
#     <code>...</code>                     (assistant turn ends here)          #
#     <sandbox_output>...</sandbox_output>                                     #
#     ...<answer>...</answer>                                                  #
# --------------------------------------------------------------------------- #
_CODE_TAG_RE = re.compile(r"<code>[\s\S]*?</code>", re.IGNORECASE)
_SANDBOX_TAG_RE = re.compile(
    r"<sandbox_output>[\s\S]*?</sandbox_output>", re.IGNORECASE
)


def split_prefix_into_rounds(
    prefix_text: str,
) -> tuple[str, list[tuple[str, str, str]]]:
    """Return (initial_think, [(before_code, code_block, sandbox_block), ...]).

    Boundary rules match the training-time agent loop:
      * An assistant turn ENDS at ``</code>``.
      * A user turn is exactly ``<sandbox_output>...</sandbox_output>``.
      * Text between ``</sandbox_output>`` and the NEXT ``<code>`` is the
        "intra-round think" of the NEXT assistant turn (its ``before_code``).

    Round 0's ``before_code`` is always empty because that prelude sits in
    ``initial_think`` (everything preceding the first ``<code>``).
    """
    if not prefix_text:
        return "", []

    code_matches = list(_CODE_TAG_RE.finditer(prefix_text))
    sbox_matches = list(_SANDBOX_TAG_RE.finditer(prefix_text))
    if not code_matches:
        return prefix_text, []

    initial_think = prefix_text[: code_matches[0].start()]
    rounds: list[tuple[str, str, str]] = []

    for i, cm in enumerate(code_matches):
        code_block = cm.group(0)

        sbox_block = ""
        for sm in sbox_matches:
            if sm.start() >= cm.end():
                sbox_block = sm.group(0)
                break

        if i == 0:
            before_code = ""
        else:
            prev_code = code_matches[i - 1]
            prev_sbox_end = prev_code.end()
            for sm in sbox_matches:
                if sm.start() >= prev_code.end():
                    prev_sbox_end = sm.end()
                    break
            before_code = prefix_text[prev_sbox_end : cm.start()]

        rounds.append((before_code, code_block, sbox_block))

    return initial_think, rounds


# --------------------------------------------------------------------------- #
# Sidecar client (singleton)                                                   #
# --------------------------------------------------------------------------- #
class SideCarClient:
    """Wraps an OpenAI-compat vLLM endpoint plus a couple of health checks.

    A single instance is shared across the whole process. It is created
    lazily on first use so a broken side-car cannot prevent training from
    starting.
    """

    _singleton_lock = threading.Lock()
    _singleton: Optional["SideCarClient"] = None

    @classmethod
    def get_singleton(cls) -> "SideCarClient":
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    def __init__(self) -> None:
        self.url = os.environ.get("CF_SIDECAR_URL", "").strip()
        self.model = os.environ.get("CF_SIDECAR_MODEL", "").strip()
        self.api_key = os.environ.get("CF_SIDECAR_API_KEY", "EMPTY") or "EMPTY"
        self.timeout = float(os.environ.get("CF_SIDECAR_TIMEOUT", "180") or "180")
        self.max_retry = int(os.environ.get("CF_SIDECAR_MAX_RETRY", "2") or "2")
        self.max_rounds = int(os.environ.get("CF_MAX_ROUNDS", "6") or "6")
        self.max_new_tokens = int(
            os.environ.get("CF_MAX_NEW_TOKENS", "1536") or "1536"
        )
        self.sandbox_root = os.environ.get(
            "CF_SANDBOX_ROOT",
            str(_SFT_DIR / "_sandbox_cf"),
        )
        os.makedirs(self.sandbox_root, exist_ok=True)

        self.enabled = True
        self._disable_reason: Optional[str] = None
        self._client = None
        self._health_checked = False
        self._exec_sandbox = None

        if not self.url or not self.model:
            self.enabled = False
            missing = []
            if not self.url:
                missing.append("CF_SIDECAR_URL")
            if not self.model:
                missing.append("CF_SIDECAR_MODEL")
            self._disable_reason = (
                f"missing env vars: {', '.join(missing)}. "
                "Tool-level faith reward will be 0 for all samples. "
                "To enable, export CF_SIDECAR_URL and CF_SIDECAR_MODEL "
                "in rl.sh (see cf_faith/serve_judge_notes.md for how to "
                "start a side-car vLLM server on a separate machine)."
            )
            _log("skip", self._disable_reason)
            return

        _log(
            "init",
            f"side-car configured: {self.model} @ {self.url} "
            f"(max_rounds={self.max_rounds}, max_new_tokens={self.max_new_tokens}, "
            f"timeout={self.timeout}s, sandbox_root={self.sandbox_root})",
        )

    # ------------------------------------------------------------------ #
    def _lazy_build_client(self) -> bool:
        if self._client is not None and self._exec_sandbox is not None:
            return True
        try:
            from openai import OpenAI  # noqa: E402

            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.url,
                timeout=self.timeout,
            )
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"failed to build OpenAI client for side-car {self.url}: {e}"
            )
            _log("error", self._disable_reason)
            traceback.print_exc()
            return False

        try:
            from swift.trainers.sandbox import execute_code_in_sandbox  # noqa: E402

            self._exec_sandbox = execute_code_in_sandbox
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"cannot import swift.trainers.sandbox.execute_code_in_sandbox: {e}"
            )
            _log("error", self._disable_reason)
            traceback.print_exc()
            return False
        return True

    # ------------------------------------------------------------------ #
    def _health_check(self) -> bool:
        if self._health_checked:
            return self.enabled
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "reply with OK"}],
                temperature=0.0,
                max_tokens=8,
            )
            _ = resp.choices[0].message.content
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"side-car health check failed at {self.url}: {e}. "
                f"Verify with:  curl -sf {self.url}/models"
            )
            _log("error", self._disable_reason)
            self._health_checked = True
            return False
        self._health_checked = True
        _log("init", f"side-car health OK ({self.model} @ {self.url})")
        return True

    # ------------------------------------------------------------------ #
    def chat_once(self, messages: list[dict], no_tool: bool = False) -> str:
        """One vLLM chat call with the Thyme-style stop tokens.

        Args:
            messages: OpenAI-style chat messages.
            no_tool: if True, add ``<code>`` (the OPEN tag) to the stop list
                so the model cannot even start emitting a tool call. This is
                used by counterfactual rollouts where "skip tool t" means
                "no further tool calls allowed either" — the whole point is
                to isolate the marginal effect of tool t.

        Returns empty string on failure so the agent loop can decide what
        to do."""
        # Import here to avoid a hard dep at module load time (Step 2 users
        # who don't touch the side-car should not pay this import).
        from eval_safety_meme import STOP_TOKENS  # type: ignore

        stop_list = list(STOP_TOKENS)
        if no_tool and "<code>" not in stop_list:
            stop_list.append("<code>")

        last_err: Optional[BaseException] = None
        for attempt in range(self.max_retry):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    max_tokens=self.max_new_tokens,
                    stop=stop_list,
                    extra_body={"skip_special_tokens": False},
                )
                return resp.choices[0].message.content or ""
            except Exception as e:
                last_err = e
                if attempt < self.max_retry - 1:
                    time.sleep(1.5)
                    continue
        if last_err is not None:
            _log(
                "error",
                f"chat call failed after {self.max_retry} retries "
                f"(url={self.url}, model={self.model}): {last_err}",
            )
        return ""


# --------------------------------------------------------------------------- #
# Trajectory slicing                                                           #
# --------------------------------------------------------------------------- #
def slice_trajectory_before_code(
    assistant_text: str, skip_code_idx: int
) -> Optional[str]:
    """Return assistant_text truncated to just BEFORE the t-th ``<code>``
    tag. Returns None if there are not enough code blocks.

    The counterfactual continuation will pick up from the resulting text,
    letting the reference policy decide whether to skip the tool entirely
    or re-attempt something different.
    """
    if skip_code_idx < 0:
        return None
    starts = find_code_starts(assistant_text or "")
    if skip_code_idx >= len(starts):
        return None
    return assistant_text[: starts[skip_code_idx]]


# --------------------------------------------------------------------------- #
# Counterfactual agent loop (one sample)                                       #
# --------------------------------------------------------------------------- #
def _run_cf_agent(
    image_path: str,
    prefix_assistant_text: str,
    qid: str,
    client: SideCarClient,
    prior_image_paths: Optional[list] = None,
) -> dict:
    """Run the Thyme agent loop counterfactually starting from ``prefix``.

    Rules:
      * ``prefix_assistant_text`` is everything the ORIGINAL policy said up
        to just before the t-th ``<code>`` open tag.
      * Prior (0..t-1) tool rounds are REPLAYED as multi-turn messages so
        the side-car sees byte-identical visual evidence: each round's
        ``<code>`` is an assistant turn, each ``<sandbox_output>`` is a
        user turn with the ORIGINAL sandbox image attached from
        ``prior_image_paths[k]``.
      * From that point onward, we let the side-car generate. If it emits
        ANY ``<code>`` block, we do NOT execute the sandbox -- we send
        back ``CF_TOOL_UNAVAILABLE_MSG`` inside ``<sandbox_output>``
        markers, with no image. The model has to answer from the
        evidence already present in context.

    Returns:
        {"cot", "pred", "raw_text", "n_rounds", "status",
         "n_cf_tool_attempts"}
    """
    from eval_safety_meme import (  # type: ignore
        _build_initial_user_text,
        _load_system_prompt,
        _default_system_prompt_path,
        extract_think_cot,
        parse_answer_verdict,
    )

    system_prompt = _load_system_prompt(_default_system_prompt_path())
    user_text = _build_initial_user_text(image_path)

    orig_b64 = _encode_image_path(image_path)
    if orig_b64 is None:
        return {
            "cot": "", "pred": "unknown", "raw_text": "",
            "n_rounds": 0, "status": "image_read_failed",
            "n_cf_tool_attempts": 0,
        }

    # ---- seed messages: system + initial user (original meme) ----------
    messages: list[dict] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{orig_b64}"},
                },
                {"type": "text", "text": user_text},
            ],
        },
    ]

    # ---- replay prefix as multi-turn messages --------------------------
    initial_think, rounds = split_prefix_into_rounds(prefix_assistant_text)
    prior_image_paths = list(prior_image_paths or [])

    trajectory_parts: list[str] = []
    if prefix_assistant_text:
        trajectory_parts.append(prefix_assistant_text)

    if rounds:
        # Round 0: assistant turn = initial_think + code_block
        # Round i>=1: assistant turn = before_code_i + code_block_i
        # Between assistant turns: user turn = <sandbox_output> + image_i + </sandbox_output>
        for r_idx, (before_code, code_block, _sbox_block) in enumerate(rounds):
            if r_idx == 0:
                assistant_turn_text = (initial_think or "") + code_block
            else:
                assistant_turn_text = (before_code or "") + code_block
            messages.append(
                {"role": "assistant", "content": assistant_turn_text}
            )

            user_content: list[dict] = [
                {"type": "text", "text": "<sandbox_output>"}
            ]
            img_p = prior_image_paths[r_idx] if r_idx < len(prior_image_paths) else ""
            attached = False
            if img_p and os.path.isfile(img_p):
                b64 = _encode_image_path(img_p)
                if b64:
                    user_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
                    attached = True
            if not attached:
                user_content.append({
                    "type": "text",
                    "text": "[sandbox image, no longer available]",
                })
                _log(
                    "warn",
                    f"qid={qid}: prior sandbox image #{r_idx} missing "
                    f"({img_p!r}); using textual placeholder",
                )
            user_content.append({"type": "text", "text": "</sandbox_output>"})
            messages.append({"role": "user", "content": user_content})

        # After the LAST replayed round, the ORIGINAL trajectory would
        # have continued with some "before-next-code" think text. That
        # text is NOT in the prefix (it belongs to what we truncated at
        # the t-th <code>), so we just let the side-car generate freshly.
    elif initial_think:
        # No completed rounds -- prefix is just initial_think. Prime it
        # as an already-open assistant turn.
        messages.append({"role": "assistant", "content": initial_think})

    status = "success"
    n_cf_tool_attempts = 0

    for round_idx in range(client.max_rounds):
        try:
            resp_text = client.chat_once(messages)
        except Exception as e:
            status = f"api_error:{e!s}"
            break
        if not resp_text:
            status = "empty_response"
            break

        trajectory_parts.append(resp_text)

        # Terminal: <answer>
        if "<answer>" in resp_text and "</answer>" not in resp_text:
            trajectory_parts[-1] = resp_text + "</answer>"
            break
        if "</answer>" in resp_text:
            break

        # Model tried to call the tool. INTERCEPT: do NOT execute sandbox.
        if resp_text.rstrip().endswith("</code>"):
            resp_text_full = resp_text
        elif "<code>" in resp_text:
            resp_text_full = resp_text + "</code>"
            trajectory_parts[-1] = resp_text_full
        else:
            status = "no_code_no_answer"
            break

        n_cf_tool_attempts += 1

        messages.append({"role": "assistant", "content": resp_text_full})
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "<sandbox_output>"},
                {"type": "text", "text": CF_TOOL_UNAVAILABLE_MSG},
                {"type": "text", "text": "</sandbox_output>"},
            ],
        })
        trajectory_parts.append(
            f"<sandbox_output>{CF_TOOL_UNAVAILABLE_MSG}</sandbox_output>"
        )
    else:
        status = "max_rounds_no_answer"

    raw_text = "".join(trajectory_parts)
    pred = parse_answer_verdict(raw_text)
    if pred == "unknown" and status == "success":
        status = "no_answer"
    cot = extract_think_cot(raw_text)

    return {
        "cot": cot,
        "pred": pred,
        "raw_text": raw_text,
        "n_rounds": len(trajectory_parts),
        "status": status,
        "n_cf_tool_attempts": n_cf_tool_attempts,
    }


# --------------------------------------------------------------------------- #
# Public API                                                                   #
# --------------------------------------------------------------------------- #
def rollout_counterfactual(
    image_path: str,
    original_assistant_text: str,
    skip_code_idx: int,
    qid: str = "unknown",
    prior_image_paths: Optional[list] = None,
) -> dict:
    """Run a single counterfactual rollout: disable the t-th tool call.

    Args:
        image_path: absolute path of the ORIGINAL meme image.
        original_assistant_text: the FULL assistant text (Thyme trajectory).
        skip_code_idx: which tool call (0-indexed) to counterfactually
            disable. Prior calls (0..t-1) are replayed with their original
            sandbox images; the t-th call (and any subsequent tool
            emission) is intercepted and answered with
            ``CF_TOOL_UNAVAILABLE_MSG``.
        qid: identifier for logging.
        prior_image_paths: absolute paths of the sandbox images produced
            by the FIRST t tool calls, in order. Only the first t entries
            are consumed (one per prior round). In training these come
            from ``kwargs["images"][1 : t+1]`` (index 0 is the original
            meme).
    """
    client = SideCarClient.get_singleton()
    if not client.enabled:
        return {
            "cot": "", "pred": "unknown", "raw_text": "",
            "n_rounds": 0, "status": "sidecar_disabled",
            "n_cf_tool_attempts": 0,
        }
    if not client._lazy_build_client() or not client._health_check():
        return {
            "cot": "", "pred": "unknown", "raw_text": "",
            "n_rounds": 0, "status": "sidecar_unhealthy",
            "n_cf_tool_attempts": 0,
        }

    prefix = slice_trajectory_before_code(original_assistant_text, skip_code_idx)
    if prefix is None:
        return {
            "cot": "", "pred": "unknown", "raw_text": "",
            "n_rounds": 0, "status": "no_such_code_idx",
            "n_cf_tool_attempts": 0,
        }

    try:
        return _run_cf_agent(
            image_path=image_path,
            prefix_assistant_text=prefix,
            qid=qid,
            client=client,
            prior_image_paths=prior_image_paths,
        )
    except Exception as e:
        _log("error", f"rollout_counterfactual crashed on qid={qid}: {e}")
        traceback.print_exc()
        return {
            "cot": "", "pred": "unknown", "raw_text": "",
            "n_rounds": 0, "status": f"cf_crash:{e!s}",
            "n_cf_tool_attempts": 0,
        }


def rollout_counterfactuals_for_batch(
    requests: list[dict],
    max_workers: Optional[int] = None,
) -> list[dict]:
    """Batched counterfactual rollout.

    Args:
        requests: each dict has keys:
            image_path (str), assistant_text (str), skip_code_idx (int),
            qid (str, optional),
            prior_image_paths (list[str], optional)
        max_workers: HTTP concurrency (defaults to ``CF_ROLLOUT_WORKERS`` env
            or 8). Note the side-car's continuous batcher already batches
            internally; here we just make sure we keep it busy.

    Returns:
        List of the same length; each entry is the dict returned by
        ``rollout_counterfactual``.
    """
    n = len(requests)
    if n == 0:
        return []
    if max_workers is None:
        max_workers = int(os.environ.get("CF_ROLLOUT_WORKERS", "8") or "8")

    client = SideCarClient.get_singleton()
    if not client.enabled:
        _log_once(
            "skip",
            f"sidecar_disabled:{client._disable_reason}",
            f"rollout_counterfactuals_for_batch called but side-car disabled: "
            f"{client._disable_reason}",
        )
        return [
            {"cot": "", "pred": "unknown", "raw_text": "",
             "n_rounds": 0, "status": "sidecar_disabled",
             "n_cf_tool_attempts": 0}
            for _ in range(n)
        ]

    results: list[dict] = [None] * n  # type: ignore
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        fut2idx = {
            ex.submit(
                rollout_counterfactual,
                image_path=req["image_path"],
                original_assistant_text=req["assistant_text"],
                skip_code_idx=int(req["skip_code_idx"]),
                qid=str(req.get("qid", f"idx_{i}")),
                prior_image_paths=req.get("prior_image_paths", None),
            ): i
            for i, req in enumerate(requests)
        }
        for fut in as_completed(fut2idx):
            i = fut2idx[fut]
            try:
                results[i] = fut.result()
            except Exception as e:
                _log("error", f"worker for req#{i} crashed: {e}")
                results[i] = {
                    "cot": "", "pred": "unknown", "raw_text": "",
                    "n_rounds": 0, "status": f"worker_crash:{e!s}",
                    "n_cf_tool_attempts": 0,
                }
    return results  # type: ignore
