"""Safety-specific GRPO reward plugin for Thyme.

Mirrors the three ORMs Thyme uses in RL (``fmt_orm``, ``vqa_orm``, ``cst_orm``)
but re-implements VQA and CST using deterministic rules, because our task
(harmful / not_harmful meme classification) is a strict binary label. This
removes the dependency on the Qwen2.5-VL-72B judge server that the original
plugin at ``examples/train/grpo/plugin/agent_rm.py`` requires.

Registered plugin names (referenced from the training script):
    safety_fmt_orm   -- structural check on <think>...</think><answer>...</answer>
    safety_vqa_orm   -- correctness by exact-match on the binary label
    safety_cst_orm   -- consistency between the last-500 chars of <think> and the answer

Weights, matching Thyme's defaults, are configurable via env vars:
    FMT_WEIGHT  (default 0.5)
    VQA_WEIGHT  (default 1.0)
    CST_WEIGHT  (default 0.5)

Reward range is [0, weight] per sample (never negative), consistent with
INVALID_REWARD_VALUE = 0 in the original agent_rm.py.
"""

from __future__ import annotations

import asyncio
import os
import re
from typing import List

from swift.plugin import ORM, orms


# ---------------------------------------------------------------------------
# Text extraction utilities (kept identical in behavior to agent_rm.py so that
# any downstream tooling relying on the same regex still works).
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_FORMAT_RE = re.compile(
    r"^(?:(?!</think>).)*</think>\s<answer>(?:(?!</answer>).)*</answer>\Z",
    re.DOTALL,
)


def extract_thinking(solution: str) -> str:
    m = _THINK_RE.search(solution)
    thinking = m.group(1) if m else ""
    if thinking == "":
        # Fallback: everything before </think>.
        thinking = solution.split("</think>")[0].replace("<think>", "")
    thinking = thinking.replace("<image>", "")
    thinking = thinking.replace("</sandbox_output>", "")
    thinking = thinking.replace("<sandbox_output>", "")
    return thinking


def extract_answer(solution: str) -> str:
    m = _ANSWER_RE.search(solution)
    return (m.group(1) if m else "").strip()


def format_check(solution: str) -> bool:
    """Strict Thyme format: exactly one <think>...</think><answer>...</answer>."""
    if _FORMAT_RE.fullmatch(solution) is None:
        return False
    return (
        solution.count("<think>") == 1
        and solution.count("</think>") == 1
        and solution.count("<answer>") == 1
        and solution.count("</answer>") == 1
    )


# ---------------------------------------------------------------------------
# Answer normalization for the binary safety label.
# ---------------------------------------------------------------------------

_HARMFUL_TOKENS = {"harmful", "harm", "yes", "yes_harmful"}
_NOT_HARMFUL_TOKENS = {
    "not_harmful",
    "not harmful",
    "notharmful",
    "non_harmful",
    "non-harmful",
    "safe",
    "benign",
    "no",
    "no_harm",
}


def _normalize_label(raw: str) -> str:
    """Canonicalize a free-form model answer into 'harmful' / 'not_harmful' / ''."""
    if raw is None:
        return ""
    s = raw.strip().lower().strip(".!,\"' ")
    if s == "":
        return ""
    if s in _HARMFUL_TOKENS:
        return "harmful"
    if s in _NOT_HARMFUL_TOKENS:
        return "not_harmful"
    # Handle short phrases containing 'not' and 'harm'.
    if "not" in s and "harm" in s:
        return "not_harmful"
    if s == "harmful" or s.startswith("harmful"):
        return "harmful"
    return ""


# ---------------------------------------------------------------------------
# ORM 1 -- Format reward (rule-based, matches Thyme's FMTORM exactly).
# ---------------------------------------------------------------------------

class FMTORM(ORM):
    # Name deliberately matches Thyme's agent_rm.py so that GRPOTrainer's
    # reward-gating logic in _score_completions (which does
    # ``reward_name_list.index('FMTORM')``) still works.
    def __init__(self) -> None:
        self.weight = float(os.getenv("FMT_WEIGHT", 0.5))
        self.norm = bool(int(os.getenv("FMT_NORM", 0)))
        self.loop = asyncio.get_event_loop()

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:  # noqa: D401
        return self.loop.run_until_complete(self._batch(completions))

    async def _batch(self, completions) -> List[float]:
        return [self._score(c) for c in completions]

    def _score(self, completion: str) -> float:
        if format_check(completion):
            return self.weight
        return -self.weight if self.norm else 0.0


# ---------------------------------------------------------------------------
# ORM 2 -- VQA (correctness) reward. Rule-based binary equality against the
# ground-truth safety label. Follows the same "return in [0, weight] and use
# INVALID_REWARD_VALUE=0 on parse failure" convention as agent_rm.py's VQAORM.
# ---------------------------------------------------------------------------

class VQAORM(ORM):
    # See note on FMTORM: class name mirrors agent_rm.py so that
    # ``reward_name_list.index('VQAORM')`` in GRPOTrainer resolves correctly
    # and the VQA->cst/code gating logic remains identical to the baseline.
    def __init__(self) -> None:
        self.weight = float(os.getenv("VQA_WEIGHT", 1.0))
        # Kept for compatibility with Thyme's env-var contract; unused because
        # our reward is already in [0, 1] which GRPO's group-relative
        # advantage handles well.
        self.norm = bool(int(os.getenv("VQA_NORM", 0)))
        self.loop = asyncio.get_event_loop()

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        return self.loop.run_until_complete(self._batch(completions, solution))

    async def _batch(self, completions, solution) -> List[float]:
        return [self._score(c, g) for c, g in zip(completions, solution)]

    def _score(self, completion: str, gt: str) -> float:
        raw = extract_answer(completion)
        if raw == "":
            return 0.0
        pred = _normalize_label(raw)
        gt_norm = _normalize_label(gt) or (gt or "").strip().lower()
        if pred == "" or gt_norm == "":
            return 0.0
        outcome = 1.0 if pred == gt_norm else 0.0
        if self.norm:
            outcome = (outcome - 0.5) * 2
        return outcome * self.weight


# ---------------------------------------------------------------------------
# ORM 3 -- Consistency reward. Rule-based version of Thyme's CSTORM: checks
# whether the last 500 chars of the <think> block are semantically consistent
# with the extracted final answer. We use keyword presence, since the safety
# vocabulary is very restricted.
# ---------------------------------------------------------------------------

_HARM_KEYWORDS = (
    "harmful",
    "harm",
    "hateful",
    "hate",
    "attack",
    "mockery",
    "harass",
    "threat",
    "distress",
    "dehumaniz",
    "slander",
    "exclusion",
    "misinformation",
    "misogyn",
    "racist",
    "sexist",
    "homophob",
    "transphob",
    "dangerous",
    "offensive",
    "toxic",
)
_NOT_HARM_KEYWORDS = (
    "not harmful",
    "not_harmful",
    "no harm",
    "harmless",
    "benign",
    "safe",
    "innocent",
    "innocuous",
    "not offensive",
    "no offense",
    "wholesome",
    "positive",
    "supportive",
    "affirm",
    "celebrat",
    "empower",
)


def _thinking_supports(thinking_tail: str, answer_norm: str) -> bool:
    """Very cheap textual consistency check on the last 500 chars of <think>."""
    text = thinking_tail.lower()
    if answer_norm == "harmful":
        # Must be talking about harm, and not repeatedly denying it.
        has_harm_kw = any(k in text for k in _HARM_KEYWORDS)
        has_not_kw = any(k in text for k in _NOT_HARM_KEYWORDS)
        # If both are mentioned, prefer the *later* one: whichever occurs
        # closest to the end of the thinking is treated as the conclusion.
        if has_harm_kw and has_not_kw:
            last_harm = max(text.rfind(k) for k in _HARM_KEYWORDS)
            last_not = max(text.rfind(k) for k in _NOT_HARM_KEYWORDS)
            return last_harm > last_not
        return has_harm_kw
    if answer_norm == "not_harmful":
        has_not_kw = any(k in text for k in _NOT_HARM_KEYWORDS)
        has_harm_kw = any(k in text for k in _HARM_KEYWORDS)
        if has_harm_kw and has_not_kw:
            last_harm = max(text.rfind(k) for k in _HARM_KEYWORDS)
            last_not = max(text.rfind(k) for k in _NOT_HARM_KEYWORDS)
            return last_not > last_harm
        return has_not_kw
    return False


class CSTORM(ORM):
    # See note on FMTORM: class name mirrors agent_rm.py.
    def __init__(self) -> None:
        self.weight = float(os.getenv("CST_WEIGHT", 0.5))
        self.norm = bool(int(os.getenv("CST_NORM", 0)))
        self.loop = asyncio.get_event_loop()

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        return self.loop.run_until_complete(self._batch(completions))

    async def _batch(self, completions) -> List[float]:
        return [self._score(c) for c in completions]

    def _score(self, completion: str) -> float:
        raw = extract_answer(completion)
        if raw == "":
            return 0.0
        pred = _normalize_label(raw)
        if pred == "":
            return 0.0
        thinking = extract_thinking(completion)
        # Same 500-char tail slice as Thyme's CSTORM.
        tail = thinking[-500:] if len(thinking) > 500 else thinking
        consistent = _thinking_supports(tail, pred)
        outcome = 1.0 if consistent else 0.0
        if self.norm:
            outcome = (outcome - 0.5) * 2
        return outcome * self.weight


# ---------------------------------------------------------------------------
# ORM 4 -- Faith reward (SafeFaithScore). Calls two remote vLLM judge
# servers via HTTP (Qwen3-32B for text stages; Qwen3-VL-32B for VEM),
# reusing viscot-harmeme's FaithScore + CAF pipeline verbatim.
#
# All failure modes fall back to F=0 for the affected samples (and log
# a message with the prefix "[cf-faith]"). See cf_faith/faith_judge.py.
#
# Configuration is via env vars (see cf_faith/serve_judge_notes.md):
#     FS_TEXT_URL / FS_TEXT_MODEL      -- Qwen3-32B server
#     FS_VEM_URL  / FS_VEM_MODEL       -- Qwen3-VL-32B server
#     FAITH_ALPHA                      -- F = alpha*FaithScore + (1-alpha)*CAF
#     FAITH_WEIGHT                     -- multiplier on F (default 0.5)
#     FS_API_WORKERS / FS_TIMEOUT / FS_MAX_RETRY / FS_TAXONOMY
# ---------------------------------------------------------------------------

class FaithORM(ORM):
    """SafeFaithScore-based reward. Zero-cost when disabled (either
    ``FS_TEXT_URL`` / ``FS_VEM_URL`` unset, or the judge servers are down):
    returns 0.0 for every completion and prints a single ``[cf-faith][skip]``
    warning at first call."""

    def __init__(self) -> None:
        self.weight = float(os.getenv("FAITH_WEIGHT", 0.5))
        # Lazy-construct on first __call__ so that if faith_judge (or
        # viscot-harmeme) has an import error, other ORMs still work.
        self._judge = None
        self._judge_init_failed = False

    # ------------------------------------------------------------------ #
    def _get_judge(self):
        """Lazy singleton. Returns None if judge could not be built."""
        if self._judge is not None or self._judge_init_failed:
            return self._judge
        try:
            # Ensure cf_faith package is importable regardless of CWD.
            import os as _os
            import sys as _sys
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from cf_faith.faith_judge import SafeFaithScoreJudge  # noqa: E402

            self._judge = SafeFaithScoreJudge.get_singleton()
        except Exception as e:
            self._judge_init_failed = True
            print(
                f"[cf-faith][error] FaithORM failed to build SafeFaithScoreJudge: "
                f"{e}. FaithORM will return 0 for all samples.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
        return self._judge

    # ------------------------------------------------------------------ #
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []

        judge = self._get_judge()
        if judge is None:
            return [0.0] * n

        # Extract per-sample image_path (Swift dataset column ``images`` is
        # a list-of-lists: outer index = sample, inner = list of paths).
        images_col = kwargs.get("images", None) or [[] for _ in range(n)]
        # Robust to shorter columns (should never happen, but be safe).
        if len(images_col) < n:
            images_col = list(images_col) + [[] for _ in range(n - len(images_col))]

        # target_label / subset live under kwargs["meta"] if the dataset row
        # carried them (see build_rl_dataset.py which stashes subset there).
        meta_col = kwargs.get("meta", None) or [{} for _ in range(n)]
        if len(meta_col) < n:
            meta_col = list(meta_col) + [{} for _ in range(n - len(meta_col))]

        # ---- Build per-sample items for the judge ------------------------
        items: List[dict] = []
        for i, comp in enumerate(completions):
            gt = ""
            if solution is not None and i < len(solution):
                gt = (solution[i] or "").strip().lower()

            # Extract <think> content and normalized answer from the model
            # completion. FaithScore expects a "cot" string that describes
            # what the model claims to see, not the raw answer.
            try:
                cot_raw = extract_thinking(comp) or ""
            except Exception:
                cot_raw = ""
            try:
                ans_raw = extract_answer(comp) or ""
            except Exception:
                ans_raw = ""
            pred_norm = _normalize_label(ans_raw)

            # image path: pick the first entry of images_col[i] if any.
            image_path = ""
            imgs_i = images_col[i]
            if isinstance(imgs_i, (list, tuple)) and imgs_i:
                first = imgs_i[0]
                # Swift may pass either a path string, a PIL image, or a
                # {"path": ...} dict — accept all three.
                if isinstance(first, str):
                    image_path = first
                elif isinstance(first, dict):
                    image_path = first.get("path", "") or ""
                else:
                    image_path = getattr(first, "path", "") or ""
            elif isinstance(imgs_i, str):
                image_path = imgs_i

            meta_i = meta_col[i] if isinstance(meta_col[i], dict) else {}
            qid = meta_i.get("question_id", f"batch_idx_{i}")

            items.append({
                "image_path": image_path,
                "cot": cot_raw,
                "pred": pred_norm,
                "label": _normalize_label(gt) or gt,
                "target_label": meta_i.get("target_label", ""),
                "question_id": qid,
            })

        # ---- Score. Any exception -> full zeros. -------------------------
        try:
            results = judge.score_batch(items)
        except Exception as e:
            print(
                f"[cf-faith][error] FaithORM.score_batch crashed on batch of "
                f"{n}: {e}. Falling back to F=0 for this batch.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
            return [0.0] * n

        # ---- Apply weight and return -------------------------------------
        rewards: List[float] = []
        n_ok = 0
        for r in results:
            F = float(r.get("F", 0.0)) if isinstance(r, dict) else 0.0
            if isinstance(r, dict) and r.get("status") == "ok":
                n_ok += 1
            rewards.append(F * self.weight)

        # Batch-level sanity log (throttled by _log_once inside judge)
        if n_ok < n:
            print(
                f"[cf-faith][score] batch: {n_ok}/{n} scored OK, "
                f"rest fell back to 0. weight={self.weight}",
                flush=True,
            )
        return rewards


# ---------------------------------------------------------------------------
# ORM 5 -- Tool-level faithfulness reward (CF-FaithGRPO).
#
# For each rollout, for each <code> tool call t in the trajectory:
#    1) run a COUNTERFACTUAL rollout that skips call t, using a FIXED
#       side-car vLLM (Qwen2.5-VL-7B loaded with the initial SFT ckpt,
#       served on a separate machine, HTTP-only from training's POV).
#    2) score both the original CoT and the counterfactual CoT with the
#       same SafeFaithScore judge -> get F(τ) and F(τ_{-t}).
#    3) Δ_t = F(τ) - F(τ_{-t}); bucketize into +1 / 0 / -1 with threshold
#       TOOL_DELTA_EPSILON so noise near zero is ignored.
#    4) reward_i = TOOL_WEIGHT * Σ_t bucketize(Δ_t)  ∈ [-N*TOOL_WEIGHT, +N*TOOL_WEIGHT]
#
# Because the sign-bucketed sum can only reward tools that measurably
# increase F and punish tools that measurably decrease F, this is a clean
# anti-hacking objective: spurious extra <code> calls contribute 0.
#
# Envs (all optional; unset => reward=0 for every sample, training unaffected):
#     CF_SIDECAR_URL / CF_SIDECAR_MODEL       -- side-car endpoint
#     TOOL_WEIGHT           default 0.3       -- lambda_t
#     TOOL_DELTA_EPSILON    default 0.05      -- +/-eps -> 0 bucket
#     CF_ROLLOUT_WORKERS    default 8         -- HTTP concurrency for CFs
#     CF_MAX_TOOLS_PER_SAMPLE default 3       -- cap tools per sample
#     TOOL_MAX_SAMPLES_PER_BATCH default -1   -- optional cap for coarse rate limiting
# ---------------------------------------------------------------------------

def _bucketize(delta: float, eps: float) -> int:
    if delta > eps:
        return 1
    if delta < -eps:
        return -1
    return 0


class FaithToolORM(ORM):
    """Tool-level faithfulness reward via counterfactual rollout.

    Zero-cost when disabled (either side-car not configured, or judge not
    configured). All failures fall back to per-sample reward = 0 with an
    explicit log line so the batch keeps going.
    """

    def __init__(self) -> None:
        self.weight = float(os.getenv("TOOL_WEIGHT", 0.3))
        self.eps = float(os.getenv("TOOL_DELTA_EPSILON", 0.05))
        self.max_tools_per_sample = int(
            os.getenv("CF_MAX_TOOLS_PER_SAMPLE", "3")
        )
        # Optional throttle for exploratory runs: if positive, only the
        # first K samples in the batch actually do counterfactuals; the
        # rest get reward = 0 (does NOT hurt training since GRPO is
        # group-relative and other rewards still fire on all samples).
        self.max_samples_per_batch = int(
            os.getenv("TOOL_MAX_SAMPLES_PER_BATCH", "-1")
        )
        self._judge = None
        self._judge_init_failed = False
        self._sidecar_probed = False

    # ------------------------------------------------------------------ #
    def _get_judge(self):
        if self._judge is not None or self._judge_init_failed:
            return self._judge
        try:
            import os as _os
            import sys as _sys
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from cf_faith.faith_judge import SafeFaithScoreJudge  # noqa: E402

            self._judge = SafeFaithScoreJudge.get_singleton()
        except Exception as e:
            self._judge_init_failed = True
            print(
                f"[cf-faith][error] FaithToolORM failed to build judge: {e}. "
                f"Returning 0 for all samples.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
        return self._judge

    # ------------------------------------------------------------------ #
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []

        judge = self._get_judge()
        if judge is None or not getattr(judge, "enabled", False):
            # Judge disabled -> can't compute Δ_t. Silent 0.
            return [0.0] * n

        # Lazy import so this module doesn't need cf_rollout at import time.
        try:
            import os as _os
            import sys as _sys
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from cf_faith.cf_rollout import (
                SideCarClient, count_code_calls,
                rollout_counterfactuals_for_batch,
            )
        except Exception as e:
            if not self._sidecar_probed:
                print(
                    f"[cf-faith][error] FaithToolORM cannot import cf_rollout: "
                    f"{e}. Returning 0 for all samples.",
                    flush=True,
                )
                self._sidecar_probed = True
            return [0.0] * n

        # Probe side-car availability (once per process).
        sidecar = SideCarClient.get_singleton()
        if not sidecar.enabled:
            if not self._sidecar_probed:
                print(
                    f"[cf-faith][skip] FaithToolORM: side-car disabled "
                    f"({sidecar._disable_reason}). Returning 0 for all samples.",
                    flush=True,
                )
                self._sidecar_probed = True
            return [0.0] * n

        # ---- Pull batch context -----------------------------------------
        images_col = kwargs.get("images", None) or [[] for _ in range(n)]
        if len(images_col) < n:
            images_col = list(images_col) + [[] for _ in range(n - len(images_col))]
        meta_col = kwargs.get("meta", None) or [{} for _ in range(n)]
        if len(meta_col) < n:
            meta_col = list(meta_col) + [{} for _ in range(n - len(meta_col))]

        def _pick_image(imgs_i) -> str:
            if isinstance(imgs_i, (list, tuple)) and imgs_i:
                first = imgs_i[0]
                if isinstance(first, str):
                    return first
                if isinstance(first, dict):
                    return first.get("path", "") or ""
                return getattr(first, "path", "") or ""
            if isinstance(imgs_i, str):
                return imgs_i
            return ""

        def _extract_all_paths(imgs_i) -> List[str]:
            """Return the ORDERED list of absolute image paths for sample i.
            Index 0 is the ORIGINAL meme; index k (k>=1) is the k-th
            sandbox-produced image.

            Handles the three formats Swift may pass:
              * list[str]     -- rare, treat each str as a path
              * list[dict]    -- Thyme's native format: {"bytes":..., "path":...}
              * list[PIL]     -- unusual; fall back to getattr(., "path", "")
            Missing / non-file entries are filtered out silently.
            """
            out: List[str] = []
            if not isinstance(imgs_i, (list, tuple)):
                return out
            for item in imgs_i:
                p = ""
                if isinstance(item, str):
                    p = item
                elif isinstance(item, dict):
                    p = item.get("path", "") or ""
                else:
                    p = getattr(item, "path", "") or ""
                if p:
                    out.append(p)
            return out

        # ---- Enumerate counterfactual (sample_i, tool_t) pairs ----------
        cf_reqs: List[dict] = []           # requests for rollout_counterfactuals_for_batch
        cf_map: List[tuple[int, int]] = [] # (sample_i, tool_t) parallel to cf_reqs
        # Per-sample bookkeeping for later aggregation:
        n_tools_per_sample: List[int] = [0] * n
        image_paths: List[str] = [""] * n
        all_image_paths: List[List[str]] = [[] for _ in range(n)]
        original_cots: List[str] = [""] * n
        original_preds: List[str] = [""] * n
        original_labels: List[str] = [""] * n
        target_labels: List[str] = [""] * n
        qids: List[str] = [""] * n

        # Optional throttle: only the first K samples get CF work.
        sample_budget = n if self.max_samples_per_batch < 0 else min(
            n, self.max_samples_per_batch
        )

        for i, comp in enumerate(completions):
            try:
                cot_i = extract_thinking(comp) or ""
                ans_i = extract_answer(comp) or ""
            except Exception:
                cot_i, ans_i = "", ""
            pred_i = _normalize_label(ans_i)
            gt_i = ""
            if solution is not None and i < len(solution):
                gt_i = (solution[i] or "").strip().lower()
            label_i = _normalize_label(gt_i) or gt_i
            image_paths[i] = _pick_image(images_col[i])
            all_image_paths[i] = _extract_all_paths(images_col[i])
            original_cots[i] = cot_i
            original_preds[i] = pred_i
            original_labels[i] = label_i
            meta_i = meta_col[i] if isinstance(meta_col[i], dict) else {}
            target_labels[i] = meta_i.get("target_label", "") or ""
            qids[i] = meta_i.get("question_id", f"batch_idx_{i}")

            if i >= sample_budget:
                continue
            if not image_paths[i] or not os.path.isfile(image_paths[i]):
                continue
            if pred_i not in ("harmful", "not_harmful"):
                continue
            if label_i not in ("harmful", "not_harmful"):
                continue

            # We slice the ASSISTANT text (the completion), not the cot,
            # because we need to preserve <code>...</code> boundaries.
            n_tools = count_code_calls(comp or "")
            if n_tools <= 0:
                continue
            n_tools = min(n_tools, self.max_tools_per_sample)
            n_tools_per_sample[i] = n_tools
            # all_image_paths[i]:  [orig, sbox_1, sbox_2, ..., sbox_N]
            # For "disable the t-th tool" (t in [0..N-1]):
            #   prior_image_paths = the t images produced by tool calls 0..t-1
            #                     = all_image_paths[i][1 : t+1]
            #   (t=0 -> empty list, correct)
            sbox_imgs = all_image_paths[i][1:]  # drop original meme
            for t in range(n_tools):
                cf_reqs.append({
                    "image_path": image_paths[i],
                    "assistant_text": comp,
                    "skip_code_idx": t,
                    "qid": f"{qids[i]}_cf{t}",
                    "prior_image_paths": sbox_imgs[:t],  # 0..t-1
                })
                cf_map.append((i, t))

        if not cf_reqs:
            # No sample has any tool call, or side-car / judge unusable.
            # Reward is 0 uniformly.
            return [0.0] * n

        # ---- Score ORIGINAL F(τ) for every sample that has tools --------
        # We reuse the FaithScore judge cache: if FaithORM already ran in
        # this batch, F(τ) will be cached. Either way, we call judge once
        # for the union {original τ} ∪ {counterfactual τ_{-t}} so that F
        # is computed by the *same* judge instance in one shot.
        need_orig_idx = [
            i for i in range(sample_budget) if n_tools_per_sample[i] > 0
        ]
        orig_items = [
            {
                "image_path": image_paths[i],
                # cot for FaithScore is a natural-language description; we
                # strip <sandbox_output> / <code> markers -- match what
                # FaithORM feeds in (already done by extract_thinking).
                "cot": original_cots[i],
                "pred": original_preds[i],
                "label": original_labels[i],
                "target_label": target_labels[i],
                "question_id": qids[i],
            }
            for i in need_orig_idx
        ]

        # ---- Run counterfactual rollouts (batched) ----------------------
        print(
            f"[cf-faith][score] FaithToolORM: {len(cf_reqs)} counterfactual "
            f"rollouts across {len(need_orig_idx)}/{n} samples (weight="
            f"{self.weight}, eps={self.eps})",
            flush=True,
        )
        try:
            cf_results = rollout_counterfactuals_for_batch(cf_reqs)
        except Exception as e:
            print(
                f"[cf-faith][error] FaithToolORM: counterfactual rollout crashed: "
                f"{e}. Returning 0 for all samples in this batch.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
            return [0.0] * n

        # ---- Score F(τ) and F(τ_{-t}) together --------------------------
        # Package as one big batch so stage1/2/3/4 batchers stay saturated.
        # Layout:  joint_items = [orig for i in need_orig_idx]
        #                      ++ [cf   for k in range(len(cf_reqs))]
        joint_items: List[dict] = list(orig_items)
        cf_items_start = len(joint_items)
        for k, (sample_i, tool_t) in enumerate(cf_map):
            cfres = cf_results[k] if k < len(cf_results) else {}
            joint_items.append({
                "image_path": image_paths[sample_i],
                "cot": cfres.get("cot", "") or "",
                "pred": cfres.get("pred", "unknown"),
                "label": original_labels[sample_i],
                "target_label": target_labels[sample_i],
                "question_id": f"{qids[sample_i]}_cf{tool_t}",
            })

        try:
            joint_results = judge.score_batch(joint_items)
        except Exception as e:
            print(
                f"[cf-faith][error] FaithToolORM: joint F scoring crashed: "
                f"{e}. Returning 0 for all samples in this batch.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
            return [0.0] * n

        # Slice out F(τ) and F(τ_{-t})
        F_orig_by_sample: dict[int, float] = {}
        for pos, i in enumerate(need_orig_idx):
            r = joint_results[pos] if pos < len(joint_results) else {}
            F_orig_by_sample[i] = float(r.get("F", 0.0)) if isinstance(r, dict) else 0.0

        F_cf: List[float] = []
        for k in range(len(cf_reqs)):
            pos = cf_items_start + k
            r = joint_results[pos] if pos < len(joint_results) else {}
            F_cf.append(float(r.get("F", 0.0)) if isinstance(r, dict) else 0.0)

        # ---- Aggregate: Σ_t bucketize(F(τ) - F(τ_{-t})) per sample ------
        deltas_sum: List[float] = [0.0] * n
        n_pos = 0
        n_zero = 0
        n_neg = 0
        for k, (sample_i, tool_t) in enumerate(cf_map):
            F_orig = F_orig_by_sample.get(sample_i, 0.0)
            cf_status = cf_results[k].get("status") if isinstance(cf_results[k], dict) else ""
            if cf_status != "success":
                # A crashed / degraded CF is treated as "no signal" -> 0.
                # Do NOT count this as evidence that the tool was harmful.
                continue
            delta = F_orig - F_cf[k]
            bucket = _bucketize(delta, self.eps)
            deltas_sum[sample_i] += bucket
            if bucket > 0:
                n_pos += 1
            elif bucket < 0:
                n_neg += 1
            else:
                n_zero += 1

        rewards = [self.weight * d for d in deltas_sum]

        print(
            f"[cf-faith][score] FaithToolORM: tool buckets "
            f"+1={n_pos} 0={n_zero} -1={n_neg}  "
            f"reward range=[{min(rewards):.3f}, {max(rewards):.3f}]",
            flush=True,
        )
        return rewards


# ---------------------------------------------------------------------------
# ORM 6 -- TFAToolORM: Teacher-Forced visual Ablation reward.
#
# Reads a per-sample list of D_t values (nats/token) that the GRPO trainer
# computed in ``_generate_and_score_completions`` via TWO teacher-forced
# forwards (orig image vs. white image, same shape). See
# ``教师强制视觉消融奖励_实现文档.md`` for the design rationale and
# ``cf_faith/tfa_core.py`` for the shared primitives.
#
# This ORM is INTENTIONALLY minimal: it only turns the raw D_t values into
# bucketized {-1, 0, +1} and multiplies by TFA_WEIGHT. The full quality
# gate  (correct AND F(τ) ≥ F_high  ->  +1; wrong OR F(τ) ≤ F_low with
# high D_t  ->  -1) is applied AFTER this ORM runs, inside
# ``grpo_trainer._score_completions``, so it can see the freshly computed
# VQAORM and FaithORM rewards. That way the gating logic lives next to
# the existing VQA-based gating for CODEORM/CSTORM (which does the same
# trick) and stays in ONE place.
#
# Trainer contract (see grpo_trainer._score_completions):
#   - ``reward_kwargs['tfa_D_per_sample']`` : List[Optional[List[float]]]
#         Length == batch size (# rollouts). Each entry is either None
#         (TFA disabled / crop missing / OOM / etc.) or a list of raw
#         D_t values (one per tool call up to TFA_MAX_TOOLS_PER_SAMPLE).
#   - When ``tfa_D_per_sample`` is missing entirely, this ORM returns 0
#     for every sample (behaviourally equivalent to disabled).
#
# Envs:
#     TFA_ENABLE          default 0   -- when 0, ORM short-circuits to 0
#     TFA_WEIGHT          default 0.3 -- lambda_t; same role as TOOL_WEIGHT
#     TFA_EPSILON         default 0.02-- |D_t| <= eps -> bucket 0
# ---------------------------------------------------------------------------

class TFAToolORM(ORM):
    def __init__(self) -> None:
        self.enabled = int(os.getenv("TFA_ENABLE", "0")) == 1
        # TFA has its OWN weight so a run can compare TOOL_MODE=cf vs
        # TOOL_MODE=tfa without both sharing TOOL_WEIGHT accidentally.
        # Default 0.3 mirrors TOOL_WEIGHT.
        self.weight = float(os.getenv("TFA_WEIGHT", 0.5))
        # eps in nats/token. See 1b offline scan: |D_t| p10/p90 ≈ ±0.04,
        # so eps=0.02 makes the middle ~50% of samples fall in bucket 0.
        # 20260804: lowered default 0.02 -> 0.01 because online D_t is
        # tighter than 1b (Step-2 seeded CoT is more tool-agnostic), so
        # too many samples end up in the dead zone at 0.02.
        self.eps = float(os.getenv("TFA_EPSILON", 0.01))
        self._skip_logged = False

    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []
        if not self.enabled:
            if not self._skip_logged:
                print(
                    "[tfa][skip] TFAToolORM: TFA_ENABLE!=1, returning 0 "
                    "for every sample (equivalent to disabled).",
                    flush=True,
                )
                self._skip_logged = True
            return [0.0] * n

        # Trainer stashes per-sample D_t lists here.
        D_lists = kwargs.get("tfa_D_per_sample", None)
        if D_lists is None:
            if not self._skip_logged:
                print(
                    "[tfa][skip] TFAToolORM: TFA_ENABLE=1 but "
                    "'tfa_D_per_sample' not in reward_kwargs. Trainer TFA "
                    "path likely disabled; returning 0.",
                    flush=True,
                )
                self._skip_logged = True
            return [0.0] * n

        # Pad / truncate to length n just in case.
        if len(D_lists) < n:
            D_lists = list(D_lists) + [None] * (n - len(D_lists))
        elif len(D_lists) > n:
            D_lists = list(D_lists)[:n]

        rewards: List[float] = []
        n_pos = 0
        n_zero = 0
        n_neg = 0
        n_none = 0
        for D_i in D_lists:
            if D_i is None or len(D_i) == 0:
                rewards.append(0.0)
                n_none += 1
                continue
            bucket_sum = 0
            for D_t in D_i:
                if D_t is None or D_t != D_t:  # NaN
                    continue
                if D_t > self.eps:
                    bucket_sum += 1
                    n_pos += 1
                elif D_t < -self.eps:
                    # 20260804: negative bucket retired -- see bucket_D_t
                    # docstring in cf_faith/tfa_core.py. We KEEP the count
                    # (n_neg) for observability but do NOT subtract from
                    # bucket_sum: TFA is now a purely positive signal.
                    n_neg += 1
                else:
                    n_zero += 1
            rewards.append(float(bucket_sum) * self.weight)

        print(
            f"[tfa][score] TFAToolORM: buckets +1={n_pos} 0={n_zero} "
            f"-1={n_neg}  n_no_D={n_none}/{n}  "
            f"weight={self.weight}  eps={self.eps}  "
            f"reward range=[{min(rewards) if rewards else 0:.3f}, "
            f"{max(rewards) if rewards else 0:.3f}]",
            flush=True,
        )
        return rewards

# ---------------------------------------------------------------------------
# ORM 7 -- Offline counterfactual faithfulness reward (Step 5).
#
# Semantics
# ---------
# For every rollout τ with question_id qid, look up a pre-computed reference
# faithfulness score F_ref(qid) that was produced OFFLINE by the raw
# Qwen2.5-VL-7B-Instruct model running the viscot-harmeme *direct* prompt
# (no <think>/<answer> tags, no tool). Then:
#
#     Δ         = FaithScore_atomic(τ)  -  F_ref(qid)
#     bucket    = +1  if Δ >  OFFLINE_CF_EPS
#                 -1  if Δ < -OFFLINE_CF_EPS
#                  0  otherwise
#     reward_i  = OFFLINE_CF_WEIGHT * bucket
#
# The bucketed sign reward answers "did the current policy's tool-augmented
# reasoning improve atomic faithfulness over a no-tool reference?" and is
# invariant to any batch-level CAF fluctuation (Δ only involves FaithScore,
# not CAF -- see chat decision A).
#
# Compared to Step-3 (side-car CF rollout) and Step-4 (TFA teacher-forced
# ablation):
#   * NO side-car GPU host required.
#   * NO extra forward at training time (F_ref is a JSON dict lookup).
#   * F(τ) reuse is FREE: SafeFaithScoreJudge already caches its per-sample
#     scores by sha1(image_path|cot|pred|label), so this ORM's judge call
#     collapses to a pure cache hit when FaithORM runs in the same batch.
#
# Envs (all optional; unset => reward = 0 for every sample, silent):
#     F_REF_PATH              path to build_f_ref_table.py output json
#     OFFLINE_CF_WEIGHT       default 0.5  (lambda)
#     OFFLINE_CF_EPS          default 0.05 (bucket threshold)
#     OFFLINE_CF_LOG_MISS     default 100  (log at most K unique missing qids)
# ---------------------------------------------------------------------------


class FaithOfflineCFORM(ORM):
    """Offline-counterfactual faithfulness reward.

    * Loads a ``{qid -> {"F_ref": float, ...}}`` table from ``F_REF_PATH``
      at first call (lazy) and caches it on the instance.
    * At every batch, computes each rollout's atomic FaithScore by reusing
      ``SafeFaithScoreJudge.score_batch`` (which caches F(τ) within the
      optimizer step, so cost = judge cache hit).
    * Bucketizes ``Δ = FaithScore_atomic(τ) - F_ref[qid]`` with symmetric
      threshold ``OFFLINE_CF_EPS`` and returns ``weight * bucket``.
    * Fully zero-cost when ``F_REF_PATH`` is unset or the FaithScore judge
      is unavailable: returns 0 for every sample and prints one warning.
    """

    def __init__(self) -> None:
        self.weight = float(os.getenv("OFFLINE_CF_WEIGHT", "0.5"))
        self.eps = float(os.getenv("OFFLINE_CF_EPS", "0.05"))
        self.log_miss_cap = int(os.getenv("OFFLINE_CF_LOG_MISS", "100"))
        self._table: dict[str, dict] | None = None
        self._table_load_failed = False
        self._judge = None
        self._judge_init_failed = False
        self._logged_missing: set[str] = set()
        self._missing_total = 0

    # ------------------------------------------------------------------ #
    def _get_table(self):
        """Lazy-load the F_ref json. Returns None if the file is not set /
        not readable / has no ``table`` field.

        Retry semantics (important for the common "training started before
        the offline F_ref table finished generating" case):
          * F_REF_PATH unset  -> permanent skip (never retry; silent 0).
          * file does NOT exist yet -> RETRY every call (rate-limited warn),
            do NOT set _table_load_failed. The table auto-engages as soon as
            the file appears, so a running job picks it up without restart.
          * file exists but JSON broken / no 'table' -> permanent failure.
        """
        if self._table is not None or self._table_load_failed:
            return self._table
        path = os.getenv("F_REF_PATH", "").strip()
        if not path:
            self._table_load_failed = True
            print(
                "[cf-faith][skip] FaithOfflineCFORM: F_REF_PATH not set; "
                "returning reward=0 for every sample.",
                flush=True,
            )
            return None
        if not os.path.isfile(path):
            # Not ready yet -> retry on next batch instead of giving up.
            # Rate-limit the warning so we don't flood the log every step.
            self._miss_warn_total = getattr(self, "_miss_warn_total", 0) + 1
            if self._miss_warn_total <= 5 or self._miss_warn_total % 200 == 0:
                print(
                    f"[cf-faith][warn] FaithOfflineCFORM: F_REF_PATH={path} "
                    f"not found yet (retry later). reward=0 this batch. "
                    f"(warn #{self._miss_warn_total})",
                    flush=True,
                )
            return None
        try:
            import json as _json
            with open(path, "r", encoding="utf-8") as f:
                payload = _json.load(f)
            table = payload.get("table") if isinstance(payload, dict) else None
            if not isinstance(table, dict) or not table:
                self._table_load_failed = True
                print(
                    f"[cf-faith][error] FaithOfflineCFORM: F_REF_PATH={path} "
                    f"missing / empty 'table' field. Returning reward=0.",
                    flush=True,
                )
                return None
            self._table = table
            meta = payload.get("_meta", {}) if isinstance(payload, dict) else {}
            print(
                f"[cf-faith][init] FaithOfflineCFORM: loaded {len(table)} "
                f"F_ref rows from {path}  (per_subset={meta.get('per_subset', {})}, "
                f"weight={self.weight}, eps={self.eps})",
                flush=True,
            )
            return self._table
        except Exception as e:
            self._table_load_failed = True
            print(
                f"[cf-faith][error] FaithOfflineCFORM: failed to load "
                f"F_REF_PATH={path}: {e}. Returning reward=0.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
            return None

    def _get_judge(self):
        if self._judge is not None or self._judge_init_failed:
            return self._judge
        try:
            import os as _os
            import sys as _sys
            _here = _os.path.dirname(_os.path.abspath(__file__))
            if _here not in _sys.path:
                _sys.path.insert(0, _here)
            from cf_faith.faith_judge import SafeFaithScoreJudge  # noqa: E402

            self._judge = SafeFaithScoreJudge.get_singleton()
        except Exception as e:
            self._judge_init_failed = True
            print(
                f"[cf-faith][error] FaithOfflineCFORM failed to build "
                f"SafeFaithScoreJudge: {e}. Returning reward=0 for every sample.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
        return self._judge

    # ------------------------------------------------------------------ #
    def __call__(self, completions, solution=None, **kwargs) -> List[float]:
        n = len(completions)
        if n == 0:
            return []

        table = self._get_table()
        if table is None:
            return [0.0] * n
        judge = self._get_judge()
        if judge is None or not getattr(judge, "enabled", False):
            # Judge disabled -> can't score F(τ). Silent 0.
            return [0.0] * n

        # ---- Pull per-sample context (mirrors FaithORM exactly) ----------
        images_col = kwargs.get("images", None) or [[] for _ in range(n)]
        if len(images_col) < n:
            images_col = list(images_col) + [[] for _ in range(n - len(images_col))]
        meta_col = kwargs.get("meta", None) or [{} for _ in range(n)]
        if len(meta_col) < n:
            meta_col = list(meta_col) + [{} for _ in range(n - len(meta_col))]

        items: List[dict] = []
        qids: List[str] = [""] * n
        for i, comp in enumerate(completions):
            gt = ""
            if solution is not None and i < len(solution):
                gt = (solution[i] or "").strip().lower()
            try:
                cot_raw = extract_thinking(comp) or ""
            except Exception:
                cot_raw = ""
            try:
                ans_raw = extract_answer(comp) or ""
            except Exception:
                ans_raw = ""
            pred_norm = _normalize_label(ans_raw)

            image_path = ""
            imgs_i = images_col[i]
            if isinstance(imgs_i, (list, tuple)) and imgs_i:
                first = imgs_i[0]
                if isinstance(first, str):
                    image_path = first
                elif isinstance(first, dict):
                    image_path = first.get("path", "") or ""
                else:
                    image_path = getattr(first, "path", "") or ""
            elif isinstance(imgs_i, str):
                image_path = imgs_i

            meta_i = meta_col[i] if isinstance(meta_col[i], dict) else {}
            qid = meta_i.get("question_id", f"batch_idx_{i}")
            qids[i] = qid
            items.append({
                "image_path": image_path,
                "cot": cot_raw,
                "pred": pred_norm,
                "label": _normalize_label(gt) or gt,
                "target_label": meta_i.get("target_label", ""),
                "question_id": qid,
            })

        # ---- Score. Any exception -> full zeros. -------------------------
        # SafeFaithScoreJudge caches per-sample scores by sha1(image_path|
        # cot|pred|label), so if FaithORM has already scored this batch,
        # every call here is a cache hit.
        try:
            results = judge.score_batch(items)
        except Exception as e:
            print(
                f"[cf-faith][error] FaithOfflineCFORM.score_batch crashed "
                f"on batch of {n}: {e}. Falling back to reward=0 for this batch.",
                flush=True,
            )
            import traceback as _tb
            _tb.print_exc()
            return [0.0] * n

        # ---- Compute Δ + bucket ------------------------------------------
        rewards: List[float] = []
        n_pos = n_zero = n_neg = n_miss = 0
        for i, r in enumerate(results):
            qid = qids[i]
            entry = table.get(qid) if qid else None
            if not entry:
                n_miss += 1
                # Rate-limited logging of unique missing qids so we don't
                # flood the training log while still surfacing dataset
                # mismatches early.
                if qid and qid not in self._logged_missing \
                        and len(self._logged_missing) < self.log_miss_cap:
                    self._logged_missing.add(qid)
                    print(
                        f"[cf-faith][miss] FaithOfflineCFORM: qid={qid} not "
                        f"in F_ref table (reward=0). This is safe fallback.",
                        flush=True,
                    )
                rewards.append(0.0)
                continue
            try:
                f_ref = float(entry.get("F_ref", 0.0))
                f_ref = max(0.0, min(1.0, f_ref))
            except (TypeError, ValueError):
                rewards.append(0.0)
                continue
            fs_tau = 0.0
            if isinstance(r, dict) and r.get("status") == "ok":
                fs_tau = float(r.get("faithscore_atomic", 0.0))
                fs_tau = max(0.0, min(1.0, fs_tau))
            else:
                # Judge failed on this specific sample -> can't compute Δ.
                rewards.append(0.0)
                continue

            delta = fs_tau - f_ref
            if delta > self.eps:
                bucket = 1
                n_pos += 1
            elif delta < -self.eps:
                bucket = -1
                n_neg += 1
            else:
                bucket = 0
                n_zero += 1
            rewards.append(self.weight * bucket)

        self._missing_total += n_miss
        print(
            f"[cf-faith][score] FaithOfflineCFORM: batch={n}  +={n_pos} "
            f"0={n_zero} -={n_neg}  miss={n_miss}  weight={self.weight} "
            f"eps={self.eps}  total_miss_so_far={self._missing_total}",
            flush=True,
        )
        return rewards

# ---------------------------------------------------------------------------
# Register with ms-swift's plugin registry.
# ---------------------------------------------------------------------------

orms["safety_fmt_orm"] = FMTORM
orms["safety_vqa_orm"] = VQAORM
orms["safety_cst_orm"] = CSTORM
orms["safety_faith_orm"] = FaithORM
orms["safety_faith_tool_orm"] = FaithToolORM
orms["safety_tfa_tool_orm"] = TFAToolORM
orms["safety_faith_offline_cf_orm"] = FaithOfflineCFORM
