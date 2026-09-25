"""Training-side wrapper around viscot-harmeme's SafeFaithScore (FaithScore
+ CAF) pipeline, driven by two remote vLLM servers.

Design goals
------------
* **Reuse, don't fork.** We import ``viscot-harmeme.faithscore`` modules
  and call the same ``_run_text_stages`` / ``_run_vem_stage`` functions
  that ``run.py`` uses at eval time, so training-time F and eval-time F
  are computed by the same code path.

* **Zero local GPU footprint.** Both text (Stage 1/2/4) and VEM (Stage 3)
  backends are configured as ``vllm-http``: they hit external vLLM servers
  by URL. Nothing is loaded into the training process.

* **Defensive.** Any of the following are transparently handled:
    - ``FS_TEXT_URL`` / ``FS_VEM_URL`` not set  -> ``score_batch`` returns
      neutral ``F = 0.0`` for all items, once-per-process warning.
    - Health-check on either judge server fails at first call
      -> permanent skip mode with explicit error message.
    - Any transient failure inside a scoring call -> returns ``F = 0.0``
      for the affected sample, logs, continues.

Env vars (read by ``SafeFaithScoreJudge.__init__``)
---------------------------------------------------
    FS_TEXT_URL       : e.g. http://x.x.x.x:8001/v1  (required to enable)
    FS_TEXT_MODEL     : served-model-name on that server (e.g. Qwen3-32B)
    FS_TEXT_API_KEY   : default 'EMPTY'
    FS_VEM_URL        : e.g. http://x.x.x.x:8002/v1  (required to enable)
    FS_VEM_MODEL      : served-model-name (e.g. Qwen3-VL-32B)
    FS_VEM_API_KEY    : default 'EMPTY'
    FS_API_WORKERS    : per-backend HTTP concurrency (default 16)
    FS_TAXONOMY       : stage4 taxonomy for CAF (default 'harmeme')
    FAITH_ALPHA       : F = alpha * FaithScore + (1-alpha) * CAF  (default 0.7)
    FS_TIMEOUT        : per-call HTTP timeout in seconds (default 120)
    FS_MAX_RETRY      : per-call retry count (default 3)
    FS_PER_SUB_STAGE2 : 1 = per-sub atomic-fact extraction (default),
                        0 = whole-description Stage 2 (faster, less granular)

Public API
----------
    judge = SafeFaithScoreJudge()      # reads env vars
    if judge.enabled:
        results = judge.score_batch(items)
        # items[i] = {"image_path", "cot", "pred", "label",
        #             "target_label" (optional), "question_id" (optional)}
        # results[i] = {"F", "faithscore_atomic", "caf_composite",
        #               "n_facts", "status"}
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Optional

# --------------------------------------------------------------------------- #
# Make the atomic-faithfulness pipeline importable                            #
# --------------------------------------------------------------------------- #
# ``VISCOT_ROOT`` must point at the checkout that provides the ``faithscore``
# package (atomic-fact extraction + visual-entailment judging). It is only
# required when the faithfulness reward is actually enabled; if it is unset
# the judge reports a clear error and disables itself.
VISCOT_ROOT = os.environ.get("VISCOT_ROOT", "")
if VISCOT_ROOT and VISCOT_ROOT not in sys.path:
    sys.path.insert(0, VISCOT_ROOT)


# --------------------------------------------------------------------------- #
# Logging helpers (prefix everything with [cf-faith] for easy grep)           #
# --------------------------------------------------------------------------- #
def _log(tag: str, msg: str) -> None:
    print(f"[cf-faith][{tag}] {msg}", flush=True)


# Deduplicate identical warnings so the training log is not flooded.
_WARNED_ONCE: set[str] = set()


def _log_once(tag: str, key: str, msg: str) -> None:
    if key in _WARNED_ONCE:
        return
    _WARNED_ONCE.add(key)
    _log(tag, msg)


# --------------------------------------------------------------------------- #
# Neutral / failure result helpers                                             #
# --------------------------------------------------------------------------- #
def _neutral_result(status: str) -> dict:
    """Result dict returned when the judge cannot produce a meaningful F.

    ``F = 0.0`` is intentionally used (not 0.5) so that a broken judge
    contributes zero reward instead of a positive baseline. Group-relative
    GRPO advantage then folds it into the group mean cleanly.
    """
    return {
        "F": 0.0,
        "faithscore_atomic": 0.0,
        "caf_composite": 0.0,
        "n_facts": 0,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# Judge                                                                        #
# --------------------------------------------------------------------------- #
class SafeFaithScoreJudge:
    """Batched SafeFaithScore judge over two remote vLLM servers.

    Lifecycle:
      1. ``__init__`` reads env vars, decides ``enabled`` (True if both URLs
         are set), and lazily builds the two ``vllm-http`` backends.
      2. First ``score_batch`` call runs a one-off health check on both
         servers; if either fails, we permanently switch to skip mode.
      3. Subsequent ``score_batch`` calls reuse the same backends.
    """

    # Class-level lock so ORMs called by different reward funcs share
    # a single judge instance safely (see ``get_singleton`` below).
    _singleton_lock = threading.Lock()
    _singleton: Optional["SafeFaithScoreJudge"] = None

    def __init__(self) -> None:
        # ---- read env ------------------------------------------------------
        self.text_url = os.environ.get("FS_TEXT_URL", "").strip()
        self.text_model = os.environ.get("FS_TEXT_MODEL", "").strip()
        self.text_api_key = os.environ.get("FS_TEXT_API_KEY", "EMPTY") or "EMPTY"
        self.vem_url = os.environ.get("FS_VEM_URL", "").strip()
        self.vem_model = os.environ.get("FS_VEM_MODEL", "").strip()
        self.vem_api_key = os.environ.get("FS_VEM_API_KEY", "EMPTY") or "EMPTY"
        self.api_workers = int(os.environ.get("FS_API_WORKERS", "16") or "16")
        self.timeout = float(os.environ.get("FS_TIMEOUT", "120") or "120")
        self.max_retry = int(os.environ.get("FS_MAX_RETRY", "3") or "3")
        self.taxonomy = os.environ.get("FS_TAXONOMY", "harmeme") or "harmeme"
        self.alpha = float(os.environ.get("FAITH_ALPHA", "0.7") or "0.7")
        self.per_sub_stage2 = bool(
            int(os.environ.get("FS_PER_SUB_STAGE2", "1") or "1")
        )

        # ---- decide enabled -----------------------------------------------
        self.enabled: bool = True
        self._disable_reason: Optional[str] = None
        if not self.text_url or not self.vem_url:
            missing = []
            if not self.text_url:
                missing.append("FS_TEXT_URL")
            if not self.vem_url:
                missing.append("FS_VEM_URL")
            self.enabled = False
            self._disable_reason = (
                f"missing env vars: {', '.join(missing)}. "
                "Faith reward will be 0 for all samples. "
                "To enable, export FS_TEXT_URL + FS_TEXT_MODEL + "
                "FS_VEM_URL + FS_VEM_MODEL in rl.sh before training."
            )
            _log("skip", self._disable_reason)
            return
        if not self.text_model:
            self.enabled = False
            self._disable_reason = "FS_TEXT_URL set but FS_TEXT_MODEL is empty"
            _log("skip", self._disable_reason)
            return
        if not self.vem_model:
            self.enabled = False
            self._disable_reason = "FS_VEM_URL set but FS_VEM_MODEL is empty"
            _log("skip", self._disable_reason)
            return
        if not (0.0 <= self.alpha <= 1.0):
            _log(
                "init",
                f"FAITH_ALPHA={self.alpha} clipped to [0,1]",
            )
            self.alpha = max(0.0, min(1.0, self.alpha))

        # ---- backends (lazy) ----------------------------------------------
        self._text_backend = None
        self._vem_backend = None
        self._health_checked = False

        # Shared F-cache: FaithORM and FaithToolORM both need F(τ) for the
        # ORIGINAL trajectory of every completion. GRPO calls reward
        # functions sequentially in the same batch, so we cache by the
        # SHA1 of (image_path, cot, pred, label) and let the second caller
        # skip its half of the compute.
        self._f_cache: dict[str, dict] = {}
        self._f_cache_lock = threading.Lock()
        # Cap to avoid unbounded growth across training steps.
        self._f_cache_cap = int(
            os.environ.get("FAITH_CACHE_MAX", "4096") or "4096"
        )

        _log(
            "init",
            f"SafeFaithScoreJudge configured "
            f"(alpha={self.alpha}, per_sub_stage2={self.per_sub_stage2}, "
            f"taxonomy={self.taxonomy}, api_workers={self.api_workers})",
        )
        _log("init", f"  text  : {self.text_model} @ {self.text_url}")
        _log("init", f"  vem   : {self.vem_model} @ {self.vem_url}")

    # ------------------------------------------------------------------ #
    @classmethod
    def get_singleton(cls) -> "SafeFaithScoreJudge":
        """Process-wide singleton. Both FaithORM and (future) FaithToolORM
        share the same judge to avoid duplicate backend objects."""
        with cls._singleton_lock:
            if cls._singleton is None:
                cls._singleton = cls()
            return cls._singleton

    # ------------------------------------------------------------------ #
    def _lazy_build_backends(self) -> bool:
        """Build the two vllm-http backends on first use. Returns True on
        success. On failure, flips ``self.enabled`` to False and returns
        False (all subsequent calls skip)."""
        if self._text_backend is not None and self._vem_backend is not None:
            return True

        try:
            from faithscore.backends.base import build_backend  # type: ignore
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"could not import viscot-harmeme faithscore module "
                f"(VISCOT_ROOT={VISCOT_ROOT!r}): {e}. "
                f"Faith reward disabled. Fix VISCOT_ROOT env var to point at "
                f"a checkout of viscot-harmeme."
            )
            _log("error", self._disable_reason)
            traceback.print_exc()
            return False

        try:
            self._text_backend = build_backend(
                "vllm-http",
                model_name=self.text_model,
                base_url=self.text_url,
                api_key=self.text_api_key,
                api_workers=self.api_workers,
                timeout=self.timeout,
                max_retry=self.max_retry,
            )
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"failed to build text backend "
                f"(url={self.text_url}, model={self.text_model}): {e}"
            )
            _log("error", self._disable_reason)
            traceback.print_exc()
            return False

        try:
            self._vem_backend = build_backend(
                "vllm-http",
                model_name=self.vem_model,
                base_url=self.vem_url,
                api_key=self.vem_api_key,
                api_workers=self.api_workers,
                timeout=self.timeout,
                max_retry=self.max_retry,
            )
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"failed to build VEM backend "
                f"(url={self.vem_url}, model={self.vem_model}): {e}"
            )
            _log("error", self._disable_reason)
            traceback.print_exc()
            return False

        return True

    # ------------------------------------------------------------------ #
    def _health_check(self) -> bool:
        """One-off health check: send one trivial call to each backend.
        Returns True on success. On failure, permanently disables the judge."""
        if self._health_checked:
            return self.enabled

        assert self._text_backend is not None and self._vem_backend is not None
        try:
            out_txt = self._text_backend.chat_batch(
                prompts=["Reply with the single word: OK"],
                image_paths_per_prompt=None,
                max_new_tokens=8,
                temperature=0.0,
            )
            if not out_txt or not out_txt[0]:
                raise RuntimeError(
                    f"text judge returned empty response "
                    f"(server up but chat_batch returned {out_txt!r}); "
                    f"check {self.text_url}/models and that "
                    f"served-model-name matches FS_TEXT_MODEL={self.text_model!r}"
                )
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"text judge health check failed at {self.text_url}: {e}. "
                f"Verify server:  curl -sf {self.text_url}/models"
            )
            _log("error", self._disable_reason)
            self._health_checked = True
            return False

        try:
            out_vem = self._vem_backend.chat_batch(
                prompts=["Reply with the single word: OK"],
                image_paths_per_prompt=[[]],  # explicitly empty image list
                max_new_tokens=8,
                temperature=0.0,
            )
            if not out_vem or not out_vem[0]:
                raise RuntimeError(
                    f"vem judge returned empty response "
                    f"(server up but chat_batch returned {out_vem!r}); "
                    f"check {self.vem_url}/models and that "
                    f"served-model-name matches FS_VEM_MODEL={self.vem_model!r}"
                )
        except Exception as e:
            self.enabled = False
            self._disable_reason = (
                f"vem judge health check failed at {self.vem_url}: {e}. "
                f"Verify server:  curl -sf {self.vem_url}/models"
            )
            _log("error", self._disable_reason)
            self._health_checked = True
            return False

        self._health_checked = True
        _log(
            "init",
            f"health check OK (text: {self.text_model!r} @ {self.text_url}; "
            f"vem: {self.vem_model!r} @ {self.vem_url})",
        )
        return True

    # ------------------------------------------------------------------ #
    def _build_args_namespace(self) -> argparse.Namespace:
        """Construct the ``args`` namespace expected by the ``run.py``
        helper functions we reuse."""
        return argparse.Namespace(
            # ``_run_text_stages`` and ``_run_vem_stage`` only read a small
            # subset of ``args``; provide sensible defaults for the rest so
            # the run.py signature stays compatible if we later reuse more.
            text_backend="vllm-http",
            text_model=self.text_model,
            text_url=self.text_url,
            text_api_key=self.text_api_key,
            vem_backend="vllm-http",
            vem_model=self.vem_model,
            vem_url=self.vem_url,
            vem_api_key=self.vem_api_key,
            per_sub_stage2=self.per_sub_stage2,
            skip_attribution=False,
            taxonomy=self.taxonomy,
            # unused by the helpers but referenced in some print statements
            input=None,
            question_file=None,
            output=None,
            metrics=None,
            intermediate=None,
            stage="all",
            gpu_mem_util=0.85,
            tp_size=1,
            max_model_len=8192,
            vem_max_model_len=32768,
            api_workers=self.api_workers,
            strip_verdict=False,   # cot is already verdict-free upstream
            limit=-1,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _cache_key(item: dict) -> str:
        """Build a stable cache key for an item. Uses (image_path, cot,
        pred, label) so that two callers with identical inputs collapse."""
        import hashlib
        payload = "|".join([
            str(item.get("image_path", "")),
            str(item.get("cot", "")),
            str(item.get("pred", "")),
            str(item.get("label", "")),
        ])
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[dict]:
        with self._f_cache_lock:
            return self._f_cache.get(key)

    def _cache_put(self, key: str, value: dict) -> None:
        with self._f_cache_lock:
            if len(self._f_cache) >= self._f_cache_cap:
                # FIFO evict half the cache when full (cheap; we don't
                # need LRU here).
                for k in list(self._f_cache.keys())[: self._f_cache_cap // 2]:
                    self._f_cache.pop(k, None)
            self._f_cache[key] = value

    # ------------------------------------------------------------------ #
    def score_batch(self, items: list[dict]) -> list[dict]:
        """Score a batch of trajectories.

        Args:
            items: each dict must have keys:
                image_path: absolute path to the original meme image
                cot: text of <think>...</think> (already stripped of
                     <sandbox_output> / <code> tags -- caller's responsibility)
                pred: "harmful" | "not_harmful" | anything else -> unknown
                label: ground-truth "harmful" | "not_harmful"
                target_label: optional target group name for CAF stage4
                              (empty string OK)
                question_id: optional identifier used only in logs

        Returns:
            Same length as ``items``. Each entry is a dict with keys:
                F, faithscore_atomic, caf_composite, n_facts, status
            ``F`` is always in [0,1]. ``status`` is 'ok' or an error tag.
        """
        n = len(items)
        if n == 0:
            return []
        if not self.enabled:
            _log_once(
                "skip",
                f"judge_disabled:{self._disable_reason}",
                f"score_batch called but judge is disabled: {self._disable_reason}",
            )
            return [_neutral_result("judge_disabled") for _ in range(n)]

        if not self._lazy_build_backends():
            return [_neutral_result("backend_build_failed") for _ in range(n)]

        if not self._health_check():
            return [_neutral_result("health_check_failed") for _ in range(n)]

        # ---- L2 per-item validation --------------------------------------
        # Build the "samples" list expected by _run_text_stages. Items that
        # fail validation get a neutral placeholder and are excluded from the
        # actual F computation.
        results: list[dict] = [_neutral_result("pending") for _ in range(n)]
        valid_indices: list[int] = []
        samples: list[dict] = []
        # Track which valid_indices came from a cache hit vs need scoring.
        # We keep this parallel to ``samples`` so the assemble loop below
        # can still map results correctly.
        n_cache_hits = 0
        for i, it in enumerate(items):
            image_path = (it.get("image_path") or "").strip()
            cot = (it.get("cot") or "").strip()
            pred = (it.get("pred") or "").strip().lower()
            label = (it.get("label") or "").strip().lower()
            qid = it.get("question_id", f"idx_{i}")
            if not image_path or not os.path.isfile(image_path):
                _log(
                    "warn",
                    f"sample qid={qid} missing image_path={image_path!r}, F=0",
                )
                results[i] = _neutral_result("missing_image")
                continue
            if not cot:
                _log("warn", f"sample qid={qid} empty cot, F=0")
                results[i] = _neutral_result("empty_cot")
                continue
            if pred not in ("harmful", "not_harmful"):
                _log(
                    "warn",
                    f"sample qid={qid} pred={pred!r} not in "
                    "{harmful,not_harmful}, F=0",
                )
                results[i] = _neutral_result("bad_pred")
                continue
            if label not in ("harmful", "not_harmful"):
                _log(
                    "warn",
                    f"sample qid={qid} label={label!r} not in "
                    "{harmful,not_harmful}, F=0",
                )
                results[i] = _neutral_result("bad_label")
                continue

            # ---- cache hit? -----------------------------------------------
            # If FaithORM already scored this exact (image, cot, pred, label)
            # in the same optimizer step, reuse it. This is safe because F
            # is deterministic (temperature=0.0 on both judges).
            cache_key = self._cache_key({
                "image_path": image_path,
                "cot": cot,
                "pred": pred,
                "label": label,
            })
            cached = self._cache_get(cache_key)
            if cached is not None:
                results[i] = dict(cached)
                results[i]["status"] = cached.get("status", "ok")
                n_cache_hits += 1
                continue

            samples.append(
                {
                    "question_id": qid,
                    "image_path": image_path,
                    "text": cot,
                    "pred": pred,
                    "label": label,
                    "label_id": 1 if label == "harmful" else 0,
                    "bbox_source": "code",   # matches Thyme eval output
                    "target_label": it.get("target_label", "") or "",
                    "_cache_key": cache_key,   # so assemble loop can write back
                }
            )
            valid_indices.append(i)

        if n_cache_hits > 0:
            _log(
                "score",
                f"cache hit: {n_cache_hits}/{n} samples reused (skipped F recompute)",
            )

        if not samples:
            # Either all items were invalid, or all were cache hits.
            return results

        # ---- run stages ---------------------------------------------------
        try:
            from faithscore.run import _run_text_stages, _run_vem_stage  # type: ignore
            from faithscore.score import atomic_faithscore  # type: ignore
            from faithscore.stage4_attribution import (  # type: ignore
                compute_composite_hallucination,
            )
        except Exception as e:
            _log(
                "error",
                f"could not import faithscore helpers from viscot-harmeme "
                f"(VISCOT_ROOT={VISCOT_ROOT!r}): {e}",
            )
            traceback.print_exc()
            for i in valid_indices:
                results[i] = _neutral_result("faithscore_import_failed")
            return results

        args_ns = self._build_args_namespace()

        try:
            _stage4_intermediate, attribution_block = _run_text_stages(
                samples, args_ns, self._text_backend
            )
        except Exception as e:
            _log("error", f"stage1/2/4 (text) failed on batch of {len(samples)}: {e}")
            traceback.print_exc()
            for i in valid_indices:
                results[i] = _neutral_result("text_stages_failed")
            return results

        try:
            (
                facts_per_sample,
                _fact_cats,
                _fact_to_subidx,
                scores_per_sample,
                _raws,
            ) = _run_vem_stage(samples, self._vem_backend)
        except Exception as e:
            _log("error", f"stage3 (VEM) failed on batch of {len(samples)}: {e}")
            traceback.print_exc()
            for i in valid_indices:
                results[i] = _neutral_result("vem_stage_failed")
            return results

        # ---- per-sample atomic FaithScore --------------------------------
        try:
            _, per_sample_atomic = atomic_faithscore(scores_per_sample)
        except Exception as e:
            _log("error", f"atomic_faithscore aggregation failed: {e}")
            traceback.print_exc()
            per_sample_atomic = [0.0 for _ in samples]

        # ---- batch-level CAF composite (attribution block is batch-wide) -
        # ``compute_composite_hallucination`` returns something like
        #   {"composite_attribution_faithfulness": <float>, ...}
        # which is a scalar per-batch. In the RL setting we want a per-sample
        # signal, so we allocate the batch composite score equally across all
        # samples that were IN SCOPE for stage4. Samples out of scope get 1.0
        # (i.e. no attribution hallucination detected).
        caf_scalar = 0.5  # neutral fallback if compute fails
        try:
            comp = compute_composite_hallucination(attribution_block or {})
            caf_scalar = float(comp.get("composite_attribution_faithfulness", 0.5))
            caf_scalar = max(0.0, min(1.0, caf_scalar))
        except Exception as e:
            _log("warn", f"compute_composite_hallucination failed: {e}; use 0.5")

        # ---- assemble per-sample F ---------------------------------------
        for k, sample_i in enumerate(valid_indices):
            fs_atom = float(per_sample_atomic[k]) if k < len(per_sample_atomic) else 0.0
            fs_atom = max(0.0, min(1.0, fs_atom))
            n_facts = len(facts_per_sample[k]) if k < len(facts_per_sample) else 0
            # F = alpha * FaithScore + (1-alpha) * CAF
            F = self.alpha * fs_atom + (1.0 - self.alpha) * caf_scalar
            F = max(0.0, min(1.0, F))
            entry = {
                "F": F,
                "faithscore_atomic": fs_atom,
                "caf_composite": caf_scalar,
                "n_facts": n_facts,
                "status": "ok",
            }
            results[sample_i] = entry
            # Write back to shared cache so a subsequent ORM in the same
            # optimizer step (e.g. FaithToolORM asking about F(τ)) skips
            # the recompute.
            ck = samples[k].get("_cache_key") if k < len(samples) else None
            if ck:
                self._cache_put(ck, entry)

        return results
