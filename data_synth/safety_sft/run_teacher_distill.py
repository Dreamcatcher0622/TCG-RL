"""
Step 2 of Thyme safety SFT data synthesis.

Runs the teacher Qwen2.5-VL-72B in Thyme's multi-turn sandbox paradigm
to produce ``N`` candidate trajectories per meme question.

For each input record (produced by ``build_distill_input.py``) we run
``run_evaluation``-style multi-turn rollout, but with three differences
vs. the original Thyme eval:

  1. ``do_sample=True`` with ``temperature``/``top_p`` so that we can draw
     ``N`` diverse trajectories per question.
  2. We override the system prompt with our safety prompt
     (``prompt_safety.txt``).
  3. We persist the **full ``conversation_history``** for every accepted
     run, including image paths, so that ``convert_to_swift_sft.py`` can
     later re-build a Thyme-format multi-turn SFT example.

Output (jsonl): one line per ``(question_id, trial_idx)`` trajectory:

    {
      "question_id": "...",
      "subset": "...",
      "trial_idx": 0..N-1,
      "image_path": "/abs/.../orig.png",
      "gt_label": "harmful" | "not_harmful",
      "predicted": "harmful" | "not_harmful" | "<unparsed>",
      "n_iterations": 3,
      "n_code_blocks": 1,
      "n_sandbox_images": 1,
      "thinking_chars": 412,
      "conversation": [        # exactly what Thyme run_evaluation builds
         {"role": "system",    "content": [{"type":"text", "text":"..."}]},
         {"role": "user",      "content": [{"type":"image","image":"/orig.png"},
                                           {"type":"text", "text":"<image>\nIs this meme..."}]},
         {"role": "assistant", "content": [{"type":"text","text":"<think>... <code>...</code>"},
                                           {"type":"text","text":"<sandbox_output>"},
                                           {"type":"image","image":"/.../crop_1.jpg"},
                                           {"type":"text","text":"</sandbox_output>"},
                                           {"type":"text","text":" final-step text </think><answer>...</answer>"}]}
      ],
      "raw_assistant_text": "...full concatenated assistant text...",
      "error": null
    }

Usage (single-process; the model itself is TP-sharded across all 8 GPUs):

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \\
    python run_teacher_distill.py \\
        --teacher_path /path/to/Qwen2.5-VL-72B-Instruct \\
        --input_jsonl  ./output/merged_train.jsonl \\
        --output_jsonl ./output/trajectories_raw.jsonl \\
        --temp_dir     ./_sandbox_tmp \\
        --prompt_path  ./prompt_safety.txt \\
        --num_per_question 4 \\
        --temperature 0.9 \\
        --top_p 0.95 \\
        --max_iterations 5 \\
        --max_new_tokens 2048 \\
        --tensor_parallel_size 8 \\
        --max_pixels 3211264 \\
        --limit 0
"""

from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import re
import sys
import time
import traceback
from typing import Any

# --- Load Thyme's sandbox module DIRECTLY from its file path.
#
# We deliberately avoid `from swift.trainers.sandbox import ...` because that
# triggers `swift/__init__.py` -> `swift/utils/logger.py` -> `modelscope`,
# which drags in a huge dep chain we do not need for distillation.
# `swift/trainers/sandbox.py` is self-contained (only stdlib + PIL + numpy +
# cv2 + autopep8 + timeout_decorator), so importing it as a single-file module
# works and is much cheaper.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_THYME_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_SANDBOX_PATH = os.path.join(_THYME_ROOT, "swift", "trainers", "sandbox.py")

import importlib.util as _importlib_util
_spec = _importlib_util.spec_from_file_location("thyme_sandbox", _SANDBOX_PATH)
if _spec is None or _spec.loader is None:
    raise ImportError(f"cannot locate sandbox.py at {_SANDBOX_PATH}")
_sandbox_mod = _importlib_util.module_from_spec(_spec)
sys.modules["thyme_sandbox"] = _sandbox_mod
_spec.loader.exec_module(_sandbox_mod)
execute_code_in_sandbox = _sandbox_mod.execute_code_in_sandbox  # type: ignore

# vLLM offline batched inference + Qwen-VL helpers.
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info  # type: ignore

# Conform to Thyme run_evaluation: stop at </code> or </answer> so that
# we can pause to either run sandbox code or terminate.
SPECIAL_STOP_STRINGS = ["</code>", "</answer>"]
CODE_RE = re.compile(
    r"<code>\s*(?:```\s*)?(?:python\s*)?([\s\S]*?)\s*(?:```\s*)?</code>",
    re.IGNORECASE,
)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_prompt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read().rstrip()


def _inject_img_path_if_missing(code: str, real_path: str) -> str:
    """Inject `img_path = "<real_path>"` at the top of the code if the teacher
    forgot to assign it.

    Why we need this:
      Our safety prompt intentionally leaves `# img_path = "..."` COMMENTED OUT
      in the few-shot examples to prevent the teacher from copying a fake
      placeholder string. Many teachers dutifully copy the *commented* form,
      never define `img_path`, and then crash with `NameError` when the next
      line `img = cv2.imread(img_path)` runs. Diagnostics on the pilot run
      showed 128 / 400 trajectories dying this exact way.

    Behaviour:
      * If the code already contains a top-level assignment `img_path = ...`
        (uncommented), we leave it untouched.
      * Otherwise we prepend `img_path = "<real_path>"` (and mirror it to
        `image_path` for models that use the alternate name).
      * Fully static analysis via `ast` — comments never count as assignments.
      * If the code is syntactically broken we fall back to a plain textual
        check for a ``^img_path\\s*=`` line.
    """
    if not real_path:
        return code

    def _has_top_level_assign(src: str) -> bool:
        try:
            tree = ast.parse(src)
        except SyntaxError:
            # Fallback: look for a non-comment line that starts an img_path=
            for raw_line in src.splitlines():
                stripped = raw_line.lstrip()
                if stripped.startswith("#"):
                    continue
                if re.match(r"(img_path|image_path)\s*=", stripped):
                    return True
            return False
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id in ("img_path", "image_path"):
                        return True
            # Walrus / AugAssign are unlikely for a path variable; ignore.
        return False

    if _has_top_level_assign(code):
        return code

    # Escape the path for safe embedding in a Python string literal.
    safe = real_path.replace("\\", "\\\\").replace('"', '\\"')
    prelude = (
        f'img_path = "{safe}"  # injected by run_teacher_distill (missing assignment)\n'
        f'image_path = img_path\n'
    )
    return prelude + code


_SANDBOX_MAGIC_PREFIX = "/mnt/data/temp_processed_images/"
_BAD_OUT_DIR_RE = re.compile(
    r"(['\"])(/tmp/|/var/tmp/|/dev/shm/|\./)(?P<name>[A-Za-z0-9_.\-/]+\.(?:jpg|jpeg|png|bmp|gif|tiff|webp))\1",
    re.IGNORECASE,
)


def _rewrite_out_paths_to_sandbox_dir(code: str) -> str:
    """Rewrite image output paths that the teacher hard-coded outside the
    sandbox-visible directory.

    Why we need this:
      The Thyme sandbox only recognizes files saved under its ``temp_output_dir``
      (or under the magic prefix ``/mnt/data/temp_processed_images/`` which it
      automatically remaps to ``temp_output_dir``). Teachers frequently follow
      the natural Python idiom ``out = '/tmp/crop.jpg'`` — the file is written
      successfully and its path is printed, but the sandbox silently drops it
      because ``/tmp/xxx.jpg`` is not under the whitelist.

      The pilot diagnostic on the previous run showed 136 / 400 trajectories
      failing exactly this way ("no ERROR block; code ran but produced no
      image"). Rewriting those literals fixes the whole class in one pass.

    Rule:
      Any string literal that looks like a saved image path under ``/tmp/``,
      ``/var/tmp/``, ``/dev/shm/`` or a bare ``./`` is rewritten to live under
      ``/mnt/data/temp_processed_images/``. The magic prefix is honoured by
      sandbox.py which will rewrite it again to the actual per-run temp dir.
      This preserves the original filename so downstream `print(out)` still
      matches the file we wrote.
    """
    def _sub(m: "re.Match[str]") -> str:
        quote = m.group(1)
        name = m.group("name")
        # Strip any leading path components, keep only the basename to be safe.
        base = os.path.basename(name)
        return f"{quote}{_SANDBOX_MAGIC_PREFIX}{base}{quote}"

    return _BAD_OUT_DIR_RE.sub(_sub, code)




def _build_initial_user_text(image_path: str) -> str:
    """First-turn user message text (mirrors Thyme's generate_prompt_final_qa)."""
    try:
        from PIL import Image
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


def _parse_predicted_label(assistant_text: str) -> str:
    m = ANSWER_RE.search(assistant_text)
    if not m:
        return "<unparsed>"
    raw = m.group(1).strip().lower()
    if raw in ("harmful", "harm", "yes_harmful"):
        return "harmful"
    if raw in ("not_harmful", "not harmful", "notharmful", "non_harmful", "safe", "no", "not"):
        return "not_harmful"
    if "not" in raw and "harm" in raw:
        return "not_harmful"
    if "harm" in raw:
        return "harmful"
    return "<unparsed>"


def _conversation_to_assistant_text(conversation: list[dict]) -> str:
    """Concatenate every text item authored by the assistant, in order.

    Sandbox <sandbox_output> wrapper tokens are authored by the assistant
    (Thyme writes them into the assistant turn together with the image),
    so they are included.  Image content items become a placeholder
    ``[IMAGE: <path>]`` so that simple regex-based filters (length, code
    counts) still work.
    """
    out_parts: list[str] = []
    for msg in conversation:
        if msg["role"] != "assistant":
            continue
        for item in msg["content"]:
            if item["type"] == "text":
                out_parts.append(item["text"])
            elif item["type"] == "image":
                out_parts.append(f"[IMAGE: {item['image']}]")
    return "".join(out_parts)


def _conversation_stats(conversation: list[dict]) -> dict[str, int]:
    text = _conversation_to_assistant_text(conversation)
    code_blocks = CODE_RE.findall(text)
    img_count = 0
    for msg in conversation:
        if msg["role"] != "assistant":
            continue
        for item in msg["content"]:
            if item["type"] == "image":
                img_count += 1
    think_match = re.search(r"<think>([\s\S]*?)</think>", text)
    thinking_chars = len(think_match.group(1)) if think_match else len(text)
    return {
        "n_code_blocks": len(code_blocks),
        "n_sandbox_images": img_count,
        "thinking_chars": thinking_chars,
    }


# ---------------------------------------------------------------------------
# Per-trajectory rollout (one (question, trial) pair)
# ---------------------------------------------------------------------------

def run_one_trajectory(
    *,
    llm: LLM,
    processor,
    sampling_params: SamplingParams,
    system_prompt: str,
    image_path: str,
    user_text: str,
    temp_dir: str,
    item_id: str,
    max_iterations: int,
) -> dict[str, Any]:
    """Run a single multi-turn sandbox rollout and return its full state."""

    conversation: list[dict] = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": user_text},
            ],
        },
    ]

    prev_exec_ctx: dict | None = None
    error_msg: str | None = None
    actual_iterations = 0

    current_image_for_code = image_path  # the path Thyme exposes to user-code

    for it in range(max_iterations):
        actual_iterations = it + 1

        # --- Build vLLM chat input from current conversation ---
        text_prompt = processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=(it == 0),
        )
        if it != 0 and text_prompt.endswith("<|im_end|>\n"):
            # When we resume an open assistant turn we must NOT close it.
            text_prompt = text_prompt[: -len("<|im_end|>\n")]

        image_inputs, video_inputs = process_vision_info(conversation)
        mm_data: dict[str, Any] = {}
        if image_inputs:
            mm_data["image"] = image_inputs
        if video_inputs:
            mm_data["video"] = video_inputs

        try:
            outputs = llm.generate(
                [{"prompt": text_prompt, "multi_modal_data": mm_data}],
                sampling_params=sampling_params,
                use_tqdm=False,
            )
        except Exception as e:
            error_msg = f"vllm.generate failed at iter {it}: {e}"
            break

        gen_text = outputs[0].outputs[0].text
        finish_reason = outputs[0].outputs[0].finish_reason
        # vLLM's stop_strings strip the stop suffix; re-append for parsing parity.
        stop_str = getattr(outputs[0].outputs[0], "stop_reason", None)
        if isinstance(stop_str, str) and stop_str in SPECIAL_STOP_STRINGS:
            gen_text_for_parse = gen_text + stop_str
        else:
            gen_text_for_parse = gen_text

        # --- Case A: a final answer is in this segment -> finish ---
        if "</answer>" in gen_text_for_parse:
            seg = [{"type": "text", "text": gen_text_for_parse}]
            if conversation[-1]["role"] == "assistant":
                conversation[-1]["content"] += seg
            else:
                conversation.append({"role": "assistant", "content": seg})
            break

        # --- Case B: a code block ended -> run sandbox & inject result ---
        m = CODE_RE.search(gen_text_for_parse)
        if m:
            code_to_execute = m.group(1).strip()
            # Fallback: some teacher outputs never assign img_path (they copy
            # the commented-out example line). Inject the real path so the
            # sandbox does not die with NameError.
            code_to_execute = _inject_img_path_if_missing(
                code_to_execute, current_image_for_code
            )
            # Fallback: teachers often save crops to /tmp/... which the
            # sandbox does not recognize. Rewrite those literals to the magic
            # prefix that sandbox.py remaps to temp_output_dir.
            code_to_execute = _rewrite_out_paths_to_sandbox_dir(code_to_execute)
            try:
                processed_paths, _stdout, sb_err, cur_ctx = execute_code_in_sandbox(
                    code_to_execute,
                    current_image_for_code,
                    item_id=item_id,
                    temp_output_dir=temp_dir,
                    previous_execution_context=prev_exec_ctx,
                )
            except Exception as e:
                processed_paths, sb_err, cur_ctx = [], f"sandbox raised {e}", None

            if not processed_paths:
                # Failed code: treat this generation as void, do not advance ctx.
                # We still record it in the conversation so downstream filters
                # can see the broken trajectory.
                fail_msg = (
                    f"<sandbox_output>[ERROR] {sb_err or 'no output produced'}"
                    "</sandbox_output>"
                )
                seg = [{"type": "text", "text": gen_text_for_parse + "\n" + fail_msg}]
                if conversation[-1]["role"] == "assistant":
                    conversation[-1]["content"] += seg
                else:
                    conversation.append({"role": "assistant", "content": seg})
                # Allow the model another iteration to recover.
                continue

            prev_exec_ctx = cur_ctx
            seg: list[dict] = [
                {"type": "text", "text": gen_text_for_parse},
                {"type": "text", "text": "<sandbox_output>"},
            ]
            for p in processed_paths:
                if os.path.isfile(p):
                    seg.append({"type": "image", "image": p})
                else:
                    seg.append({"type": "text", "text": str(p)})
            seg.append({"type": "text", "text": "</sandbox_output>"})

            if conversation[-1]["role"] == "assistant":
                conversation[-1]["content"] += seg
            else:
                conversation.append({"role": "assistant", "content": seg})
            continue

        # --- Case C: hit max-tokens or some odd stop without code/answer ---
        if finish_reason in ("length", "stop") and "<answer>" not in gen_text_for_parse:
            seg = [{"type": "text", "text": gen_text_for_parse}]
            if conversation[-1]["role"] == "assistant":
                conversation[-1]["content"] += seg
            else:
                conversation.append({"role": "assistant", "content": seg})
            error_msg = f"stopped without <answer>; finish_reason={finish_reason}"
            break

        # Default fall-through: append and let next iter try.
        seg = [{"type": "text", "text": gen_text_for_parse}]
        if conversation[-1]["role"] == "assistant":
            conversation[-1]["content"] += seg
        else:
            conversation.append({"role": "assistant", "content": seg})

    raw_assistant_text = _conversation_to_assistant_text(conversation)
    predicted = _parse_predicted_label(raw_assistant_text)
    stats = _conversation_stats(conversation)
    return {
        "conversation": conversation,
        "raw_assistant_text": raw_assistant_text,
        "predicted": predicted,
        "n_iterations": actual_iterations,
        "error": error_msg,
        **stats,
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_path", required=True)
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True)
    ap.add_argument("--temp_dir", required=True)
    ap.add_argument("--prompt_path", required=True)
    ap.add_argument("--num_per_question", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--max_iterations", type=int, default=5)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--tensor_parallel_size", type=int, default=8)
    ap.add_argument("--max_pixels", type=int, default=3211264)
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0,
                    help="if >0, only process the first N questions")
    ap.add_argument("--resume", action="store_true",
                    help="if set, skip (qid, trial) already in --output_jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_jsonl)) or ".", exist_ok=True)

    # Tell qwen-vl preprocessor to honour our pixel cap.
    os.environ.setdefault("MAX_PIXELS", str(args.max_pixels))
    os.environ.setdefault("FPS_MAX_FRAMES", "10")

    system_prompt = _read_prompt(args.prompt_path)

    # ---- Load questions ----
    questions: list[dict] = []
    with open(args.input_jsonl, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                questions.append(json.loads(ln))
    if args.limit > 0:
        questions = questions[: args.limit]

    # ---- Resume support ----
    done_keys: set[tuple[str, int]] = set()
    if args.resume and os.path.isfile(args.output_jsonl):
        with open(args.output_jsonl, "r", encoding="utf-8") as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    done_keys.add((r["question_id"], int(r["trial_idx"])))
                except Exception:
                    continue
        print(f"[resume] {len(done_keys)} (qid, trial) already finished")

    # ---- Init vLLM ----
    print(f"[init] loading {args.teacher_path} on TP={args.tensor_parallel_size} ...")
    llm = LLM(
        model=args.teacher_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
        limit_mm_per_prompt={"image": 8, "video": 0},
        seed=args.seed,
        dtype="bfloat16",
    )
    processor = AutoProcessor.from_pretrained(
        args.teacher_path, trust_remote_code=True
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=SPECIAL_STOP_STRINGS,
        include_stop_str_in_output=False,
        seed=None,  # let vLLM pick fresh per request → diversity across trials
    )

    # ---- Main loop ----
    out_f = open(args.output_jsonl, "a", encoding="utf-8")
    t0 = time.time()
    n_written = 0
    n_correct = 0
    n_with_sandbox = 0

    try:
        for q_idx, q in enumerate(questions):
            qid = q["question_id"]
            user_text = _build_initial_user_text(q["image_path"])
            for trial in range(args.num_per_question):
                if (qid, trial) in done_keys:
                    continue
                item_id = f"{qid}#trial{trial}"
                try:
                    res = run_one_trajectory(
                        llm=llm,
                        processor=processor,
                        sampling_params=sampling_params,
                        system_prompt=system_prompt,
                        image_path=q["image_path"],
                        user_text=user_text,
                        temp_dir=args.temp_dir,
                        item_id=item_id,
                        max_iterations=args.max_iterations,
                    )
                except Exception as e:
                    res = {
                        "conversation": [],
                        "raw_assistant_text": "",
                        "predicted": "<unparsed>",
                        "n_iterations": 0,
                        "n_code_blocks": 0,
                        "n_sandbox_images": 0,
                        "thinking_chars": 0,
                        "error": f"trajectory crashed: {e}\n{traceback.format_exc()}",
                    }

                rec = {
                    "question_id": qid,
                    "subset": q["subset"],
                    "trial_idx": trial,
                    "image_path": q["image_path"],
                    "gt_label": q["label"],
                    **{k: res[k] for k in (
                        "predicted", "n_iterations",
                        "n_code_blocks", "n_sandbox_images",
                        "thinking_chars", "error",
                    )},
                    "conversation": res["conversation"],
                    "raw_assistant_text": res["raw_assistant_text"],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out_f.flush()
                n_written += 1
                if rec["predicted"] == rec["gt_label"]:
                    n_correct += 1
                if rec["n_sandbox_images"] > 0:
                    n_with_sandbox += 1

            if (q_idx + 1) % 20 == 0 or q_idx + 1 == len(questions):
                dt = time.time() - t0
                rate = (q_idx + 1) / max(dt, 1e-6)
                acc = n_correct / max(n_written, 1)
                sb = n_with_sandbox / max(n_written, 1)
                print(
                    f"[{q_idx+1:6d}/{len(questions)}] "
                    f"trajectories={n_written:6d}  "
                    f"verdict_acc={acc:.3f}  sandbox_use={sb:.3f}  "
                    f"q/s={rate:.3f}  elapsed={dt/60:.1f}min"
                )
    finally:
        out_f.close()

    print(f"[done] wrote {n_written} trajectories to {args.output_jsonl}")


if __name__ == "__main__":
    main()
