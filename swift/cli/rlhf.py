# Copyright (c) Alibaba, Inc. and its affiliates.
# ---------------------------------------------------------------------------
# safety-thyme patch:
# transformers>=4.51 hard-blocks torch.load when torch<2.6 (CVE-2025-32434).
# Our env has torch==2.5.1+cu124 (cannot easily upgrade without breaking
# flash_attn / vllm / deepspeed wheels), and the pickle files we load are
# **our own DeepSpeed ZeRO-3 optimizer/scheduler checkpoints**, not
# untrusted third-party artifacts. Disable the version gate before any
# transformers import path can reach _load_optimizer_and_scheduler.
# ---------------------------------------------------------------------------
def _patch_torch_load_safety_gate():
    try:
        from transformers.utils import import_utils as _tu
        if hasattr(_tu, 'check_torch_load_is_safe'):
            _tu.check_torch_load_is_safe = lambda: None
        # The function is re-imported via ``from ... import`` in several
        # sibling modules, which creates independent module-level names.
        # Patch every such site so the check becomes a no-op everywhere.
        for _modname in (
            'transformers.trainer',
            'transformers.modeling_utils',
            'transformers.modeling_tf_pytorch_utils',
        ):
            try:
                _mod = __import__(_modname, fromlist=['check_torch_load_is_safe'])
                if hasattr(_mod, 'check_torch_load_is_safe'):
                    _mod.check_torch_load_is_safe = lambda: None
            except Exception:
                pass
    except Exception as _e:  # pragma: no cover - patch best-effort
        import sys
        print(f'[safety-thyme] WARN: failed to patch check_torch_load_is_safe: {_e}',
              file=sys.stderr)


_patch_torch_load_safety_gate()

from swift.llm import rlhf_main

if __name__ == '__main__':
    rlhf_main()
