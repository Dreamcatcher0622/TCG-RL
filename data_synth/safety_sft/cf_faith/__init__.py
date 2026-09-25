"""CF-FaithGRPO components for Thyme safety RL training.

Currently exposed:
    * ``faith_judge.SafeFaithScoreJudge`` -- training-side wrapper around
      viscot-harmeme's FAITHSCORE + CAF pipeline, driven by two remote
      vLLM servers (Qwen3-32B text + Qwen3-VL-32B VEM).

Future (Step 3, side-car reachable):
    * ``cf_rollout.rollout_counterfactual``
    * ``weight_sync``
"""
