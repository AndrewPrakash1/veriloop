"""Policy-side glue. No task logic lives here.

This is the ONLY package that knows a model exists. `veriloop/` must stay importable and
usable with no GPU, no model download and no network — that property is what lets the
reward engine and the hackability audit be built and tested before any GPU hour is spent.
"""

from agents.harness import ScriptedPolicy, build_harness, scripted_policy
from agents.strategy import build_strategy

__all__ = ["build_strategy", "build_harness", "ScriptedPolicy", "scripted_policy"]
