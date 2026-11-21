"""Derives the model-facing tool schemas from `veriloop.env.tools.TOOL_SPECS`.

One source of truth. Hand-writing the schemas a second time is how the model ends up
offered a tool the environment does not have — which is itself one of the failure modes
under study, so do not introduce it by accident.
"""

from __future__ import annotations


def openai_tools() -> list[dict]:
    raise NotImplementedError("Phase 1")


def prompt_block() -> str:
    """Human-readable tool list for the prompt (an alternative to function-calling)."""
    raise NotImplementedError("Phase 1")
