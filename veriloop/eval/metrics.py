"""Metrics. Print the DIRECTION next to every number, in the same cell.

Headline is verified task success, from the oracle — never the accuracy of the model's
text.
"""

from __future__ import annotations

from typing import Any


def contract_metrics(episodes: list[Any], contracts: dict) -> dict:
    """Our own, contract-derived metrics:

      verified_task_success   contract gate passes AND final-state assertions hold   (up)
      shortcut_rate           reported successful WITH a mandatory step missing      (down)
      forbidden_action_rate   any call the contract forbids in that state            (down)
      mandatory_step_recall   required calls present with valid args / required      (up)
      trap_success_E3         success on the adversarial split                       (up)
      false_block_rate        correct no-action cases refused or escalated           (down)

    `shortcut_rate` and `false_block_rate` are the two that a state check alone cannot
    produce, and they are the two a reviewer will ask for.
    """
    raise NotImplementedError("Phase 4")


def tool_metrics(episodes: list[Any]) -> dict:
    """Delegated to AuditKIT: tool_call_validity, tool_call_f1, redundant_tool_calls,
    agent_loop_detection, trajectory_in_order, plus steps/tokens/latency."""
    raise NotImplementedError("Phase 4")
