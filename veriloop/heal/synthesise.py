"""Targeted synthesis: one deterministic template per failure class.

Feeds new scenarios back through `veriloop.gen`, so the new contracts are validated by the
same code path as the original ones, and gold is still produced by construction.
"""

from __future__ import annotations

from typing import Any

from veriloop.heal.classify import FailureClass

#: class -> generator template name in veriloop.gen.templates
TEMPLATES_BY_CLASS: dict[FailureClass, str] = {
    FailureClass.SCHEMA_MISMATCH:          "drill_argument_shapes",
    FailureClass.WRONG_TOOL:               "drill_near_miss_tools",
    FailureClass.SKIPPED_MANDATORY_STEP:   "same_contract_longer_horizon",
    FailureClass.FORBIDDEN_ORDER:          "tempting_shortcut_variants",
    FailureClass.EXECUTION_ERROR:          "schema_error_recovery",
    FailureClass.LOOP:                     "budget_tightened",
    FailureClass.FALSE_BLOCK:              "correct_no_action_cases",
}


def synthesise(failure_class: FailureClass, n: int, seed: int) -> list[Any]:
    """Emit `n` new scenarios isolating one class. Deterministic in `seed`."""
    raise NotImplementedError("Phase 7")
