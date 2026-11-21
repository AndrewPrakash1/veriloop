"""The independent outcome oracle.

Re-executes the episode's effects from its recorded route against a fresh world built
from the same seed and pre-state, then asserts the contract's `final_state`. It does not
call `state_verified`; it deliberately re-derives the verdict so the two can disagree.
"""

from __future__ import annotations

from typing import Any, Mapping

from veriloop.contracts.schema import Contract


def replay_and_assert(
    contract: Contract,
    route: list[dict],
    seed: int,
) -> Mapping[str, Any]:
    """Replay `route`, then evaluate every `final_state` assertion independently.

    Returns {"verified": bool, "failures": [...], "unscored": [...]}. A missing log
    field must produce `unscored`, not `False`.
    """
    raise NotImplementedError("Phase 4")
