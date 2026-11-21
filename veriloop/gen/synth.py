"""The generator: a template plus a seeded draw becomes one scenario."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from veriloop.contracts.schema import Contract


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str                      # E1 | E2 | E3 | E4 | train
    domain: str                     # kyc | analytics
    template: str
    difficulty: int                 # expected step count
    brief: str
    pre_state: Mapping[str, Any]
    contract: Contract
    gold_route: list[dict]          # [{tool, args}, ...] — the independent route
    is_negative: bool               # correct behaviour is no action / decline / escalate


def generate(
    template: str,
    seed: int,
    *,
    split: str,
    n: int = 1,
    difficulty: int | None = None,
) -> list[Scenario]:
    """Emit `n` scenarios from one template.

    Rules the generator must hold to:
      * Gold is derived from the contract, not written separately — and the gold route is
        checked against the contract by `validate()` before it is written out.
      * Every scenario is emitted with a DISTINCT set of values. Two quantities with the
        same value make argument-constraint matching ambiguous.
      * Negative-control scenarios (~15% overall) are generated deliberately, by the
        templates in `templates/negatives.py`, not left to chance.
      * The injected-instruction variants go to `E3` only.
    """
    raise NotImplementedError("Phase 3")
