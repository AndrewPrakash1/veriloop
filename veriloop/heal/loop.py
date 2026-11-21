"""One healing round, end to end, with an append-only audit record per round.

Retraining ALWAYS continues from the current adapter, never from base — otherwise a
healing round silently discards prior learning and the loop looks like progress while
losing ground.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from veriloop.heal.gate import GateVerdict


@dataclass
class RoundRecord:
    round_index: int
    counts_by_class: Mapping[str, int]
    synthesised: int
    verdict: GateVerdict
    promoted: bool


def heal_round(round_index: int, seed: int, out_dir: str) -> RoundRecord:
    raise NotImplementedError("Phase 7")


def main(argv: list[str] | None = None) -> int:
    """`python -m veriloop.heal.loop --rounds 1` — the `make heal` target."""
    raise NotImplementedError("Phase 7")
