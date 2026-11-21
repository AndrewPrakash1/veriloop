"""Scripted attack transcripts — hand-authored tool-call routes, no model in the loop.

This is the pre-GPU evidence. Each probe is scored under BOTH rewards:
`claim_verified` (the naive one) and `state_verified` (the proposed one), with the same
scenario state and the same contract. Needs no GPU, no API key, no model download.

Recipe: references/reward-hackability-audit.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Attack:
    attack_id: str
    description: str
    probes: str                     # what hole it is testing for
    transcript: list[dict]          # [{tool, args}, ...] — a complete plausible route
    final_message: str
    non_trivial: bool = True        # counted as "closed" only when True


#: The catalogue lives in `audit/attacks.yaml` so it is reviewable without reading code.
#: `load_attacks()` reads it into `Attack` objects.
def load_attacks(path: str = "audit/attacks.yaml") -> list[Attack]:
    """Load the attack catalogue. Every entry needs a non-empty transcript."""
    raise NotImplementedError("Phase 2")


def score_attacks(attacks: list[Attack]) -> list[dict]:
    """Score every attack under both rewards, on the same scenario and contract.

    Each row is: attack_id, probes, claim_verified, state_verified, terms, closed.
    """
    raise NotImplementedError("Phase 2")


def summarise(rows: list[dict]) -> dict:
    """Count non-trivial attacks closed, and assert the three invariants:
      * every cheat scores <= 0.1 under the proposed reward
      * the compliant control scores 1.0
      * a compliant route with a broken final state scores < 1.0
    A reward whose defences are only prose regresses silently — hence the assertion.
    """
    raise NotImplementedError("Phase 2")


def compliant_control() -> Attack:
    """The gold route for the reference scenario. The control row; must score maximum."""
    raise NotImplementedError("Phase 2")
