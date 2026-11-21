"""The redeploy gate — the safety property of the whole loop.

Two conditions, both required:
  1. the target metric RISES on a frozen slice, by at least the pre-registered margin;
  2. the trap split (E3) does NOT regress beyond its tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass
class GateVerdict:
    promote: bool
    reason: str
    scores: Mapping[str, Any]


class RedeployGate:
    def __init__(self, margin: float = 0.0, trap_tolerance: float = 0.0):
        self.margin = margin
        self.trap_tolerance = trap_tolerance

    def evaluate(self, before: Mapping[str, float], after: Mapping[str, float]) -> GateVerdict:
        """`before`/`after` carry verified_success, trap_success and shortcut_rate."""
        raise NotImplementedError("Phase 7")
