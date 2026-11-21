"""The six terms, each computed separately so the audit can attribute a score.

    Gamma = 1  iff  no forbidden call occurred
           and  every executed call had schema-valid arguments
           and  the episode ended in a terminal action within the step budget

    M = required calls present with valid args / required calls
    S = valid_calls / total_calls
    F = 1 iff every final-state assertion holds
    E = clamp(1 - (steps - min_steps) / (budget - min_steps), 0, 1)
    D = identical_calls / total_calls          # identical on CANONICALISED args
    L = 1 if a repeated (tool, args) cycle is detected, else 0

Keep every term separately inspectable. When an arm scores lower than expected, the only
way to find out why is to see which term moved.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from veriloop.contracts.schema import Contract
from veriloop.env.audit_log import AuditLog


@dataclass(frozen=True)
class Terms:
    gamma: int
    M: float
    S: float
    F: float
    E: float
    D: float
    L: int
    missing_mandatory: tuple[str, ...]      # for the failure classifier
    forbidden_seen: tuple[str, ...]

    def as_row(self) -> dict:
        """Flat dict for logs and for the hackability table."""
        raise NotImplementedError("Phase 2")


def compute_terms(
    contract: Contract,
    log: AuditLog,
    snapshot: Mapping[str, list[dict]],
    *,
    terminated: bool,
    steps: int,
) -> Terms:
    """Compute all six terms.

    Implementation notes that are load-bearing:
      * `must_call[i].when` is evaluated against the FINAL snapshot. An unsatisfied `when`
        DROPS the constraint; it does not fail it.
      * `must_not_call[i].when` is evaluated at the step index recorded on the call, not
        at the end — a call can be forbidden in one state and fine in another.
      * `D` and `L` are computed on canonicalised args (`sort_keys=True`, separators fixed),
        or `reformatted_repeat` slips past.
      * `E` is computed but the reward only ever pays it under the `M = S = F = 1` guard.
        Do not apply the guard here; apply it in `state_verified`, so this function stays
        a faithful description of the episode.
    """
    raise NotImplementedError("Phase 2")


def multiplicity_ok(log: AuditLog, tool: str) -> bool:
    """Helper: did `tool` fire exactly once when the contract implies it should?"""
    raise NotImplementedError("Phase 2")
