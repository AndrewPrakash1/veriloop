"""The proposed reward. Read the SHAPE, not the constants.

    R_state = Gamma * [ 0.4*1[M = 1] + 0.3*F + 0.1*S + 0.2*E*1[M = S = F = 1] ]
              - 0.3*D - 0.3*L - 0.5*(1 - Gamma)

Four structural decisions, and they are the contribution — the constants are tunables and
should survive being swapped wholesale:

  1. `Gamma` is a HARD GATE, and `Gamma = 0` drives the reward NEGATIVE. This is the
     anti-omission floor. A reward that pays for the mere absence of violations creates a
     "safe wrong" trajectory that banks a score for doing nothing, and the policy finds it.
  2. `1[M = 1]` is a CLIFF, not a ramp. Ramping it makes a workflow that completed 5 of 6
     mandatory steps look like progress when it is legally incomplete — the task is done
     or it is not.
  3. `E` sits behind the `M = S = F = 1` guard. An unguarded speed term lets "be fast"
     outbid "be right", which is the precise failure this reward exists to prevent.
  4. `D` and `L` SUBTRACT rather than gate. A gated looping episode collapses onto one
     dead reward value and stops returning gradient; a subtracted one can still learn to
     stop.

Precedence to state aloud in the write-up: COMPLETION > STATE > SCHEMA > EFFICIENCY.
"""

from __future__ import annotations

from typing import Mapping

from veriloop.contracts.schema import Contract
from veriloop.env.audit_log import AuditLog

#: Tunables, pinned in Phase 2 and frozen before the first training run. Changing any of
#: these mid-experiment destroys the attribution the experiment exists to establish; a
#: later change is reported as a stated deviation with its reason.
W = {
    "mandatory": 0.4,
    "final_state": 0.3,
    "schema": 0.1,
    "efficiency": 0.2,
    "duplicate": 0.3,
    "loop": 0.3,
    "gate_penalty": 0.5,
}


def state_verified(
    contract: Contract,
    log: AuditLog,
    snapshot: Mapping[str, list[dict]],
    *,
    terminated: bool,
    steps: int,
) -> float:
    """Score one episode. Deterministic, pure, and identical on every replay."""
    raise NotImplementedError("Phase 2")
