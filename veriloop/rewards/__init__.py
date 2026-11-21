"""The reward engine. PURE: no I/O, no model, no randomness, no clock.

`state_verified(contract, audit_log, snapshot) -> float` is the training signal.
`claim_verified(final_message) -> float` is the deliberately naive comparison arm.

Why purity is a hard rule and not a style preference:
  * a pure reward is unit-testable, so the hackability audit can run on the CPU before a
    single GPU hour is spent;
  * a pure reward is recomputable by a second implementation, so a bug shows up as a
    DISAGREEMENT with AuditKIT's independent oracle instead of as silent agreement;
  * a reward that reads the model's prose cannot see the failure mode this project
    attacks — a shortcut that claims success is paid exactly as much as the real work,
    and paid faster.
"""

from veriloop.rewards.state_verified import state_verified
from veriloop.rewards.claim_verified import claim_verified
from veriloop.rewards.terms import Terms, compute_terms

__all__ = ["state_verified", "claim_verified", "Terms", "compute_terms"]
