"""The proposed reward. Read the SHAPE, not the constants.

    R_state = Γ · [ 0.5·1[M = 1] + 0.3·F + 0.2·E·1[M = S = F = 1] ]
              − 0.3·D − 0.3·L − 0.5·(1 − Γ)

Four structural decisions, and they are the contribution — the constants are tunables and
should survive being swapped wholesale:

  1. `Γ` is a HARD GATE, and `Γ = 0` drives the reward NEGATIVE. This is the anti-omission
     floor. A reward that pays for the mere absence of violations creates a "safe wrong"
     trajectory that banks a score for doing nothing, and the policy finds it.
  2. `1[M = 1]` is a CLIFF, not a ramp. Ramping it makes a workflow that completed 5 of 6
     mandatory steps look like progress when it is legally incomplete — the task is done or
     it is not.
  3. `E` sits behind the `M = S = F = 1` guard. An unguarded speed term lets "be fast"
     outbid "be right", which is the precise failure this reward exists to prevent.
  4. `D` and `L` SUBTRACT rather than gate. A gated looping episode collapses onto one dead
     reward value and stops returning gradient; a subtracted one can still learn to stop.

Precedence to state aloud in the write-up: **COMPLETION > STATE > SCHEMA > EFFICIENCY**.

---

## WHY THERE IS NO `0.1·S` TERM — A BUG FOUND WHILE IMPLEMENTING THIS

The first draft of this formula was

    R = Γ · [ 0.4·1[M=1] + 0.3·F + 0.1·S + 0.2·E·guard ] − ...

with `S` = share of schema-valid calls. **`S` is a constant given `Γ`, so that term was not
a schema reward at all — it was a flat 0.1 added to every non-gated episode.**

Proof: `Γ = 1` requires "every executed call had schema-valid arguments". Therefore every
recorded call is valid, therefore `S = valid/total = 1`. `Γ = 0` multiplies the whole
positive branch away anyway. So `S ∈ {1 under Γ=1, irrelevant under Γ=0}`.

The consequence was concrete and reachable. A degenerate policy — escalate, finish, do
nothing else — sets `Γ = 1`, scores `M = 1/7`, `F = 0`, so it collected the full `0.1·S`
which is **`0.10` for zero work**. Under mean-centred advantages that is *positive* whenever
the rest of the group went negative, so the policy would be nudged toward blanket escalation.
That is exactly the "safe wrong trajectory" the design forbids in `rlvr-reward-design`, and
the flat offset was the loophole.

The fix is structural, not a re-tune: `S` is **kept as a diagnostic** (`Terms.S`, reported in
every row and used by the failure classifier) and **not paid**. The weights renormalise to
`0.5 / 0.3 / 0.2`, which still sums to exactly `1.0` so a perfect episode scores `1.0` and a
do-nothing episode scores `0.0`.

Do not "restore" the `S` term. `tests/test_reward_terms.py::test_schema_validity_is_not_paid`
is the guard, and `W["schema"]` stays at `0.0` so the decision is visible rather than hidden.
"""

from __future__ import annotations

from typing import Mapping

from veriloop.contracts.schema import Contract
from veriloop.env.audit_log import AuditLog
from veriloop.rewards.terms import Terms, compute_terms

#: Tunables, pinned in Phase 2 and frozen before the first training run. Changing any of
#: these mid-experiment destroys the attribution the experiment exists to establish; a later
#: change is reported as a stated deviation, with its reason.
#:
#: `schema` is 0.0 ON PURPOSE. See the module docstring: S is constant under Γ = 1, so
#: paying it only creates a floor for a do-nothing trajectory.
W = {
    "mandatory": 0.5,
    "final_state": 0.3,
    "efficiency": 0.2,
    "schema": 0.0,
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
    return reward_from_terms(
        compute_terms(contract, log, snapshot, terminated=terminated, steps=steps)
    )


def reward_from_terms(t: Terms) -> float:
    """Combine the terms. Split out so a caller holding `Terms` can rescore without recomputing.

    Read this function as the design. Every line is a decision:
    """
    # 2. THE CLIFF. 1[M = 1], never M. A workflow that completed 5 of 6 mandatory steps has
    #    completed none of them as far as a compliance reviewer is concerned.
    completion = W["mandatory"] * (1.0 if t.M >= 1.0 else 0.0)

    #    The end state is graded (0 or 1) but continuous with completion: M says the calls
    #    were made, F says the world agrees.
    state = W["final_state"] * t.F

    # 3. THE GUARD. Efficiency is paid ONLY when the episode is fully compliant and correct.
    #    Unguarding this is the single change that turns the reward back into the failure it
    #    exists to prevent. `S` appears in the guard for continuity with the write-up; under
    #    Γ = 1 it is always 1.0, so the guard is effectively M = F = 1.
    guard = 1.0 if (t.M >= 1.0 and t.S >= 1.0 and t.F >= 1.0) else 0.0
    efficiency = W["efficiency"] * t.E * guard

    # 1. THE GATE. Everything above is multiplied by Γ, so Γ = 0 erases the positive branch
    #    entirely rather than merely discounting it.
    positive = t.gamma * (completion + state + efficiency + W["schema"] * t.S)

    # 4. SUBTRACTED, NOT GATED. A looping episode must still return a graded value, or it
    #    collapses onto one dead number and the policy can never learn to stop.
    return positive - W["duplicate"] * t.D - W["loop"] * t.L - W["gate_penalty"] * (1 - t.gamma)
