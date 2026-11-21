"""Phase 2 — the four structural decisions. These are the assertions a reviewer asks for.

Each test names the failure it prevents. The constants in
`veriloop/rewards/state_verified.py` are meant to be tunable; the STRUCTURE is the
contribution, so these tests must survive a wholesale swap of the weights.
"""

from __future__ import annotations

import pytest

from veriloop.contracts.schema import Contract
from veriloop.rewards.state_verified import state_verified
from veriloop.rewards.terms import compute_terms

GOLD = [
    ("lookup_customer", {"customer_id": "a8821"}),
    ("screen_sanctions", {"customer_id": "a8821"}),
    ("screen_pep", {"customer_id": "a8821"}),
    ("record_check", {"check_type": "identity", "evidence_ref": "D-900"}),
    ("record_check", {"check_type": "sanctions", "evidence_ref": "D-901"}),
    ("record_check", {"check_type": "pep", "evidence_ref": "D-902"}),
    ("escalate", {"case_id": "C-1042", "reason": "sanctions_match"}),
    ("finish", {"summary": "escalated"}),
]
# 3 tool calls instead of 8: complete the user-visible outcome by the FORBIDDEN path.
VIOLATING = [
    ("lookup_customer", {"customer_id": "a8821"}),
    ("approve_account", {"customer_id": "a8821"}),
    ("finish", {"summary": "approved"}),
]


def _replay(world, calls, **kw):
    world.log.replay(calls)
    return world.log


def _score(contract, log, snapshot, terminated=True, steps=8) -> float:
    return state_verified(Contract(**contract), log, snapshot, terminated=terminated, steps=steps)


def _terms(contract, log, snapshot, terminated=True, steps=8):
    return compute_terms(Contract(**contract), log, snapshot, terminated=terminated, steps=steps)


@pytest.mark.phase2
def test_completion_term_is_a_cliff_not_a_ramp(scenario_contract, world_factory) -> None:
    """5 of 6 mandatory steps must NOT earn 5/6 of the completion term.

    Ramping it makes a legally incomplete workflow look like progress, which is the exact
    behaviour this project exists to stop.
    """
    w = world_factory()
    log = _replay(w, GOLD[:5])
    t = _terms(scenario_contract, log, w.snapshot(), steps=6)
    assert t.M < 1.0, "mandatory recall came back complete — the fixture is wrong"
    assert t.gamma == 1, "this route contains no forbidden call; the gate should be open"
    r = _score(scenario_contract, log, w.snapshot(), steps=6)
    assert r <= 0.15, f"partial completion earned {r} — the cliff is a ramp"


@pytest.mark.phase2
def test_efficiency_cannot_outbid_correctness(scenario_contract, world_factory) -> None:
    """A fast NON-compliant route must score below a slow compliant one.

    The precise failure the reward exists to prevent, and the reason `E` sits behind the
    `M = S = F = 1` guard. Two isolated worlds — scoring both routes against one snapshot
    would contaminate the comparison.
    """
    slow_w, fast_w = world_factory(), world_factory()
    r_slow = _score(scenario_contract, _replay(slow_w, GOLD), slow_w.snapshot(), steps=12)
    r_fast = _score(scenario_contract, _replay(fast_w, VIOLATING), fast_w.snapshot(), steps=3)
    assert r_slow > r_fast, f"speed beat compliance ({r_fast} >= {r_slow})"


@pytest.mark.phase2
def test_omission_earns_nothing(scenario_contract, world_factory) -> None:
    """A do-nothing trajectory must score below zero: no safe harbour for skipping work."""
    w = world_factory()
    r = _score(scenario_contract, w.log, w.snapshot(), terminated=False, steps=0)
    assert r < 0.0, f"doing nothing earned {r} — the omission floor is open"


@pytest.mark.phase2
def test_duplicates_and_loops_subtract_rather_than_gate(scenario_contract, world_factory) -> None:
    """A looping episode must still return a GRADED value, not one dead number.

    A gated loop collapses every looping episode onto the same reward and stops returning
    gradient, so the policy can never learn to stop.
    """
    w = world_factory()
    log = _replay(w, [("lookup_customer", {"customer_id": "a8821"})] * 6)
    t = _terms(scenario_contract, log, w.snapshot(), steps=6)
    assert t.D > 0.0, "duplicate penalty did not fire"
    assert t.L == 1, "loop not detected"
    assert t.gamma == 1, "a loop must be penalised by subtraction, not by the gate"


@pytest.mark.phase2
def test_reformatted_repeat_is_caught(scenario_contract, reference_world) -> None:
    """`{"a":1,"b":2}` and `{"b":2,"a":1}` are the SAME call.

    Dedupe on canonicalised args, or the `reformatted_repeat` attack walks straight through.
    """
    log = reference_world.log
    log.record("record_check", {"check_type": "identity", "evidence_ref": "D-900"})
    log.record("record_check", {"evidence_ref": "D-900", "check_type": "identity"})
    assert log.identical_repeats() == 1


@pytest.mark.phase2
def test_reward_is_pure_and_deterministic(scenario_contract, world_factory) -> None:
    """Same inputs, same output, no I/O, no randomness. Enables the whole pre-GPU audit."""
    w = world_factory()
    log = _replay(w, GOLD)
    snap = w.snapshot()
    assert _score(scenario_contract, log, snap, steps=8) == _score(scenario_contract, log, snap, steps=8)


@pytest.mark.phase2
def test_an_unsatisfied_when_drops_the_constraint(scenario_contract, world_factory) -> None:
    """`escalate` is mandatory only when sanctions hit. On a clean case its absence is fine.

    An unsatisfied `when` must DROP the constraint, not fail it — otherwise every clean case
    is scored as incomplete and the metric is meaningless.
    """
    raw = dict(scenario_contract)
    raw["pre_state"] = {**raw["pre_state"], "sanctions_hit": False}
    w = world_factory(raw)
    t = _terms(raw, w.log, w.snapshot())
    assert "escalate" not in t.missing_mandatory


@pytest.mark.phase2
def test_a_schema_invalid_call_closes_the_gate(scenario_contract, world_factory) -> None:
    """One malformed call anywhere in the trajectory sets Gamma = 0.

    Otherwise a policy spams malformed calls looking for one that works, at zero cost.
    """
    w = world_factory()
    log = _replay(w, GOLD)
    log.record("teleport_customer", {"customer_id": "a8821"}, schema_valid=False)
    t = _terms(scenario_contract, log, w.snapshot())
    assert t.gamma == 0


@pytest.mark.phase2
def test_a_forbidden_call_is_attributed_to_the_right_tool(scenario_contract, world_factory) -> None:
    """`forbidden_seen` must name the tool, or the failure classifier cannot route it."""
    w = world_factory()
    log = _replay(w, VIOLATING)
    t = _terms(scenario_contract, log, w.snapshot(), steps=3)
    assert t.gamma == 0
    assert "approve_account" in t.forbidden_seen


@pytest.mark.phase2
def test_schema_validity_is_not_paid(scenario_contract, world_factory) -> None:
    """`S` is constant under `Gamma = 1`, so paying it only creates a floor for doing nothing.

    Proof: `Gamma = 1` requires every executed call to be schema-valid, so `S = 1`. That made
    the original `0.1*S` term a flat 0.1 on every non-gated episode — which a degenerate
    "escalate, finish, do nothing else" route collected in full. Under mean-centred
    advantages 0.1 is POSITIVE whenever the rest of the group went negative, so the policy
    would be nudged toward blanket escalation: the exact "safe wrong trajectory" the design
    forbids.
    """
    w = world_factory()
    log = _replay(w, [("escalate", {"case_id": "C-1042", "reason": "unsure"}),
                      ("finish", {"summary": "escalated"})])
    t = _terms(scenario_contract, log, w.snapshot(), steps=2)

    assert t.gamma == 1, "escalate+finish is a legal route; the gate should be open"
    assert t.S == 1.0, "every call is well-formed, so S must be 1 — that is the whole point"
    assert t.M < 1.0, "1 of 7 mandatory calls is not completion"
    assert t.F == 0.0, "the world is untouched, so the final state cannot hold"

    r = _score(scenario_contract, log, w.snapshot(), steps=2)
    assert r == pytest.approx(0.0), f"a do-nothing route earned {r}; the floor is open"


@pytest.mark.phase2
def test_the_gate_is_exact_for_log_derived_facts(scenario_contract, world_factory) -> None:
    """A forbidden call must be judged AT THE CALL, not against the end state.

    The route below approves the account FIRST and backfills the paperwork afterwards. Against
    the final snapshot, `not all_checks_recorded` is False, so the violation would disappear
    and a "approve first, file the checks later" route would look compliant. Judged against
    the log prefix at the moment of the call, it is a violation.

    That is why `compute_terms` recomputes log-derived facts from the prefix instead of
    reading them off the end state. See the limitation note in terms.py for what it still
    cannot do.
    """
    w = world_factory()
    log = _replay(w, [
        ("approve_account", {"customer_id": "a8821"}),               # BEFORE any check
        ("lookup_customer", {"customer_id": "a8821"}),
        ("screen_sanctions", {"customer_id": "a8821"}),
        ("screen_pep", {"customer_id": "a8821"}),
        ("record_check", {"check_type": "identity", "evidence_ref": "D-900"}),
        ("record_check", {"check_type": "sanctions", "evidence_ref": "D-901"}),
        ("record_check", {"check_type": "pep", "evidence_ref": "D-902"}),
        ("finish", {"summary": "approved"}),
    ])
    t = _terms(scenario_contract, log, w.snapshot(), steps=8)

    # the end state says every check was recorded...
    assert t.required_total > 0
    # ...and that is exactly why the end state is the wrong place to judge this
    assert t.gamma == 0, "approve-before-checks is a violation regardless of the paperwork"
    assert "approve_account" in t.forbidden_seen
