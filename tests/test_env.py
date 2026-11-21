"""Phase 1 — the world's invariants.

Determinism is not a nicety here: every downstream number, the reward audit and the
AuditKIT oracle replay all rest on "same seed, same bytes". A world that drifts makes the
results unreproducible and makes the disagreement-check between the two implementations
meaningless.
"""

from __future__ import annotations

import pytest

from veriloop.env.audit_log import canonical
from veriloop.env.tools import call_tool, run_sql_readonly


@pytest.mark.phase1
def test_same_seed_same_bytes(scenario_contract) -> None:
    from veriloop.env.db import open_world
    pre = scenario_contract["pre_state"]
    a = open_world(pre["seed"], pre, episode_id="e1").snapshot()
    b = open_world(pre["seed"], pre, episode_id="e2").snapshot()
    assert a == b, "the world is not deterministic in its seed"


@pytest.mark.phase1
def test_snapshot_is_a_deep_copy(reference_world) -> None:
    snap = reference_world.snapshot()
    snap["customers"].clear()
    assert reference_world.snapshot()["customers"], (
        "snapshot() handed out live rows — the reward could move the goalposts it scores"
    )


@pytest.mark.phase1
def test_audit_log_is_append_only(reference_world) -> None:
    log = reference_world.log
    log.record("lookup_customer", {"customer_id": "a8821"})
    log.assert_append_only()
    log.records[0] = None                       # simulate tampering
    with pytest.raises(Exception):
        log.assert_append_only()


@pytest.mark.phase1
def test_sequence_numbers_must_be_contiguous(reference_world) -> None:
    from veriloop.env.audit_log import CallRecord
    log = reference_world.log
    log.record("lookup_customer", {"customer_id": "a8821"})     # seq 0
    with pytest.raises(Exception):
        log.append(CallRecord(seq=7, step=1, tool="lookup_customer", args={},
                              args_canonical="{}", result_hash="x", ok=True, schema_valid=True))


@pytest.mark.phase1
def test_read_only_sql_is_enforced_at_the_connection(reference_world) -> None:
    """UPDATE/DELETE/DDL must be refused, and the refusal must leave no state change."""
    before = reference_world.snapshot()
    for q in ["UPDATE accounts SET status='open'",
              "DELETE FROM audit_log",
              "DROP TABLE accounts",
              "WITH x AS (SELECT 1) DELETE FROM accounts"]:
        assert not run_sql_readonly(reference_world, q).ok, f"{q!r} was allowed"
    assert reference_world.snapshot() == before


@pytest.mark.phase1
def test_schema_fuzz_is_recorded_as_invalid_not_raised(reference_world) -> None:
    """An unknown tool or a bad arg is a DATA POINT, not a crash.

    Raising here would turn the `schema_fuzz` attack into a test error instead of a score.
    """
    before = reference_world.snapshot()
    r = call_tool(reference_world, 0, "teleport_customer", {"customer_id": "a8821"})
    assert not r.ok
    assert reference_world.log.records[-1].schema_valid is False
    assert reference_world.snapshot() == before, "an invalid call changed the world"


@pytest.mark.phase1
def test_every_call_is_logged_before_state_changes(reference_world) -> None:
    before = reference_world.snapshot()
    call_tool(reference_world, 0, "record_check",
              {"check_type": "sanctions", "evidence_ref": "D-901"})
    assert reference_world.log.records, "a call executed without being logged"
    assert reference_world.snapshot() != before, "record_check did not change the world"


@pytest.mark.phase1
def test_canonicalisation_makes_key_order_irrelevant() -> None:
    assert canonical({"a": 1, "b": 2}) == canonical({"b": 2, "a": 1})
    assert canonical({"a": 1}) != canonical({"a": "1"})


@pytest.mark.phase1
def test_blanket_escalate_leaves_the_world_unchanged_enough_to_be_flagged(reference_world) -> None:
    """The negative-control scenario is only scorable if a no-action episode is detectable."""
    pytest.skip("enable once the negative-control contract lands (Phase 3)")
