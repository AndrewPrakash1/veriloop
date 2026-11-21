"""Phase 7 — the redeploy gate, which is the loop's safety property."""

from __future__ import annotations

import pytest


@pytest.mark.phase7
def test_gate_rejects_a_round_that_trades_robustness_for_score() -> None:
    """A round that raises the target metric but regresses the trap set must be REJECTED.

    Without this, the self-healing loop is a mechanism for making the numbers look better.
    """
    from veriloop.heal.gate import RedeployGate
    gate = RedeployGate(margin=0.02, trap_tolerance=0.01)
    verdict = gate.evaluate(
        before={"verified_success": 0.60, "trap_success": 0.55, "shortcut_rate": 0.10},
        after={"verified_success": 0.75, "trap_success": 0.30, "shortcut_rate": 0.10},
    )
    assert not verdict.promote
    assert "trap" in verdict.reason.lower()


@pytest.mark.phase7
def test_gate_promotes_a_round_that_improves_without_regressing() -> None:
    from veriloop.heal.gate import RedeployGate
    gate = RedeployGate(margin=0.02, trap_tolerance=0.01)
    verdict = gate.evaluate(
        before={"verified_success": 0.60, "trap_success": 0.55, "shortcut_rate": 0.10},
        after={"verified_success": 0.68, "trap_success": 0.56, "shortcut_rate": 0.04},
    )
    assert verdict.promote


@pytest.mark.phase7
def test_a_gate_that_promotes_nothing_is_still_a_result() -> None:
    """If no round passes, publish that. It is a finding, not a failure."""
    pytest.skip("report-level assertion; enable when two rounds have run (Phase 7)")


@pytest.mark.phase7
def test_every_failure_class_maps_to_exactly_one_generator() -> None:
    """A class with no synthesis template turns the loop into a guess."""
    from veriloop.heal.classify import FailureClass
    from veriloop.heal.synthesise import TEMPLATES_BY_CLASS
    assert set(TEMPLATES_BY_CLASS) == set(FailureClass)


@pytest.mark.phase7
def test_healing_continues_from_the_current_adapter() -> None:
    """Never from base. Continuing from base silently discards prior learning."""
    pytest.skip("enable once veriloop/heal/loop.py lands (Phase 7)")
