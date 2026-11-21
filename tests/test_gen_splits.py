"""Phase 3 — leakage and coverage. These are the tests that keep the numbers honest.

Leakage between train and eval turns every downstream result into fiction, and no amount of
statistics recovers from it.
"""

from __future__ import annotations

import json

import pytest


@pytest.mark.phase3
def test_no_scenario_in_two_splits(tmp_path) -> None:
    data = json.loads((tmp_path / "manifest.json").read_text()) if (tmp_path / "manifest.json").exists() else None
    pytest.skip("enable once make splits has run (Phase 3)")


@pytest.mark.phase3
def test_e3_never_reaches_a_training_file() -> None:
    """The trap set is a probe. The moment it enters a rollout batch it stops being one."""
    pytest.skip("enable once data/splits/ exists (Phase 3)")


@pytest.mark.phase3
def test_negative_controls_are_about_fifteen_percent() -> None:
    """Below ~10% the false-block rate is unmeasurable; above ~25% the task gets too easy."""
    pytest.skip("enable once data/splits/ exists (Phase 3)")


@pytest.mark.phase3
def test_every_contract_validates_before_it_is_written() -> None:
    pytest.skip("enable once gen/ lands (Phase 3)")


@pytest.mark.phase3
def test_gold_routes_satisfy_their_own_contracts() -> None:
    """A gold route that fails its own contract is an unwinnable episode in the batch."""
    pytest.skip("enable once gen/ lands (Phase 3)")


@pytest.mark.phase3
def test_a_sample_of_gold_is_re_derived_independently() -> None:
    """Generator and checker sharing a bug produces consistent-but-wrong gold that passes
    every automated test. The only defence is a second implementation on a sample."""
    pytest.skip("enable once gen/ lands (Phase 3)")
