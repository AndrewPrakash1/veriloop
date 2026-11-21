"""Shared fixtures.

A test that errors because a phase is unimplemented is reported as a FAILURE, not skipped.
That is deliberate: a skipped spec is a spec nobody reads. Where a test genuinely cannot
run yet (it needs a GPU or a generated split), it is `pytest.skip` with the phase named.
"""

from __future__ import annotations

import pathlib

import pytest
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONTRACTS_DIR = ROOT / "contracts"
ATTACKS_YAML = ROOT / "audit" / "attacks.yaml"


@pytest.fixture(scope="session")
def repo_root() -> pathlib.Path:
    return ROOT


@pytest.fixture(scope="session")
def raw_contracts() -> dict:
    """The hand-written example contracts, as raw YAML, keyed by the name after EXAMPLE_."""
    out = {}
    for p in sorted(CONTRACTS_DIR.glob("EXAMPLE_*.yaml")):
        out[p.stem.replace("EXAMPLE_", "")] = yaml.safe_load(p.read_text())
    assert out, "no EXAMPLE_*.yaml contracts found"
    return out


@pytest.fixture(scope="session")
def scenario_contract(raw_contracts) -> dict:
    """The reference scenario every audit row is scored against."""
    return raw_contracts["kyc_sanctions_hit"]


@pytest.fixture(scope="session")
def tool_names() -> set[str]:
    from veriloop.env.tools import TOOL_SPECS
    assert TOOL_SPECS, "TOOL_SPECS is empty — Phase 1 has not landed"
    return set(TOOL_SPECS)


@pytest.fixture
def world_factory(scenario_contract):
    """Build ISOLATED worlds on demand.

    Isolation matters: two routes replayed into one world share a snapshot, so a comparison
    between them would be scored against contaminated state. Every test that compares two
    routes must build two worlds.
    """
    from veriloop.env.db import open_world
    counter = {"n": 0}

    def _make(contract: dict | None = None, episode_id: str | None = None):
        contract = contract or scenario_contract
        pre = contract["pre_state"]
        counter["n"] += 1
        return open_world(pre["seed"], pre, episode_id=episode_id or f"test-{counter['n']}")

    return _make


@pytest.fixture
def reference_world(world_factory):
    """A fresh world for the reference scenario. Never shared between tests."""
    return world_factory()


@pytest.fixture(scope="session")
def attacks():
    from veriloop.rewards.probes import load_attacks
    loaded = load_attacks(str(ATTACKS_YAML))
    assert loaded, "attack catalogue is empty"
    return loaded
