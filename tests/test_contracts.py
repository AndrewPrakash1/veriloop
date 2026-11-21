"""Phase 2 — contract validation, at generation time.

An unsatisfiable contract that reaches a split is the most expensive silent failure in the
project: every rollout gets `Gamma = 0`, the group advantage collapses to zero, and the run
reads as "RL did nothing" instead of "data bug". So these tests are strict.
"""

from __future__ import annotations

import copy

import pytest

from veriloop.contracts.schema import Contract


@pytest.mark.phase2
def test_example_contracts_validate(raw_contracts, tool_names) -> None:
    from veriloop.contracts.validate import validate
    for name, raw in raw_contracts.items():
        validate(Contract(**raw), tool_names)


@pytest.mark.phase2
@pytest.mark.parametrize("mutate,why", [
    (lambda c: c["must_call"].append({"tool": "nonexistent_tool", "args": {}}),
     "unknown tool in must_call"),
    (lambda c: c.update(step_budget=3, min_steps=9), "min_steps > step_budget"),
    (lambda c: c.update(final_state=[], must_call=[]), "nothing required and nothing checked"),
    (lambda c: c["must_call"].append({"tool": "approve_account", "args": {"customer_id": "$case.nope"}}),
     "unresolvable $case reference"),
])
def test_bad_contracts_are_rejected(raw_contracts, tool_names, mutate, why) -> None:
    from veriloop.contracts.validate import ContractError, validate
    raw = copy.deepcopy(raw_contracts["kyc_sanctions_hit"])
    mutate(raw)
    with pytest.raises(ContractError):
        validate(Contract(**raw), tool_names)


@pytest.mark.phase2
def test_a_constraint_must_be_necessary(raw_contracts) -> None:
    """Deleting each must_call entry must make the gold route FAIL the contract.

    If it does not, the constraint is decoration and `M` is measuring nothing. This is the
    test that keeps the mandatory-step metric honest, and it is the one most likely to
    catch a generator that is emitting padding.
    """
    pytest.skip("enable once gen/ can emit a gold route for this contract (Phase 3)")


@pytest.mark.phase2
def test_empty_must_call_is_a_valid_negative_control(raw_contracts, tool_names) -> None:
    """A negative control: nothing mandatory, but the world must be asserted unchanged."""
    from veriloop.contracts.validate import validate
    raw = copy.deepcopy(raw_contracts["kyc_clean_pass"])
    raw["must_call"] = []
    raw["id"] = "negative_control_probe"
    validate(Contract(**raw), tool_names)
