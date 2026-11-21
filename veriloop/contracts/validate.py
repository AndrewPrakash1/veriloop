"""Contract validation — at GENERATION time, never at scoring time.

A contract that cannot be satisfied must never enter a split. If it does, it silently
becomes an unwinnable episode that poisons the rollout batch: every sample gets
`Gamma = 0`, the group advantage collapses to zero, and the run looks like "RL did
nothing" instead of like a data bug.

This is the single most expensive silent failure in the project, so validation is loud.
"""

from __future__ import annotations

from veriloop.contracts.schema import Contract


class ContractError(ValueError):
    """Raised when a contract is unsatisfiable, self-contradictory or malformed."""


def validate(contract: Contract, tool_names: set[str]) -> None:
    """Raise `ContractError` unless the contract is satisfiable.

    Must catch at least:
      * a tool in `must_call`/`must_not_call` that is not in `tool_names`
      * a `$case.*` reference with no matching key in `pre_state`
      * the same tool in `must_call` AND `must_not_call` with overlapping predicates
      * `min_steps > step_budget`, or a `must_call` longer than `step_budget`
      * `requires_terminal=False` while `must_not_call` forbids `finish`
      * an empty `must_call` with no `final_state` assertion — a scenario where nothing
        is required and nothing is checked is unscorable, not a negative control
      * a predicate that fails to parse
    """
    raise NotImplementedError("Phase 2")


def resolve(args: dict, pre_state: dict) -> dict:
    """Substitute `$case.*` references. DEPRECATED alias — use `resolve` from resolve.py."""
    raise NotImplementedError("Phase 2")
