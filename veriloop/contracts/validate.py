"""Contract validation — at GENERATION time, never at scoring time.

A contract that cannot be satisfied must never enter a split. If it does, it silently
becomes an unwinnable episode that poisons the rollout batch: every sample gets `Γ = 0`,
the group advantage collapses to zero, and three hours of GPU reads as *"RL did nothing"*
instead of as *"data bug"*. This is the single most expensive silent failure in the project,
so validation is loud.

Two properties of the design worth keeping:

* **All problems are reported at once.** Raising on the first defect makes a generator fix
  cycle into ten runs instead of one.
* **Nothing here touches the database or the world.** `veriloop/contracts/` may only import
  from itself — `tests/test_imports.py` enforces that — which is why the `final_state`
  table names are *not* checked here. A table that does not exist is caught by the oracle at
  evaluation time, not at generation time. That is a real gap, accepted deliberately to keep
  the layer clean.
"""

from __future__ import annotations

from veriloop.contracts.resolve import PredicateError, compile_predicate, resolve
from veriloop.contracts.schema import CallConstraint, Contract


class ContractError(ValueError):
    """Raised when a contract is unsatisfiable, self-contradictory or malformed."""


def _key(c: CallConstraint) -> str:
    """Identity of a mandatory step, for duplicate detection."""
    import json

    return f"{c.tool}({json.dumps(dict(c.args), sort_keys=True, default=str)})"


def validate(contract: Contract, tool_names: set[str]) -> None:
    """Raise `ContractError` unless the contract is satisfiable.

    Checks, in the order they are reported:

    1. `min_steps > step_budget`, or a non-positive budget
    2. more mandatory calls than there are steps to make them in
    3. an unscorable run: nothing required AND nothing asserted
    4. a tool that is not in `tool_names`
    5. a `$case.*` reference with no matching key in `pre_state`
    6. a `when` predicate that is outside the language
    7. the same tool forbidden and required, with the same predicate
    8. a duplicate mandatory step — a generator bug that would double-count in `M`
    9. `requires_terminal=False` while `finish` is forbidden
    """
    problems: list[str] = []
    pre = dict(contract.pre_state)

    # 1 — budget sanity
    if contract.step_budget <= 0:
        problems.append(f"step_budget must be positive, got {contract.step_budget}")
    if contract.min_steps > contract.step_budget:
        problems.append(
            f"min_steps ({contract.min_steps}) exceeds step_budget ({contract.step_budget})"
        )
    if contract.min_steps < 0:
        problems.append(f"min_steps must not be negative, got {contract.min_steps}")

    # 2 — the required route has to fit
    if len(contract.must_call) > contract.step_budget:
        problems.append(
            f"must_call has {len(contract.must_call)} steps but step_budget is only "
            f"{contract.step_budget} — the contract cannot be satisfied"
        )

    # 3 — unscorable: nothing required and nothing asserted is not a negative control, it is
    #     a scenario with no content. A negative control has an EMPTY must_call and a
    #     NON-EMPTY final_state asserting the world is unchanged.
    if not contract.must_call and not contract.final_state:
        problems.append(
            "nothing is required and nothing is asserted — unscorable. A negative control "
            "must keep a final_state asserting the world is unchanged."
        )

    # 4/5/6 — per constraint
    for group, constraints in (("must_call", contract.must_call),
                               ("must_not_call", contract.must_not_call)):
        for c in constraints:
            if c.tool not in tool_names:
                problems.append(f"{group}: unknown tool {c.tool!r}")
            try:
                resolve(dict(c.args), pre)
            except KeyError as exc:
                problems.append(f"{group}: {exc.args[0]}")
            except ValueError as exc:
                problems.append(f"{group}: {exc}")
            if c.when is not None:
                try:
                    compile_predicate(c.when)
                except PredicateError as exc:
                    problems.append(f"{group}: {c.tool!r} — {exc}")

    # 7 — forbidden and required with the same predicate is a contradiction.
    #     CONSERVATIVE ON PURPOSE: comparing two predicates for logical overlap needs a
    #     solver we do not have, so only an exact-string collision is flagged. A contract
    #     that forbids and requires the same tool under *different* predicates is legal
    #     (e.g. escalate if sanctions hit / escalate forbidden if they did not).
    forbidden = {c.tool: c.when for c in contract.must_not_call}
    for c in contract.must_call:
        if c.tool in forbidden and forbidden[c.tool] == c.when:
            problems.append(
                f"{c.tool!r} is both required and forbidden under the same predicate "
                f"({c.when!r}) — contradictory"
            )

    # 8 — duplicate mandatory steps. `M` counts required calls that are present, so a
    #     duplicate entry makes one call satisfy two requirements and inflates recall.
    seen: dict[str, int] = {}
    for c in contract.must_call:
        k = _key(c)
        seen[k] = seen.get(k, 0) + 1
    for k, n in seen.items():
        if n > 1:
            problems.append(f"must_call lists {k} {n} times — a duplicate inflates M")

    # 9 — an episode that must terminate but is forbidden from terminating
    if not contract.requires_terminal:
        for c in contract.must_not_call:
            if c.tool == "finish":
                problems.append(
                    "requires_terminal=False while must_not_call forbids 'finish' — the "
                    "episode could never end legally"
                )

    if problems:
        raise ContractError(
            f"contract {contract.id!r} is not satisfiable:\n  - " + "\n  - ".join(problems)
        )
