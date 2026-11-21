"""The independent outcome oracle.

Re-executes the episode's effects from its recorded route against a fresh world built
from the same seed and pre-state, then asserts the contract's `final_state`. It does not
call `state_verified`; it deliberately re-derives the verdict so the two can disagree.

WHY A SECOND IMPLEMENTATION IS THE POINT

`state_verified` answers "what score does this episode deserve", from the contract, the log
and the snapshot, in one pass, with a step budget, an efficiency term and duplicate/loop
penalties folded in. The oracle answers a smaller and dumber question: **if I run those calls
again, does the world actually end up where the contract says it must?** It has no opinion
about budgets, speed, repetition or termination, and it never imports the reward.

The two are supposed to agree on a correct episode, and to be *able* to disagree otherwise. A
disagreement is a bug in ONE of them and is worth reporting. Silent agreement between an
implementation and itself is the failure mode this file exists to prevent: a generator, a
reward and an evaluator that share one bug produce consistently wrong numbers that pass every
self-consistent check in the suite.

THE THREE OUTCOMES, AND WHY `unscored` IS NOT A FAILURE

An assertion comes back `holds`, `fails` or `unscored`. `unscored` means the evidence the
assertion needs is not available: a table the contract names that the snapshot does not
publish, or a `$case.*` reference the pre-state does not define. The library's own rule is
that unavailable evidence is scored `unscored`, never as a failure — otherwise a truncated
log or a malformed contract gets reported as the model getting the task wrong, and the model
is blamed for our bug.

`verified` is therefore True only when there are no failures **and** nothing unscored. An
episode with missing evidence has not been verified; it has failed to be checked, which is a
different and more honest statement.

WHERE `pre_state` COMES FROM

The contract carries it (`Contract.pre_state`), so the signature stays the one the rest of
the project already calls. A caller holding a scenario passes `scenario.pre_state` into the
contract, and the two are the same object by construction.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from veriloop.contracts.resolve import resolve
from veriloop.contracts.schema import Contract
from veriloop.env.db import open_world
from veriloop.env.episode import RoutePolicy, run_episode

#: The threshold form of a `final_state` op. Kept as THIS module's own regex rather than
#: imported from the reward engine: sharing the parser would share its bugs, and a checker
#: that reuses the implementation it is checking is not an independent checker.
_COUNT_OP = re.compile(r"^\s*count\s*(>=|<=|==|>|<)\s*(\d+)\s*$")

HOLDS = "holds"
FAILS = "fails"
UNSCORED = "unscored"


def _rows_for(snapshot: Mapping[str, Any], table: str) -> list[dict] | None:
    """The rows for `table`, or None when the snapshot does not publish it.

    None is "no evidence", which is NOT the same as an empty list, which is "the table is
    there and empty". Collapsing the two is how a missing table becomes a silent failure.
    """
    return snapshot[table] if table in snapshot else None


def _row_matches(row: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in wanted.items())


def _evaluate(
    assertion: Any, snapshot: Mapping[str, Any], pre: Mapping[str, Any]
) -> tuple[str, dict]:
    """Decide one `final_state` assertion. Returns `(outcome, detail)`."""
    rows = _rows_for(snapshot, assertion.table)
    if rows is None:
        return UNSCORED, {
            "table": assertion.table,
            "reason": f"the snapshot publishes no table {assertion.table!r}",
        }

    try:
        wanted = resolve(dict(assertion.assertions), pre)
    except (KeyError, ValueError) as exc:
        # `resolve` is deliberately loud, and at GENERATION time it must be. At SCORING time
        # one malformed contract must not abort a 2400-episode batch: it has to be visible as
        # an unscored assertion, which is what this branch produces.
        return UNSCORED, {
            "table": assertion.table,
            "reason": f"unresolvable reference in the assertion: {exc}",
        }

    op = str(assertion.op or "==").strip()
    detail: dict[str, Any] = {"table": assertion.table, "op": op, "expected": wanted}

    if op == "==":
        # An `==` with no assertions means "this table must not be empty". Kept because it is
        # the contract language's meaning, not a private choice of this module.
        if not wanted:
            return (HOLDS if rows else FAILS), detail
        return (HOLDS if any(_row_matches(r, wanted) for r in rows) else FAILS), detail

    match = _COUNT_OP.match(op)
    if match is None:
        raise ValueError(
            f"unsupported final_state op {op!r} on table {assertion.table!r}. This oracle "
            f"reads '==' and 'count<op><n>'. An op it cannot parse is a change to the "
            f"contract language, and it must fail loudly rather than be guessed at."
        )

    operator, n = match.group(1), int(match.group(2))
    observed = sum(1 for r in rows if _row_matches(r, wanted))
    detail["observed_count"] = observed
    holds = {
        ">=": observed >= n,
        "<=": observed <= n,
        "==": observed == n,
        ">": observed > n,
        "<": observed < n,
    }[operator]
    return (HOLDS if holds else FAILS), detail


def replay_and_assert(
    contract: Contract,
    route: Sequence[Mapping[str, Any]],
    seed: int,
) -> Mapping[str, Any]:
    """Replay `route`, then evaluate every `final_state` assertion independently.

    Returns {"verified": bool, "failures": [...], "unscored": [...]}, plus the replay facts
    the metrics layer needs (`steps`, `terminated`, `calls`).

    The replay is a REAL episode: the calls go through `veriloop.env.tools.call_tool`, so the
    world's own argument validation, read-only guard and append-only log all apply. A route
    that is malformed produces a failed `ToolResult`, not an exception — malformed calls are
    DATA in this project, and the oracle has to be able to score them.
    """
    pre = dict(contract.pre_state)
    episode_id = f"oracle-{contract.id}"

    world = open_world(seed, pre, episode_id=episode_id)
    try:
        result = run_episode(
            world, RoutePolicy(list(route)), contract.brief, contract.step_budget
        )
        snapshot = result.snapshot
        steps = result.steps
        terminated = bool(result.terminated)
        calls = len(result.log.records)
        try:
            result.log.assert_append_only()
            append_only = True
        except RuntimeError:
            append_only = False
    finally:
        world.close()

    failures: list[dict] = []
    unscored: list[dict] = []
    for index, assertion in enumerate(contract.final_state):
        outcome, detail = _evaluate(assertion, snapshot, pre)
        detail["index"] = index
        if outcome == FAILS:
            failures.append(detail)
        elif outcome == UNSCORED:
            unscored.append(detail)

    return {
        # Missing evidence is not verification. Both lists must be empty.
        "verified": not failures and not unscored,
        "failures": failures,
        "unscored": unscored,
        "assertions": len(contract.final_state),
        "steps": steps,
        "terminated": terminated,
        "calls": calls,
        "log_append_only": append_only,
        "episode_id": episode_id,
    }
