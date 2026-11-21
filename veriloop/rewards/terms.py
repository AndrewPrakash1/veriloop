"""The six terms, each computed separately so an audit can attribute a score.

    Γ = 1  iff  no forbidden call occurred
           and  every executed call had schema-valid arguments
           and  the episode ended in a terminal action within the step budget

    M = required calls present with valid args / required calls
    S = valid_calls / total_calls              # DIAGNOSTIC ONLY — see rewards/state_verified.py
    F = 1 iff every final-state assertion holds
    E = clamp(1 - (steps - min_steps) / (budget - min_steps), 0, 1)
    D = identical_calls / total_calls          # identical on CANONICALISED args
    L = 1 if a repeated (tool, args) cycle is detected, else 0

Keep every term separately inspectable. When an arm scores lower than expected, the only
way to find out why is to see which term moved.

---

## A KNOWN LIMITATION, AND WHY IT IS ACCEPTED HERE

`must_not_call.when` must be evaluated **at the step index of the call**, not at the end —
a call can be forbidden in one state and legal in another. That is why every audit record
carries a step number.

This function receives only the FINAL snapshot, so it cannot rebuild the world at an
intermediate step. It therefore splits the facts by where they can be recovered from:

* **Log-derived** facts — `checks_recorded`, `all_checks_recorded` — ARE recomputed from
  the log's prefix, so they are exact at every step. These are the facts the live contracts
  actually use.
* **Snapshot-derived** facts — `account_status` — show the FINAL value only. A predicate
  asking "was this account open at step 3" would be answered with the end state, and would
  be WRONG.

Accepted because the alternative is worse: rebuilding the prefix needs the seed and a fresh
world, which would change this signature, and `veriloop/eval/oracle.py` already does an
independent full replay where exactness matters.

**If you write a `must_not_call` predicate over a snapshot-derived fact, the gate becomes
approximate.** The guard is `tests/test_reward_terms.py::test_the_gate_is_exact_for_log_derived_facts`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from veriloop.contracts.resolve import eval_predicate, resolve
from veriloop.contracts.schema import CallConstraint, Contract
from veriloop.env.audit_log import AuditLog

#: The checks a KYC contract treats as mandatory for `all_checks_recorded`. One definition,
#: here, rather than repeated inside each contract's predicate.
CANONICAL_CHECKS = ("identity", "sanctions", "pep")

_COUNT_OP = re.compile(r"^count\s*(>=|<=|==|>|<)\s*(\d+)$")


@dataclass(frozen=True)
class Terms:
    gamma: int
    M: float
    S: float
    F: float
    E: float
    D: float
    L: int
    missing_mandatory: tuple[str, ...] = ()     # for the failure classifier
    forbidden_seen: tuple[str, ...] = ()
    required_total: int = 0

    def as_row(self) -> dict:
        """Flat dict for logs and for the hackability table."""
        return {
            "gamma": self.gamma,
            "M": round(self.M, 4),
            "S": round(self.S, 4),
            "F": round(self.F, 4),
            "E": round(self.E, 4),
            "D": round(self.D, 4),
            "L": self.L,
            "required_total": self.required_total,
            "missing_mandatory": list(self.missing_mandatory),
            "forbidden_seen": list(self.forbidden_seen),
        }


# ---------------------------------------------------------------------------- facts -------
def _account_status(snapshot: Mapping[str, list[dict]], customer_id: Any) -> str | None:
    for row in snapshot.get("accounts", []):
        if row.get("customer_id") == customer_id:
            return row.get("status")
    return None


def _facts_from_snapshot(contract: Contract, snapshot: Mapping[str, list[dict]]) -> dict:
    """Facts as they stand at the END of the episode."""
    return {
        **dict(contract.pre_state),
        "is_read_only": True,
        "account_status": _account_status(snapshot, contract.pre_state.get("customer_id")),
    }


def _record_index_of(log: AuditLog, tool: str) -> int | None:
    """Index of the FIRST record for `tool`, for prefix-based predicate evaluation.

    Uses the record INDEX, not `record.step`, so a caller that numbers steps independently
    of records still gets the right prefix.
    """
    for i, r in enumerate(log.records):
        if r.tool == tool:
            return i
    return None


def _facts_before(log: AuditLog, snapshot: Mapping[str, list[dict]], contract: Contract, idx: int) -> dict:
    """Facts as they stood immediately BEFORE `log.records[idx]` ran.

    Log-derived facts are exact (see the module docstring); snapshot-derived ones are not.
    """
    recorded = [
        r.args.get("check_type")
        for r in log.records[:idx]
        if r.tool == "record_check" and r.schema_valid
    ]
    recorded = [c for c in recorded if c]
    return {
        **_facts_from_snapshot(contract, snapshot),
        "checks_recorded": recorded,
        "all_checks_recorded": set(CANONICAL_CHECKS) <= set(recorded),
    }


# ----------------------------------------------------------------------- assertions -------
def _row_matches(row: Mapping[str, Any], wanted: Mapping[str, Any]) -> bool:
    return all(row.get(k) == v for k, v in wanted.items())


def _assertion_holds(assertion, snapshot: Mapping[str, list[dict]], pre: Mapping[str, Any]) -> bool:
    rows = snapshot.get(assertion.table, [])

    if assertion.op == "==":
        # "there EXISTS a row in this table with these values". No assertions means "the
        # table must not be empty".
        if not assertion.assertions:
            return bool(rows)
        wanted = resolve(dict(assertion.assertions), pre)
        return any(_row_matches(r, wanted) for r in rows)

    match = _COUNT_OP.match(str(assertion.op))
    if not match:
        raise ValueError(
            f"unsupported final_state op {assertion.op!r} on table {assertion.table!r}. "
            f"Supported: '==' and 'count<op><n>'."
        )
    operator, n = match.group(1), int(match.group(2))
    wanted = resolve(dict(assertion.assertions), pre)
    count = sum(1 for r in rows if _row_matches(r, wanted))
    return {
        ">=": count >= n, "<=": count <= n, "==": count == n,
        ">": count > n, "<": count < n,
    }[operator]


# --------------------------------------------------------------------------- terms --------
def _constraint_satisfied(constraint: CallConstraint, pre: Mapping[str, Any],
                          log: AuditLog) -> bool:
    """Did the episode satisfy one mandatory constraint?

    AN EMPTY `args` MAPPING MEANS **TOOL PRESENCE ONLY** — "this tool must be called", not
    "called with no arguments". That distinction matters for a free-form tool such as
    `run_sql`: no model can guess a query string, so constraining one would make the
    contract unwinnable in practice while still validating and still letting the gold route
    score 1.0. The gold call carries real arguments through `gold_args`; the CONSTRAINT does
    not name them.

    Spelled out here, at the point of use, because a second implementation of this rule got
    it wrong on the first attempt — it demanded a call whose canonical arguments were
    literally `{}` and reported a phantom `M = 0.667`. The independent-re-derivation test in
    `tests/test_gen_splits.py` is what caught it.
    """
    try:
        wanted = resolve(dict(constraint.args), pre)
    except (KeyError, ValueError):
        return False
    if not wanted:
        return log.has_call(constraint.tool)
    return log.has_call(constraint.tool, wanted)


def _describe(constraint: CallConstraint) -> str:
    if constraint.args:
        return (f"{constraint.tool}("
                f"{json.dumps(dict(constraint.args), sort_keys=True, default=str)})")
    return f"{constraint.tool}()"


def compute_terms(
    contract: Contract,
    log: AuditLog,
    snapshot: Mapping[str, list[dict]],
    *,
    terminated: bool,
    steps: int,
) -> Terms:
    """Compute all six terms. Pure: no I/O, no model, no randomness, no clock.

    Load-bearing details:
      * `must_call[i].when` is evaluated against the FINAL snapshot. An unsatisfied `when`
        DROPS the constraint; it does not fail it.
      * `must_not_call[i].when` is evaluated against the state BEFORE the call ran.
      * `D` and `L` come from `AuditLog`, which already canonicalises arguments. Do not
        recompute them from raw args, or `reformatted_repeat` walks straight through.
      * `E` is computed here but the reward only pays it under the `M = S = F = 1` guard.
        The guard lives in `state_verified`, so this stays a faithful description.
    """
    pre = dict(contract.pre_state)
    records = log.records
    total_calls = len(records)

    # --- Γ: the hard gate --------------------------------------------------------
    forbidden_seen: list[str] = []
    for c in contract.must_not_call:
        idx = _record_index_of(log, c.tool)
        if idx is None:
            continue
        if c.when is None or eval_predicate(c.when, _facts_before(log, snapshot, contract, idx)):
            forbidden_seen.append(c.tool)

    invalid_calls = sum(1 for r in records if not r.schema_valid)
    gamma = 0 if (forbidden_seen or invalid_calls or not terminated) else 1

    # --- M: mandatory recall -----------------------------------------------------
    required: list[CallConstraint] = []
    for c in contract.must_call:
        if c.when is not None and not eval_predicate(c.when, _facts_from_snapshot(contract, snapshot)):
            continue                        # dropped, not failed
        required.append(c)

    missing: list[str] = [_describe(c) for c in required
                           if not _constraint_satisfied(c, pre, log)]

    M = 1.0 if not required else (len(required) - len(missing)) / len(required)

    # --- S: schema validity ------------------------------------------------------
    # NOTE: under Γ = 1 this is ALWAYS 1.0 — Γ already requires every executed call to be
    # schema-valid. It is kept as a diagnostic and is NOT a paid term; see state_verified.
    S = 0.0 if total_calls == 0 else sum(1 for r in records if r.schema_valid) / total_calls

    # --- F: final state ----------------------------------------------------------
    F = 1.0 if all(_assertion_holds(a, snapshot, pre) for a in contract.final_state) else 0.0

    # --- E: efficiency -----------------------------------------------------------
    mn, budget = contract.min_steps, contract.step_budget
    if budget <= mn:
        E = 1.0 if steps <= mn else 0.0
    else:
        E = min(1.0, max(0.0, 1.0 - (steps - mn) / (budget - mn)))

    # --- D, L: duplicates and loops ---------------------------------------------
    D = 0.0 if total_calls == 0 else log.identical_repeats() / total_calls
    L = 1 if log.has_cycle() else 0

    return Terms(
        gamma=gamma, M=M, S=S, F=F, E=E, D=D, L=L,
        missing_mandatory=tuple(missing), forbidden_seen=tuple(forbidden_seen),
        required_total=len(required),
    )


def multiplicity_ok(log: AuditLog, tool: str) -> bool:
    """Did `tool` fire exactly once, when the contract implies it should?

    A tool that fires twice for one logical step is redundant in a way `D` can miss, because
    the two calls differ in an argument — two `record_check`s of the same `check_type` with
    different `evidence_ref` are not "identical". The healing loop uses this to name that
    failure.
    """
    return sum(1 for r in log.records if r.tool == tool) == 1
