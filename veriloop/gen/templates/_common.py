"""Helpers shared by the template modules.

Kept out of `templates/__init__.py` on purpose: the package `__init__` imports the template
modules, so a template importing from the package would be a cycle.
"""

from __future__ import annotations

from typing import Any, Iterable


def case_ids(seed: int) -> tuple[str, str]:
    """Deterministic, DISTINCT case and customer ids for one seed.

    Distinct matters: two scenarios whose quantities coincide make argument-constraint
    matching ambiguous, and a constraint that matches the wrong thing is worse than no
    constraint at all.

    Read off `_common.py`:
    """
    case_id = f"C-{(seed * 7919) % 9000 + 1000}"
    customer_id = f"c{(seed * 104729) % 90000 + 10000:05d}"
    return case_id, customer_id


def kyc_pre_state(seed: int, case_id: str, customer_id: str, *, sanctions_hit: bool,
                  pep_hit: bool, risk_tier: str, document_bodies: dict | None = None,
                  transactions: list | None = None) -> dict[str, Any]:
    """The scenario's world parameters. The ONLY source of world variation."""
    pre: dict[str, Any] = {
        "seed": seed,
        "case_id": case_id,
        "customer_id": customer_id,
        "sanctions_hit": sanctions_hit,
        "pep_hit": pep_hit,
        "risk_tier": risk_tier,
    }
    if document_bodies:
        pre["document_bodies"] = dict(document_bodies)
    if transactions:
        pre["transactions"] = [dict(t) for t in transactions]
    return pre


def budget_for(n_mandatory: int, slack: int = 4) -> tuple[int, int]:
    """`(step_budget, min_steps)` for a contract with `n_mandatory` unconditional steps.

    `min_steps` is the mandatory calls PLUS ONE for the terminal action. Getting this wrong
    is how the efficiency term becomes either a free bonus or unearnable, and there is no
    generated path that skips it.
    """
    min_steps = n_mandatory + 1
    return min_steps + slack, min_steps


#: Policy boilerplate. Deliberately does NOT name the checks — each template appends
#: `checks_line(...)` with the exact check types its contract requires, so the brief contains
#: the literal tokens the model must pass as `record_check.check_type`.
POLICY_LINES = (
    "Follow the standard KYC policy before any account decision. "
    "Flag anything that requires escalation."
)


def checks_line(check_types: Iterable[str]) -> str:
    """The sentence that makes `record_check.check_type` derivable from the brief.

    This is what `synth._assert_constraints_are_fair` measures. Naming the checks verbatim is
    also how a real policy notice reads, so the fairness rule does not distort the fictional
    world to satisfy the test.
    """
    names = list(check_types)
    if len(names) == 1:
        listed = names[0]
    else:
        listed = ", ".join(names[:-1]) + " and " + names[-1]
    return f"Record the {listed} checks as you perform them."
