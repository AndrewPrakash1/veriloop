"""Domain B — analytical investigation (Track 3 PS 3.6).

## WHY THIS DOMAIN IS SHAPED THE WAY IT IS

A naive "answer the business question" task CANNOT be verified by this reward. The reward
reads the world, never the model's text, so an analytical answer that lives in a reply is
invisible to it — and bolting on a text oracle would make the project's headline metric
depend on a judge, which is the thing the whole design avoids.

So the analytical task is shaped as an **investigation whose conclusion is a state change**:
query the transaction activity to find out whether the case crosses a reporting threshold,
then either file a suspicious activity report or leave the case alone. The SQL is how the
agent learns the fact; the *decision* is what the world records; and the final state verifies
the decision. A wrong or absent query produces the wrong decision, which fails the contract.

The threshold and the customer are both stated in the brief, and `run_sql`'s query is a
free-form argument, so its constraint is TOOL PRESENCE ONLY — no model can guess a query
string, and a constraint on one would make the contract unwinnable in practice.
"""

from __future__ import annotations

from veriloop.gen.templates._common import (
    POLICY_LINES, budget_for, case_ids, checks_line, kyc_pre_state,
)

#: The gold query. The CONSTRAINT on `run_sql` is tool-presence only (no model can guess a
#: query string), but the gold CALL must still be a valid one, so the template supplies it
#: through `gold_args`.
GOLD_QUERY = {"query": "SELECT SUM(amount) AS total FROM transactions"}

#: The reporting threshold band. Chosen so "under" and "over" are unambiguous: the generated
#: total is never within this of the threshold.
_MARGIN = 5_000.0


def _activity(rng, seed):
    """A deterministic transaction book for this seed, and the total to compare against."""
    amounts = [round(rng.uniform(2_000, 60_000), 2) for _ in range(4)]
    txns = [{"amount": a, "counterparty": f"cp-{(seed + i) % 997:03d}"}
            for i, a in enumerate(amounts)]
    return txns, round(sum(amounts), 2)


def _brief(case_id, customer_id, total, threshold, over: bool) -> str:
    direction = "meets or crosses" if over else "stays below"
    return (
        f"Case {case_id}. Customer {customer_id}. Investigate whether this customer's "
        f"transaction activity {direction} the reporting threshold of {threshold:.2f}. "
        f"Query the transaction records to establish the total before deciding. "
        + ("If the threshold is crossed, file a suspicious activity report. "
           "If it is not crossed, leave the case unchanged and finish."
           if not over else
           "A crossing must be reported as a suspicious activity report. ")
        + POLICY_LINES
    )


def analytics_over_threshold(rng, seed, *, split):
    """Activity crosses the threshold. The correct outcome is a report, not a decision."""
    case_id, customer_id = case_ids(seed)
    txns, total = _activity(rng, seed)
    threshold = round(total - _MARGIN, 2)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "run_sql", "args": {}},                    # tool presence only
        {"tool": "file_sar", "args": {"case_id": "$case.case_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory), slack=6)
    return {
        "domain": "analytics",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "threshold crossed, SAR filed"}},
        "gold_args": {"run_sql": GOLD_QUERY},
        "contract": {
            "brief": _brief(case_id, customer_id, total, threshold, over=True),
            "pre_state": kyc_pre_state(seed, case_id, customer_id, sanctions_hit=False,
                                       pep_hit=False, risk_tier="medium", transactions=txns),
            "must_call": mandatory,
            # Reporting rather than deciding: an account decision here would be wrong.
            "must_not_call": [{"tool": "approve_account"}, {"tool": "reject_account"}],
            "final_state": [
                {"table": "cases", "op": "==",
                 "assertions": {"case_id": "$case.case_id", "state": "sar_filed"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


def analytics_below_threshold_clear(rng, seed, *, split):
    """Activity is under the threshold. The correct outcome is NO action.

    This is a negative control that also exercises the novel domain: the agent must run the
    query, learn that nothing crosses the line, and then decline to act. An agent that files
    a report "to be safe" fails the final state.
    """
    case_id, customer_id = case_ids(seed)
    txns, total = _activity(rng, seed)
    threshold = round(total + _MARGIN, 2)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "run_sql", "args": {}},
    ]
    budget, min_steps = budget_for(len(mandatory), slack=6)
    return {
        "domain": "analytics",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "below threshold, no action"}},
        "gold_args": {"run_sql": GOLD_QUERY},
        "contract": {
            "brief": _brief(case_id, customer_id, total, threshold, over=False),
            "pre_state": kyc_pre_state(seed, case_id, customer_id, sanctions_hit=False,
                                       pep_hit=False, risk_tier="low", transactions=txns),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "file_sar"},
                {"tool": "approve_account"},
                {"tool": "reject_account"},
                {"tool": "escalate"},
            ],
            "final_state": [
                {"table": "cases", "op": "==",
                 "assertions": {"case_id": "$case.case_id", "state": "new"}},
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "pending"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


def analytics_below_threshold_approve(rng, seed, *, split):
    """Under the threshold AND fully screened, so the account should be opened.

    A positive control in the novel domain: without one, every E2 episode could be satisfied
    by doing nothing, and the novel-template split would reward exactly the degenerate policy
    the negative controls exist to expose.
    """
    case_id, customer_id = case_ids(seed)
    txns, total = _activity(rng, seed)
    threshold = round(total + _MARGIN, 2)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "run_sql", "args": {}},
        {"tool": "screen_sanctions", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_pep", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "record_check", "args": {"check_type": "sanctions"}},
        {"tool": "record_check", "args": {"check_type": "pep"}},
        {"tool": "approve_account", "args": {"customer_id": "$case.customer_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory), slack=6)
    return {
        "domain": "analytics",
        "is_negative": False,
        "terminal": {"tool": "finish", "args": {"summary": "clear, account opened"}},
        "gold_args": {"run_sql": GOLD_QUERY},
        "contract": {
            "brief": _brief(case_id, customer_id, total, threshold, over=False)
                     + " Once the review is clear, open the account."
                     + " " + checks_line(["identity", "sanctions", "pep"]),
            "pre_state": kyc_pre_state(seed, case_id, customer_id, sanctions_hit=False,
                                       pep_hit=False, risk_tier="low", transactions=txns),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "file_sar"},
                {"tool": "approve_account", "when": "not all_checks_recorded"},
            ],
            "final_state": [
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "open"}},
                {"table": "checks", "op": "count>=3",
                 "assertions": {"case_id": "$case.case_id"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


TEMPLATES = {
    "analytics_over_threshold": analytics_over_threshold,
    "analytics_below_threshold_clear": analytics_below_threshold_clear,
    "analytics_below_threshold_approve": analytics_below_threshold_approve,
}
