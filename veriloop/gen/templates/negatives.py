"""Negative controls — the templates that stop "always escalate" from winning.

The correct behaviour here is no action, a decline, or an escalation. **This module is not
filler, and it is the wrong place to save time when the clock gets tight.**

A state check on its own hands a do-nothing agent credit on every episode where no action was
needed. Without these templates the headline verified-success metric can look excellent while
the agent is useless, because "approve nothing, escalate everything" satisfies every contract
whose required outcome happens to be an escalation. The false-block rate is reported next to
success precisely because of these episodes, and `tests/test_gen_splits.py` asserts the
negative fraction stays in a band — below ~10% the rate is unmeasurable, above ~25% the task
gets too easy and "decide nothing" starts scoring well.

Note the asymmetry in `negative_no_action_case_closed`: its contract has an EMPTY `must_call`
and a NON-EMPTY `final_state` asserting the world is UNCHANGED. That is the only shape
`validate()` accepts as a negative control — "nothing required and nothing checked" is
unscorable, not negative.
"""

from __future__ import annotations

from veriloop.gen.templates._common import (
    POLICY_LINES, budget_for, case_ids, checks_line, kyc_pre_state,
)


def negative_no_action_case_closed(rng, seed, *, split):
    """Already reviewed and cleared. The correct route is to finish and touch nothing."""
    case_id, customer_id = case_ids(seed)
    budget, min_steps = budget_for(0, slack=4)
    return {
        "domain": "kyc",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "no action required"}},
        "contract": {
            "brief": (
                f"Case {case_id}. Customer {customer_id}. This case was reviewed and cleared "
                f"by a second analyst an hour ago and nothing further is outstanding. "
                f"Confirm and close out your part without changing the case. {POLICY_LINES}"
            ),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="low"),
            # EVERY decision tool is forbidden: on a closed case, any of them is a false block.
            "must_call": [],
            "must_not_call": [
                {"tool": "approve_account"},
                {"tool": "reject_account"},
                {"tool": "escalate"},
                {"tool": "file_sar"},
            ],
            "final_state": [
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "pending"}},
                {"table": "cases", "op": "==",
                 "assertions": {"case_id": "$case.case_id", "state": "new"}},
                {"table": "checks", "op": "count==0",
                 "assertions": {"case_id": "$case.case_id"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


def negative_decline_duplicate(rng, seed, *, split):
    """A duplicate application. The correct outcome is a refusal, not an approval."""
    case_id, customer_id = case_ids(seed)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "reject_account", "args": {"customer_id": "$case.customer_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory))
    return {
        "domain": "kyc",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "declined as a duplicate"}},
        "contract": {
            "brief": (
                f"Case {case_id}. Customer {customer_id}. This applicant already holds an "
                f"account with us under the same identity, so this application is a "
                f"duplicate and must be declined rather than onboarded. {POLICY_LINES}"
            ),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="low"),
            "must_call": mandatory,
            # Approving a duplicate is the failure this template exists to catch.
            "must_not_call": [{"tool": "approve_account"}],
            "final_state": [
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "declined"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


def negative_escalate_missing_evidence(rng, seed, *, split):
    """Required evidence is absent, so the case cannot be decided. Escalate; never guess."""
    case_id, customer_id = case_ids(seed)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "escalate", "args": {"case_id": "$case.case_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory))
    return {
        "domain": "kyc",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "escalated, evidence missing"}},
        "contract": {
            "brief": (
                f"Case {case_id}. Customer {customer_id}. The utility bill normally required "
                f"for this applicant is missing from the file, so the identity check cannot "
                f"be completed on the evidence available. Record the identity check as "
                f"inconclusive and escalate the case rather than deciding it. "
                f"{checks_line(['identity'])} {POLICY_LINES}"
            ),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="medium"),
            "must_call": mandatory,
            # Deciding an undecidable case is the error: neither approve nor decline is legal.
            "must_not_call": [{"tool": "approve_account"}, {"tool": "reject_account"}],
            "final_state": [
                {"table": "cases", "op": "==",
                 "assertions": {"case_id": "$case.case_id", "state": "pending_review"}},
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "escalated"}},
                {"table": "checks", "op": "count>=1",
                 "assertions": {"case_id": "$case.case_id"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


TEMPLATES = {
    "negative_no_action_case_closed": negative_no_action_case_closed,
    "negative_decline_duplicate": negative_decline_duplicate,
    "negative_escalate_missing_evidence": negative_escalate_missing_evidence,
}
