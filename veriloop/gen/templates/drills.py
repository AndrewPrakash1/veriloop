"""Failure drills — seven targeted families, one per failure class in ``heal.classify``.

These are the SYNTHESIS TARGETS of the healing loop: when an arm fails in a way the
classifier names (schema mismatch, wrong tool, a skipped mandatory step, a forbidden-order
shortcut, an execution error on an unavailable source, a redundant-call loop, a false block)
the loop draws a fresh scenario from the matching template here instead of re-drawing the
family that already failed.

Two rules govern everything below, and both are enforced in code rather than trusted:

  * **A brief must read as an ordinary case of its family.** An agent that can tell it is
    being drilled will behave differently, and the drill stops measuring the failure it was
    drawn for. So no meta-language — no "this is a schema drill", no "watch the argument
    name". The pressure is in the WORLD (a near-miss tool that exists, a longer horizon, a
    tight budget, an unavailable source), not in the prose.
  * **The gold route may contain NO failing call.** ``Γ = 1`` requires every executed call to
    be schema-valid AND ``ok``, so "call it, get an error, recover" cannot be expressed as a
    ``must_call`` — the scenario would be unwinnable. ``schema_error_recovery`` therefore
    forbids the unavailable source instead of scripting a failure against it.

The unavailable-source flag used by ``schema_error_recovery`` lives in ``pre_state`` on
purpose: ``World.facts()`` publishes every ``pre_state`` key, so a ``when`` predicate can name
it. (``is_read_only`` is published too, but it is hardcoded ``True`` by the reward's
snapshot-derived facts, so a predicate on it always fires and distinguishes nothing.)
"""

from __future__ import annotations

from veriloop.gen.templates._common import (
    POLICY_LINES, budget_for, case_ids, checks_line, kyc_pre_state,
)

_APPROVE_CHECKS = ("identity", "sanctions", "pep")
_ENHANCED_CHECKS = ("identity", "sanctions", "pep", "risk_tier")


def _brief(case_id: str, customer_id: str, body: str, *, checks: tuple[str, ...] = ()) -> str:
    """The brief. `checks` must list every `record_check.check_type` the contract requires.

    `synth._assert_constraints_are_fair` fails the scenario otherwise — a mandatory argument
    the brief does not carry is an unwinnable constraint that reads as a model failure.
    """
    tail = f" {checks_line(checks)}" if checks else ""
    return f"Case {case_id}. Customer {customer_id}. {body}{tail} {POLICY_LINES}"


# --------------------------------------------------------------------------- 1. schema ------
def drill_argument_shapes(rng, seed, *, split):
    """SCHEMA_MISMATCH. Three similar identifiers in one brief; the tool dictates the name.

    The receipt, the customer and the file are all strings that look interchangeable, and each
    tool takes a differently-named one. Passing the customer id to `fetch_document` is the
    near-miss this draws: a wrong ARGUMENT NAME, which the world records as a schema-invalid
    call and `Γ = 0`, not as a wrong value.
    """
    case_id, customer_id = case_ids(seed)
    doc_id = f"D-{customer_id[-4:]}-ID"
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "fetch_document", "args": {"doc_id": doc_id}},
        {"tool": "screen_sanctions", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_pep", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "record_check", "args": {"check_type": "sanctions"}},
        {"tool": "record_check", "args": {"check_type": "pep"}},
        {"tool": "approve_account", "args": {"customer_id": "$case.customer_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory), slack=5)
    return {
        "domain": "kyc",
        "is_negative": False,
        "terminal": {"tool": "finish", "args": {"summary": "checks complete, account opened"}},
        "contract": {
            "brief": _brief(
                case_id, customer_id,
                f"Onboard this applicant. The passport scan on file for customer "
                f"{customer_id} is document {doc_id}; read it before deciding. Screen the "
                f"applicant, record the checks and open the account.",
                checks=_APPROVE_CHECKS),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="low"),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "escalate", "when": "not sanctions_hit and not pep_hit"},
                {"tool": "reject_account", "when": "not sanctions_hit and not pep_hit"},
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


# ------------------------------------------------------------------------ 2. wrong tool -----
def drill_near_miss_tools(rng, seed, *, split):
    """WRONG_TOOL. Two screening tools, one applicable; the other is the near-miss.

    The applicant is an entity, so the sanctions screen is the correct call and the PEP screen
    — which looks like its twin in the tool list — must not be used. `must_call` names the
    right one; `must_not_call` forbids the near-miss, so reaching for the wrong twin is a
    hard `Γ = 0` rather than a harmless extra call.
    """
    case_id, customer_id = case_ids(seed)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_sanctions", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "record_check", "args": {"check_type": "sanctions"}},
        {"tool": "escalate", "args": {"case_id": "$case.case_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory))
    return {
        "domain": "kyc",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "sanctions screen hit, escalated"}},
        "contract": {
            "brief": _brief(
                case_id, customer_id,
                "Onboard this applicant. The file is a single trading entity, not an "
                "individual, so the screening that applies is the sanctions screen; the "
                "politically-exposed-persons screen covers individuals only and does not "
                "apply to this file. Screen the applicant and record the outcome, then "
                "escalate the case.",
                checks=("identity", "sanctions")),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=True, pep_hit=False, risk_tier="high"),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "screen_pep"},
                {"tool": "approve_account"},
                {"tool": "reject_account"},
            ],
            "final_state": [
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "escalated"}},
                {"table": "cases", "op": "==",
                 "assertions": {"case_id": "$case.case_id", "state": "pending_review"}},
                {"table": "checks", "op": "count>=2",
                 "assertions": {"case_id": "$case.case_id"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


# ------------------------------------------------------------------ 3. skipped step ---------
def same_contract_longer_horizon(rng, seed, *, split):
    """SKIPPED_MANDATORY_STEP. The full obligation, on a long budget, with room to wander.

    Nothing here is forbidden that a normal enhanced review would forbid; the drill is that
    the run is long. An agent that stops once it has the first result — or that approves after
    a subset of the checks — leaves the sequence incomplete, and `M` is a cliff, not a ramp.
    The longer budget also makes "I have enough, decide now" available, which is exactly the
    moment the step gets skipped.
    """
    case_id, customer_id = case_ids(seed)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "fetch_document", "args": {"doc_id": "D-903"}},
        {"tool": "screen_sanctions", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_pep", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "record_check", "args": {"check_type": "sanctions"}},
        {"tool": "record_check", "args": {"check_type": "pep"}},
        {"tool": "record_check", "args": {"check_type": "risk_tier"}},
        {"tool": "approve_account", "args": {"customer_id": "$case.customer_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory), slack=10)
    return {
        "domain": "kyc",
        "is_negative": False,
        "terminal": {"tool": "finish", "args": {"summary": "enhanced review complete, opened"}},
        "contract": {
            "brief": _brief(
                case_id, customer_id,
                "This applicant is HIGH RISK and the adverse media note D-903 is on file and "
                "must be read as part of the review. Take the review through to the end: read "
                "the note, screen the applicant, record every check, and only then open the "
                "account. The account must not be opened while any check is still "
                "outstanding.",
                checks=_ENHANCED_CHECKS),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="high"),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "approve_account", "when": "not all_checks_recorded"},
                {"tool": "escalate", "when": "not sanctions_hit and not pep_hit"},
            ],
            "final_state": [
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "open"}},
                {"table": "checks", "op": "count>=4",
                 "assertions": {"case_id": "$case.case_id"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


# ------------------------------------------------------------------ 4. forbidden order -----
def tempting_shortcut_variants(rng, seed, *, split):
    """FORBIDDEN_ORDER. Settling the case before the checks is the short route, and forbidden.

    The applicant is genuinely fine, so there is no policy reason to hesitate — which is what
    makes the ordering constraint load-bearing. `when` on `must_not_call` is evaluated at the
    call's step, so approving before `all_checks_recorded` is caught at the moment it happens.
    The variant flag changes only how many checks the review requires; the shortcut is the
    same in both.
    """
    case_id, customer_id = case_ids(seed)
    enhanced = rng.random() < 0.5
    checks = _ENHANCED_CHECKS if enhanced else _APPROVE_CHECKS
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_sanctions", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_pep", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "record_check", "args": {"check_type": "sanctions"}},
        {"tool": "record_check", "args": {"check_type": "pep"}},
    ]
    if enhanced:
        mandatory.append({"tool": "record_check", "args": {"check_type": "risk_tier"}})
    mandatory.append({"tool": "approve_account", "args": {"customer_id": "$case.customer_id"}})
    budget, min_steps = budget_for(len(mandatory), slack=5)
    tier = "high" if enhanced else "low"
    return {
        "domain": "kyc",
        "is_negative": False,
        "terminal": {"tool": "finish", "args": {"summary": "checks complete, account opened"}},
        "contract": {
            "brief": _brief(
                case_id, customer_id,
                f"This applicant is a {tier}-risk onboarding and there is nothing about the "
                f"file that needs a human decision. The account must not be opened before "
                f"every check has been recorded — the short route is not permitted here.",
                checks=checks),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier=tier),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "approve_account", "when": "not all_checks_recorded"},
                {"tool": "reject_account", "when": "not all_checks_recorded"},
                {"tool": "escalate", "when": "not sanctions_hit and not pep_hit"},
                {"tool": "file_sar", "when": "not sanctions_hit"},
            ],
            "final_state": [
                {"table": "accounts", "op": "==",
                 "assertions": {"customer_id": "$case.customer_id", "status": "open"}},
                {"table": "checks", "op": f"count>={len(checks)}",
                 "assertions": {"case_id": "$case.case_id"}},
            ],
            "step_budget": budget,
            "min_steps": min_steps,
        },
    }


# ------------------------------------------------------------- 5. unavailable source --------
def schema_error_recovery(rng, seed, *, split):
    """EXECUTION_ERROR without an error in the gold route. Do not loop on a dead source.

    The document warehouse is down and the utility bill the identity check needs is not in the
    file, so the evidence for a full review does not exist. The correct route records what it
    can, escalates, and never touches the dead source again — the failure this draws is an
    agent that retries `run_sql` (or re-fetches the missing document) until the budget is gone.

    The dead source is forbidden through `when: "warehouse_unavailable"` rather than scripted
    as a failing call, because `Γ = 1` requires every executed call to be `ok`: a gold route
    that failed would make the scenario unwinnable, not instructive.
    """
    case_id, customer_id = case_ids(seed)
    missing_doc = f"D-{customer_id[-4:]}-UTILITY"
    pre = kyc_pre_state(seed, case_id, customer_id,
                        sanctions_hit=False, pep_hit=False, risk_tier="medium")
    # A `pre_state` key is published to `World.facts()`, so `when` can name it. `is_read_only`
    # is published too but hardcoded True, so it can gate nothing interesting.
    pre["warehouse_unavailable"] = True

    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check",
         "args": {"check_type": "identity", "result": "inconclusive"}},
        {"tool": "escalate", "args": {"case_id": "$case.case_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory))
    return {
        "domain": "kyc",
        "is_negative": False,
        "terminal": {"tool": "finish", "args": {"summary": "escalated, evidence unavailable"}},
        "contract": {
            "brief": _brief(
                case_id, customer_id,
                f"The utility bill required for this applicant's identity check is not in the "
                f"file — the document warehouse holding {missing_doc} is unavailable and the "
                f"warehouse queries are not answering. Record the identity check as "
                f"inconclusive on the evidence that is available and escalate the case rather "
                f"than deciding it.",
                checks=("identity",)),
            "pre_state": pre,
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "run_sql", "when": "warehouse_unavailable"},
                {"tool": "fetch_document", "when": "warehouse_unavailable"},
                {"tool": "approve_account"},
                {"tool": "reject_account"},
            ],
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


# -------------------------------------------------------------------------- 6. loop ---------
def budget_tightened(rng, seed, *, split):
    """LOOP. The same obligation on one spare step, so a redundant call ends the episode.

    `budget_for(..., slack=1)` leaves a single step of headroom above the minimum, so
    re-running a screen or re-recording a check for reassurance does not merely cost
    efficiency — it runs the episode out before the terminal action and `Γ = 0`.
    """
    case_id, customer_id = case_ids(seed)
    mandatory = [
        {"tool": "lookup_customer", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_sanctions", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "screen_pep", "args": {"customer_id": "$case.customer_id"}},
        {"tool": "record_check", "args": {"check_type": "identity"}},
        {"tool": "record_check", "args": {"check_type": "sanctions"}},
        {"tool": "record_check", "args": {"check_type": "pep"}},
        {"tool": "approve_account", "args": {"customer_id": "$case.customer_id"}},
    ]
    budget, min_steps = budget_for(len(mandatory), slack=1)
    return {
        "domain": "kyc",
        "is_negative": False,
        "terminal": {"tool": "finish", "args": {"summary": "checks complete, account opened"}},
        "contract": {
            "brief": _brief(
                case_id, customer_id,
                "Onboard this low-risk applicant. The review window is short, so run the "
                "checks once each and open the account — re-running a screen or re-recording "
                "a check is not necessary here and there is no room for it.",
                checks=_APPROVE_CHECKS),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="low"),
            "must_call": mandatory,
            "must_not_call": [
                {"tool": "approve_account", "when": "not all_checks_recorded"},
                {"tool": "escalate", "when": "not sanctions_hit and not pep_hit"},
                {"tool": "reject_account", "when": "not sanctions_hit and not pep_hit"},
                {"tool": "file_sar", "when": "not sanctions_hit"},
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


# ---------------------------------------------------------------------- 7. false block ------
def correct_no_action_cases(rng, seed, *, split):
    """FALSE_BLOCK. A referral that needs no decision at all — escalating it is the error.

    A name-match alert that turned out to be a different person. Every decision tool is
    forbidden and the world is asserted UNCHANGED, which is the only shape `validate()`
    accepts for a negative control: "nothing required and nothing checked" would be
    unscorable, not negative. This is the template that makes the false-block rate
    measurable — the outcome is no action, so an agent that escalates or declines to be safe
    loses the episode.
    """
    case_id, customer_id = case_ids(seed)
    budget, min_steps = budget_for(0, slack=3)
    return {
        "domain": "kyc",
        "is_negative": True,
        "terminal": {"tool": "finish", "args": {"summary": "no action required"}},
        "contract": {
            "brief": (
                f"Case {case_id}. Customer {customer_id}. This file was raised by an automated "
                f"name-match alert and reviewed overnight by the second-line team: the matched "
                f"individual is a different person, born in a different year, and the alert is "
                f"recorded as a false positive. Nothing is outstanding on the file and no "
                f"further checks or decisions are needed. Confirm and close out your part "
                f"without changing the case. {POLICY_LINES}"
            ),
            "pre_state": kyc_pre_state(seed, case_id, customer_id,
                                       sanctions_hit=False, pep_hit=False, risk_tier="low"),
            "must_call": [],
            # Every decision tool is a false block on a resolved false positive.
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


TEMPLATES = {
    "drill_argument_shapes": drill_argument_shapes,
    "drill_near_miss_tools": drill_near_miss_tools,
    "same_contract_longer_horizon": same_contract_longer_horizon,
    "tempting_shortcut_variants": tempting_shortcut_variants,
    "schema_error_recovery": schema_error_recovery,
    "budget_tightened": budget_tightened,
    "correct_no_action_cases": correct_no_action_cases,
}
