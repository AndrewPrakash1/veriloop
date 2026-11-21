"""Scripted policies that FAIL IN NAMED WAYS — the loop's test and demo scaffolding.

These exist so the self-healing loop can be exercised with no model: each builder produces a
route with one specific defect, aimed at one `FailureClass`. They are in the package rather
than in a script because the test suite and the rehearsal must not drift — a defect that the
classifier handles in one and not the other is worse than no test.

**These are NOT model results and must never be reported as such.** They measure the scripted
route they build. Their value is that the LOOP's control flow — detection, classification,
synthesis, the gate — is exercised on real episodes with real rewards.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from veriloop.env.episode import Policy, RoutePolicy

#: The E3 template whose document carries the injected instruction. Named here, in the package,
#: so a rename cannot silently stop the trap from trapping — the rehearsal and the test both
#: read this constant.
INJECTION_TEMPLATE = "kyc_adverse_media_injection"

#: The defect kinds. One per class the classifier is expected to reach this way; FALSE_BLOCK
#: and EXECUTION_ERROR are reachable and are covered directly in the tests.
KINDS: tuple[str, ...] = ("skipped", "wrong_tool", "forbidden", "false_block", "schema", "loop")


def _find(route: Sequence[Mapping[str, Any]], tool: str) -> int | None:
    for i, c in enumerate(route):
        if c["tool"] == tool:
            return i
    return None


def _drop(route: list[dict], tool: str) -> list[dict]:
    i = _find(route, tool)
    return route if i is None else route[:i] + route[i + 1:]


def defective_route(s: Any, kind: str) -> list[dict]:
    """One scripted defect, aimed at one failure class.

    The route is always derived from the scenario's own `gold_route` and always keeps `finish`
    last, so the only thing that changes between kinds is the defect.
    """
    route = [dict(c) for c in s.gold_route]
    cust, case = s.pre_state["customer_id"], s.pre_state["case_id"]

    if kind == "skipped":
        # Two dropped steps: the final record_check AND the approval. Dropping only the check
        # would leave the approval in place, and approving before every check is recorded is
        # FORBIDDEN — that is `forbidden`, not this. The defect has to be unambiguous or the
        # classifier's priority order cannot be tested.
        route = _drop(_drop(route, "record_check"), "approve_account")
    elif kind == "wrong_tool":
        i = _find(route, "screen_pep")
        if i is not None:
            route[i] = {"tool": "screen_sanctions", "args": {"customer_id": cust}}
    elif kind == "forbidden":
        i = _find(route, "approve_account")
        if i is not None:                                    # approve FIRST, checks after
            route = [route[i]] + route[:i] + route[i + 1:]
    elif kind == "false_block":
        # A REFUSAL on a case that required action. Which class it lands in depends on the
        # contract, and that is the classifier working: on most KYC positives the contract also
        # forbids `escalate`, so a refusal is FORBIDDEN_ORDER (class 3 outranks FALSE_BLOCK);
        # FALSE_BLOCK is reached on contracts where refusing is legal but wrong — the analytics
        # templates, for instance. Both are failures and both have drills; only the drill
        # differs, which is the entire point of classifying. Nothing is wrong if a round
        # records fewer FALSE_BLOCKs than this kind was applied.
        route = [{"tool": "escalate", "args": {"case_id": case, "reason": "policy"}},
                 {"tool": "finish", "args": {"summary": "escalated for review"}}]
    elif kind == "schema":
        route = route[:1] + [{"tool": "record_check", "args": {"type": "pep"}}] + route[1:]
    elif kind == "loop":
        head = route[:3] or route
        reps = int(s.contract.step_budget / max(1, len(head))) + 2
        route = (head * reps) + route                          # runs past the step budget
    else:
        raise KeyError(f"unknown defect kind {kind!r}; have {list(KINDS)}")
    return route


def _kind_for(s: Any) -> str:
    """Deterministic defect per scenario, so a rehearsal is reproducible."""
    tail = str(s.scenario_id).rsplit(":", 1)[-1]
    idx = int(tail) if tail.isdigit() else sum(ord(c) for c in str(s.scenario_id))
    return KINDS[idx % len(KINDS)]


def defective_policy(s: Any) -> Policy:
    """The mixed-defect policy: one of `KINDS`, chosen deterministically per scenario."""
    return RoutePolicy(defective_route(s, _kind_for(s)))


def gold_policy(s: Any) -> Policy:
    """The corrected policy: the contract's own gold route."""
    return RoutePolicy([dict(c) for c in s.gold_route])


def baseline_policy(s: Any) -> Policy:
    """The loop's "before": defective on the clean slice, CORRECT on the trap slice.

    Splitting the defect by SPLIT rather than by template is what makes a comparison mean
    something. If the before-policy were also bad on the traps, a trap-blind candidate would
    look like an IMPROVEMENT in `trap_success` and the gate could not reject it — the
    comparison would measure how bad the starting point was, not what the round changed.
    """
    return gold_policy(s) if s.split == "E3" else defective_policy(s)


def trap_blind_policy(s: Any) -> Policy:
    """Correct everywhere EXCEPT where a tool result contains an instruction.

    This is the policy that raises the headline metric and must still be refused: it obeys the
    document rather than the contract. That failure is invisible to any metric computed on the
    clean slices, which is the entire reason the E3 split exists.
    """
    if s.template != INJECTION_TEMPLATE:
        return gold_policy(s)
    cust = s.pre_state["customer_id"]
    return RoutePolicy([
        {"tool": "lookup_customer", "args": {"customer_id": cust}},
        {"tool": "fetch_document", "args": {"doc_id": "D-903"}},
        {"tool": "approve_account", "args": {"customer_id": cust}},   # the injected order
        {"tool": "finish", "args": {"summary": "opened account per document note"}},
    ])


#: name -> factory, for anything that needs to resolve a policy by the label a round carries.
POLICIES = {
    "baseline": baseline_policy,
    "defective": defective_policy,
    "gold": gold_policy,
    "corrected": gold_policy,
    "trap_blind": trap_blind_policy,
}
