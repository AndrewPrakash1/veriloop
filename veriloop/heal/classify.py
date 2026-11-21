"""Seven NAMED failure classes. The class SELECTS the generator.

Named, not free-text: a label a model invents cannot choose a synthesis template, so
free-text classification turns a deterministic loop into a guess. Every class below maps
to exactly one entry in `synthesise.TEMPLATES_BY_CLASS`.

PRIORITY IS THE DESIGN, NOT AN IMPLEMENTATION DETAIL. The classes are checked in the order
below and the FIRST match wins, because a failure has one dominant cause and the causes are
not independent:

  1. `SCHEMA_MISMATCH`     a malformed call. First because it poisons everything after it:
     once a call is invalid, the episode's later calls are a response to an error the model
     would not have made with a valid schema, so their diagnostics are not about the task.
  2. `EXECUTION_ERROR`     a well-formed call the WORLD refused. Same argument one step down:
     a failed call changes what the rest of the route even means.
  3. `FORBIDDEN_ORDER`     it did something the contract forbids in that state. An explicit
     policy violation outranks every efficiency and completeness signal — this is the class
     the whole project is about.
  4. `LOOP`                a redundant cycle. Ranked here rather than last because a loop
     inflates the call count, which distorts the completeness signals below it.
  5. `FALSE_BLOCK`         it refused or escalated on a case that required action. Distinct
     from 6/7: nothing was attempted, so it is not a completeness failure, it is the
     mirror-image of a shortcut and is the reason negative controls exist.
  6. `WRONG_TOOL`          a near-miss tool was used instead of a required one. Detected from
     morphological overlap between a MISSING required tool and an extra call (see
     `_shares_root`) — a heuristic, and the reason `contract` is a parameter.
  7. `SKIPPED_MANDATORY_STEP` the catch-all: required work simply did not happen.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from veriloop.contracts.schema import Contract
from veriloop.rewards.terms import Terms


class FailureClass(str, Enum):
    SCHEMA_MISMATCH = "schema_mismatch"
    WRONG_TOOL = "wrong_tool"
    SKIPPED_MANDATORY_STEP = "skipped_mandatory_step"
    FORBIDDEN_ORDER = "forbidden_order"
    EXECUTION_ERROR = "execution_error"
    LOOP = "loop"
    FALSE_BLOCK = "false_block"


#: What the class means, in one line, for the round record and the write-up. Keeping this
#: next to the enum rather than in the docs is what stops a reviewer having to infer the
#: taxonomy from the code.
CLASS_MEANING: dict[FailureClass, str] = {
    FailureClass.SCHEMA_MISMATCH: "a malformed call: unknown tool, wrong arg name or type",
    FailureClass.EXECUTION_ERROR: "a well-formed call the world refused",
    FailureClass.FORBIDDEN_ORDER: "did something the contract forbids in that state",
    FailureClass.LOOP: "a redundant cycle of repeated calls",
    FailureClass.FALSE_BLOCK: "refused or escalated where the contract required action",
    FailureClass.WRONG_TOOL: "used a near-miss tool instead of the required one",
    FailureClass.SKIPPED_MANDATORY_STEP: "required work never happened",
}

#: Tools that end an episode or that a refusal route reaches for. Used only by the
#: FALSE_BLOCK test.
_TERMINAL = "finish"
_REFUSAL = frozenset({"escalate", "reject_account"})


def _calls(episode: Any) -> list[Any]:
    log = getattr(episode, "log", None)
    return list(getattr(log, "records", []) or [])


def _shares_root(a: str, b: str) -> bool:
    """Heuristic near-miss test: two tool names sharing a token.

    `screen_pep` / `screen_sanctions` share `screen`; `fetch_document` / `fetch_document`
    trivially match; `approve_account` / `reject_account` do not. Deliberately a heuristic and
    deliberately cruder than a fuzzy match: a false negative here downgrades the class to
    `SKIPPED_MANDATORY_STEP`, which still generates real drills, whereas a false POSITIVE would
    send the round to drill tool selection when the model's problem was completeness.
    """
    return bool(set(a.split("_")) & set(b.split("_")))


def classify(terms: Terms | None, episode: Any, *,
             contract: Contract | None = None) -> FailureClass:
    """Pick the single dominant class for one failed episode, by the priority above.

    `contract` is optional so the signature stays usable from `detect`, where the contract is
    always available and should always be passed. Without it the `WRONG_TOOL` branch cannot
    run, because deciding that a call was a near miss requires knowing which tools were
    required — and the fallback (`SKIPPED_MANDATORY_STEP`) is still a class with a real drill,
    so a caller that omits it degrades rather than breaks.
    """
    if terms is None:
        return FailureClass.SCHEMA_MISMATCH

    records = _calls(episode)

    # 1 — a malformed call outranks everything downstream of it.
    if terms.S < 1.0 or any(not r.schema_valid for r in records):
        return FailureClass.SCHEMA_MISMATCH

    # 2 — a valid call the world refused.
    if any(not r.ok for r in records):
        return FailureClass.EXECUTION_ERROR

    # 3 — an explicit policy violation.
    if terms.forbidden_seen:
        return FailureClass.FORBIDDEN_ORDER

    # 4 — a redundant cycle.
    if terms.L:
        return FailureClass.LOOP

    # 5 — refused or escalated where action was required. Distinct from 6/7: nothing was
    # attempted. Guarded on gamma so a refusal that was ALSO forbidden is classed as 3.
    if terms.required_total and terms.M == 0.0 and terms.gamma == 1:
        actions = [r.tool for r in records if r.tool != _TERMINAL]
        if actions and actions[-1] in _REFUSAL:
            return FailureClass.FALSE_BLOCK

    # 6 — a near-miss tool instead of a required one.
    if terms.missing_mandatory:
        if contract is not None:
            missing = _missing_tools(terms, contract)
            # A call that shares a root with a MISSING required tool and is not that tool is a
            # substitution. Note there is deliberately no `r.tool in required` skip: in a KYC
            # contract `screen_sanctions` and `screen_pep` are BOTH required, so skipping it
            # would mean the commonest near miss — running one screen and skipping its sibling
            # — could never be classed as a wrong tool. Forbidden tools are not skipped either,
            # but that is unreachable: class 3 is checked first.
            for r in records:
                if r.tool == _TERMINAL or r.tool in missing:
                    continue
                if any(_shares_root(r.tool, m) for m in missing):
                    return FailureClass.WRONG_TOOL
        # 7 — required work simply did not happen.
        return FailureClass.SKIPPED_MANDATORY_STEP

    # The calls were all made and all valid, yet an assertion failed: the right tool reached
    # the wrong target. That is the residual form of tool misuse, so it is classed as such.
    if terms.F < 1.0:
        return FailureClass.WRONG_TOOL

    # Ran out of budget without moving: the drill for this is a tighter budget, which is what
    # `budget_tightened` produces. Checked before the raise below because "used the whole
    # budget" is a real, common failure that no earlier branch claims.
    if not getattr(episode, "terminated", True):
        return FailureClass.LOOP

    raise ValueError(
        "classify() was given an episode that did not fail: gamma=1, M=1, S=1, F=1 and it "
        "terminated. There is no failure class to return, and returning one anyway would "
        "generate a drill for a failure that did not happen. `detect_failures` only passes "
        "flagged episodes."
    )


def _missing_tools(terms: Terms, contract: Contract) -> set[str]:
    """Which required TOOLS are absent.

    `terms.missing_mandatory` holds the human-readable descriptions the terms module produced
    with its own `_describe`, so the match is on the SAME function that generated them rather
    than on parsed text. Parsing would break the moment `_describe` changed its wording, and it
    would break silently — as a misclassified drill.
    """
    from veriloop.rewards.terms import _describe

    reported = set(terms.missing_mandatory)
    return {c.tool for c in contract.must_call if _describe(c) in reported}
