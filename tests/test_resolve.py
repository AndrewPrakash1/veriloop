"""Phase 2 — `veriloop/contracts/resolve.py`.

This file did not exist in the original spec; the suite went straight from "contracts are
data" to `validate()`, which meant `resolve` and `eval_predicate` would have been the first
code in the project that no test drove. `eval_predicate` decides whether a compliance step
is *mandatory*, so a bug in it silently re-scores every sanctions-hit case.

The last test in this file is the one that matters most: the predicate language must not be
able to execute anything.
"""

from __future__ import annotations

import pytest

from veriloop.contracts.resolve import (
    PredicateError,
    compile_predicate,
    eval_predicate,
    resolve,
)

PRE = {"customer_id": "a8821", "case_id": "C-1042", "seed": 20261121,
       "sanctions_hit": True, "pep_hit": False, "risk_tier": "high"}

FACTS = {"sanctions_hit": True, "pep_hit": False, "all_checks_recorded": False,
         "checks_recorded": ["identity"], "is_read_only": True,
         "account_status": "pending", "risk_tier": "high"}


# ------------------------------------------------------------------ resolve ------------
@pytest.mark.phase2
def test_a_lone_reference_is_substituted() -> None:
    assert resolve({"customer_id": "$case.customer_id"}, PRE) == {"customer_id": "a8821"}


@pytest.mark.phase2
def test_substitution_recurses_through_dicts_and_lists() -> None:
    nested = {"outer": [{"case": "$case.case_id"}, "$case.risk_tier"], "n": 3}
    assert resolve(nested, PRE) == {"outer": [{"case": "C-1042"}, "high"], "n": 3}


@pytest.mark.phase2
def test_non_strings_survive_with_their_types() -> None:
    """An int must stay an int — the tool layer validates on type."""
    out = resolve({"seed": 20261121, "flag": True, "none": None}, PRE)
    assert out == {"seed": 20261121, "flag": True, "none": None}
    assert isinstance(out["seed"], int) and isinstance(out["flag"], bool)


@pytest.mark.phase2
def test_a_reference_to_a_nonexistent_key_raises() -> None:
    """A placeholder would be worse than a crash: it scores as a WRONG ARGUMENT, so the
    bug surfaces as a reward problem instead of as the data problem it is."""
    with pytest.raises(KeyError):
        resolve({"customer_id": "$case.nope"}, PRE)


@pytest.mark.phase2
def test_resolved_values_keep_their_original_type() -> None:
    """`seed` is an int in pre_state — substituting it must not stringify it."""
    assert resolve("$case.seed", PRE) == 20261121
    assert isinstance(resolve("$case.seed", PRE), int)


@pytest.mark.phase2
def test_a_string_containing_case_dot_but_not_a_reference_raises() -> None:
    """Interpolation is not supported. Failing loudly here is the difference between a
    one-line contract fix and an afternoon debugging the reward."""
    with pytest.raises(ValueError):
        resolve("customer $case.customer_id approved", PRE)


@pytest.mark.phase2
def test_plain_strings_are_untouched() -> None:
    assert resolve("passport", PRE) == "passport"
    assert resolve("", PRE) == ""


# ------------------------------------------------------- eval_predicate: happy path -----
@pytest.mark.phase2
def test_a_bare_name_is_truthy_like_its_fact() -> None:
    assert eval_predicate("sanctions_hit", FACTS) is True
    assert eval_predicate("pep_hit", FACTS) is False


@pytest.mark.phase2
def test_negation() -> None:
    """`not is_read_only` is the live predicate in EXAMPLE_kyc_sanctions_hit.yaml."""
    assert eval_predicate("not is_read_only", FACTS) is False
    assert eval_predicate("not pep_hit", FACTS) is True


@pytest.mark.phase2
def test_boolean_combination() -> None:
    """`not sanctions_hit and not pep_hit` is the live predicate in the clean-pass contract."""
    assert eval_predicate("not sanctions_hit and not pep_hit", FACTS) is False
    assert eval_predicate("sanctions_hit or pep_hit", FACTS) is True


@pytest.mark.phase2
def test_membership_and_comparison() -> None:
    assert eval_predicate("'identity' in checks_recorded", FACTS) is True
    assert eval_predicate("'pep' in checks_recorded", FACTS) is False
    assert eval_predicate("account_status == 'pending'", FACTS) is True
    assert eval_predicate("len(checks_recorded) >= 1", FACTS) is True
    assert eval_predicate("len(checks_recorded) > 3", FACTS) is False


@pytest.mark.phase2
def test_chained_comparison() -> None:
    assert eval_predicate("0 < len(checks_recorded) <= 1", FACTS) is True


@pytest.mark.phase2
def test_returns_a_real_bool_not_a_truthy_value() -> None:
    """The reward multiplies by these, so a non-bool would leak a list into the arithmetic."""
    assert eval_predicate("checks_recorded", FACTS) is True
    assert isinstance(eval_predicate("len(checks_recorded)", FACTS), bool)


# ------------------------------------------------- eval_predicate: the security tests ----
@pytest.mark.phase2
def test_a_fact_the_world_does_not_expose_raises() -> None:
    """Loud, not silently False: a predicate naming a typo would otherwise make a mandatory
    step optional, and the contract would look satisfied."""
    with pytest.raises(KeyError):
        eval_predicate("sanctions_hlt", FACTS)          # note the typo


@pytest.mark.phase2
@pytest.mark.parametrize("expr,why", [
    ("__import__('os').system('rm -rf /')", "arbitrary call"),
    ("().__class__.__bases__", "attribute access"),
    ("checks_recorded[0]", "indexing"),
    ("[x for x in checks_recorded]", "comprehension"),
    ("lambda: True", "lambda"),
    ("open('/etc/passwd').read()", "file access"),
    ("f'{sanctions_hit}'", "f-string"),
    ("exec('x = 1')", "exec"),
    ("globals()", "globals"),
    ("sanctions_hit.__class__", "attribute access on a fact"),
])
def test_the_language_cannot_execute_anything(expr: str, why: str) -> None:
    """THE test in this file.

    Contracts are data, generated in thousands, and this language decides whether a
    compliance step is mandatory. If `eval_predicate` ever falls back on `eval()`, the
    injected payload would be authored by the very models under study. Every expression
    here must be rejected at PARSE time, before any evaluation happens.
    """
    with pytest.raises(PredicateError):
        compile_predicate(expr)
    with pytest.raises(PredicateError):
        eval_predicate(expr, FACTS)


@pytest.mark.phase2
def test_a_syntax_error_is_a_PredicateError_not_a_SyntaxError() -> None:
    """Callers catch one type. `validate()` reports a broken predicate as a contract
    problem, and a raw SyntaxError escaping would abort generation instead."""
    with pytest.raises(PredicateError):
        compile_predicate("sanctions_hit and")


@pytest.mark.phase2
def test_an_empty_predicate_is_rejected() -> None:
    with pytest.raises(PredicateError):
        compile_predicate("   ")


@pytest.mark.phase2
def test_keyword_arguments_are_rejected() -> None:
    """`all(x, default=False)` is not in the language, and allowing it invites smuggling."""
    with pytest.raises(PredicateError):
        compile_predicate("all(sanctions_hit, default=True)")
