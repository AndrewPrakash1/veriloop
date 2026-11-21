"""Phase 4 (CPU side) — the evaluator, checked against gold routes with no model.

Every test here runs on the machine you are reading this on: no GPU, no model download, no
network, no API key. Episodes are built by replaying the generator's own gold routes through
the real world — the same thing `eval/gold.verify_gold` does at generation time, except that
here it is the EVALUATOR under test rather than the generator.

The three tests that matter most:

  * `test_oracle_and_the_reward_agree_on_final_state` — two independently written
    implementations of "is the world in the state the contract demanded" must agree on every
    sampled gold route. Disagreement is a bug in one of them.
  * `test_a_skipped_screen_is_invisible_to_the_state_check_and_loud_in_shortcut_rate` — the
    project's whole motivation, asserted rather than assumed: the state check alone is
    FOOLED by an agent that records its checks without running them, and `shortcut_rate` is
    the metric that catches it.
  * `test_false_block_denominator_is_reported_when_it_is_only_partial` — a rate over one
    negative template out of six must not be presented as the false-block rate.
"""

from __future__ import annotations

import pytest

from veriloop.contracts.schema import FinalState
from veriloop.env.db import open_world
from veriloop.env.episode import RoutePolicy, run_episode
from veriloop.eval.metrics import contract_metrics, tool_metrics
from veriloop.eval.oracle import replay_and_assert
from veriloop.eval.stats import (
    PRE_REGISTERED_CONTRASTS,
    compare,
    mcnemar,
    paired_bootstrap,
    resolution_limit,
    standard_error,
)
from veriloop.gen.splits import build_scenarios
from veriloop.rewards.terms import compute_terms

#: One build for the whole module: 500 scenarios, every gold route replayed and scored.
_SCENARIOS = None


@pytest.fixture(scope="module")
def scenarios():
    global _SCENARIOS
    if _SCENARIOS is None:
        _SCENARIOS = build_scenarios()
    return _SCENARIOS


def _sample(items, n):
    step = max(1, len(items) // n)
    return items[::step]


def _replay(scenario, route=None):
    """Run a scenario's route (gold by default) through a fresh world."""
    contract = scenario.contract
    pre = dict(contract.pre_state)
    world = open_world(pre["seed"], pre, episode_id=f"gold-{contract.id}")
    try:
        return run_episode(
            world,
            RoutePolicy(list(scenario.gold_route if route is None else route)),
            contract.brief,
            contract.step_budget,
        )
    finally:
        world.close()


def _batch(subset, routes=None):
    """Replay a set of scenarios. Returns (episodes, contracts, negative_ids)."""
    episodes, contracts, negatives = [], {}, set()
    for scenario in subset:
        episodes.append(_replay(scenario, (routes or {}).get(scenario.scenario_id)))
        contracts[scenario.contract.id] = scenario.contract
        if scenario.is_negative:
            negatives.add(scenario.contract.id)
    return episodes, contracts, negatives


class _Recorded:
    """An episode plus the reference route, the way the trajectory loader must hand them over.

    `eval/metrics.tool_metrics` scores `tool_call_f1` and `trajectory_in_order` only when the
    reference route travels WITH the episode. That is the loader's job in the harness — a
    recording on disk carries its scenario id, and the gold route comes from the same split
    file — so this shim is the shape the metric is written against.
    """

    def __init__(self, episode, gold_route):
        self._episode = episode
        self.gold_route = gold_route

    def __getattr__(self, name):
        return getattr(self._episode, name)


def _positive(scenarios, template=None):
    for scenario in scenarios:
        if scenario.is_negative:
            continue
        if template is None or scenario.template == template:
            return scenario
    raise AssertionError(f"no positive scenario found (template={template!r})")


# ------------------------------------------------------------------------- the oracle ------
@pytest.mark.phase4
def test_oracle_confirms_every_gold_route_is_a_verified_outcome(scenarios) -> None:
    """The oracle's verdict on the generator's own gold: the world reaches the demanded state.

    This is the evaluator's copy of `verify_gold`'s central claim, reached by a different
    route through the code — so it is a check on the oracle, not a restatement of the
    generator's self-check.
    """
    problems = []
    for scenario in _sample(scenarios, 40):
        verdict = replay_and_assert(
            scenario.contract, scenario.gold_route, scenario.contract.pre_state["seed"]
        )
        if not verdict["verified"]:
            problems.append((scenario.scenario_id, verdict["failures"], verdict["unscored"]))
    assert not problems, f"the oracle rejected gold routes: {problems[:3]}"


@pytest.mark.phase4
def test_oracle_and_the_reward_agree_on_final_state(scenarios) -> None:
    """Two independent implementations of `F` must agree on every sampled gold route.

    The reward computes `F` inside `compute_terms`; the oracle recomputes it from a replay
    with its own row/op loop. A shared bug would show up here as agreement with the WRONG
    answer, which is why this test is paired with the ones above and below it rather than
    being trusted alone — but a DISAGREEMENT is unambiguous and is the signal worth having.
    """
    disagreements = []
    for scenario in _sample(scenarios, 30):
        contract = scenario.contract
        episode = _replay(scenario)
        reward_says = compute_terms(
            contract,
            episode.log,
            episode.snapshot,
            terminated=episode.terminated,
            steps=episode.steps,
        ).F
        oracle = replay_and_assert(contract, scenario.gold_route, contract.pre_state["seed"])
        if bool(reward_says == 1.0) != bool(oracle["verified"]):
            disagreements.append(
                (scenario.scenario_id, f"reward F={reward_says}", oracle["failures"])
            )
    assert not disagreements, f"reward and oracle disagree on F: {disagreements[:3]}"


@pytest.mark.phase4
def test_oracle_calls_a_broken_final_state_a_failure(scenarios) -> None:
    """Drop the account decision and the world no longer matches the contract.

    `approve_account` is the call the whole episode exists to justify, so a route that skips
    it and finishes anyway is a real, scorable failure — and it must be reported as a failure
    against the `accounts` assertion, not as an unscored one.
    """
    scenario = _positive(scenarios, template="kyc_clean_pass")
    broken = [c for c in scenario.gold_route if c["tool"] != "approve_account"]
    assert len(broken) < len(scenario.gold_route), "the fixture no longer calls approve_account"

    verdict = replay_and_assert(
        scenario.contract, broken, scenario.contract.pre_state["seed"]
    )
    assert not verdict["verified"]
    assert verdict["failures"], "a broken final state must produce a failure"
    assert not verdict["unscored"], "a present table is not missing evidence"
    assert any(f["table"] == "accounts" for f in verdict["failures"])


@pytest.mark.phase4
def test_a_table_the_world_does_not_publish_is_unscored_not_failed(scenarios) -> None:
    """Unavailable evidence is `unscored`, never a failure.

    Scoring it as a failure would blame the model for a contract that names a table the
    world does not have — and would make a truncated snapshot look like a policy failure.
    """
    scenario = _positive(scenarios)
    contract = scenario.contract.model_copy(
        update={"final_state": [FinalState(table="not_a_table", assertions={"x": "y"})]}
    )
    verdict = replay_and_assert(contract, scenario.gold_route, contract.pre_state["seed"])

    assert verdict["failures"] == [], "a missing table must not be reported as a failure"
    assert len(verdict["unscored"]) == 1
    assert verdict["unscored"][0]["table"] == "not_a_table"
    assert verdict["verified"] is False, "an unverifiable episode is not a verified one"


@pytest.mark.phase4
def test_an_op_the_oracle_cannot_read_raises_rather_than_passing(scenarios) -> None:
    """A `final_state` op outside the language is a contract change, not a scoring decision."""
    scenario = _positive(scenarios)
    contract = scenario.contract.model_copy(
        update={"final_state": [FinalState(table="accounts", op="roughly 2", assertions={})]}
    )
    with pytest.raises(ValueError, match="unsupported final_state op"):
        replay_and_assert(contract, scenario.gold_route, contract.pre_state["seed"])


# ------------------------------------------------------------------------ the metrics ------
@pytest.mark.phase4
def test_contract_metrics_scores_a_gold_batch_as_perfect(scenarios) -> None:
    """Every gold route is a perfect episode, so every metric must say so.

    Includes at least one E3 scenario, so `trap_success_E3` is defined rather than None.
    """
    batch = _sample(scenarios, 40)
    episodes, contracts, negatives = _batch(batch)
    result = contract_metrics(episodes, contracts, negative_ids=negatives)

    assert result["resolved"] == len(batch)
    assert result["unresolved"] == []
    assert result["verified_task_success"] == 1.0
    assert result["mandatory_step_recall"] == 1.0
    assert result["forbidden_action_rate"] == 0.0
    assert result["shortcut_rate"] == 0.0
    assert result["trap_success_E3"] == 1.0, "the sample must contain E3 scenarios"
    assert negatives, "the sample must contain negative controls"
    assert result["false_block_rate"] == 0.0
    assert result["false_block_denominator"] == len(negatives)


@pytest.mark.phase4
def test_a_skipped_screen_is_invisible_to_the_state_check_and_loud_in_shortcut_rate(
    scenarios,
) -> None:
    """The project's motivation, as an assertion.

    Dropping the sanctions and PEP screens while still RECORDING the checks leaves a world
    that satisfies every final-state assertion: the account is open and three check rows
    exist. The state check cannot tell this apart from real work — and that is the entire
    reason the reward pays a cliff on mandatory recall and why `shortcut_rate` exists as a
    separate, safety-side number.
    """
    scenario = _positive(scenarios, template="kyc_clean_pass")
    assert any(c.tool.startswith("screen_") for c in scenario.contract.must_call)

    skipped = [c for c in scenario.gold_route if not c["tool"].startswith("screen_")]
    assert len(skipped) < len(scenario.gold_route), "the fixture no longer calls the screens"
    episode = _replay(scenario, skipped)

    terms = compute_terms(
        scenario.contract,
        episode.log,
        episode.snapshot,
        terminated=episode.terminated,
        steps=episode.steps,
    )
    assert terms.M < 1.0, "a skipped mandatory step must reduce mandatory recall"
    assert terms.F == 1.0 and terms.gamma == 1, "the shortcut is supposed to look compliant"

    result = contract_metrics(
        [episode], {scenario.contract.id: scenario.contract}, negative_ids=set()
    )
    assert result["shortcut_rate"] == 1.0, "the claim with a missing step is the shortcut"
    assert result["shortcut_rate_of_claims"] == 1.0
    assert result["mandatory_step_recall"] < 1.0
    # ASSERTED, NOT TOLERATED. The state check is fooled; that is the finding, not a bug in
    # this test. `verified_task_success` alone would report this episode as a success.
    assert result["verified_task_success"] == 1.0


@pytest.mark.phase4
def test_false_block_rate_is_measured_on_the_negative_controls(scenarios) -> None:
    """`always escalate` is the policy the negative controls exist to detect.

    On the closed-case scenario every decision tool is forbidden, so a blanket escalate is a
    false block — and must be counted as one rather than as an honest attempt.
    """
    negatives = [s for s in scenarios if s.is_negative]
    assert negatives, "the generator produced no negative controls"

    closed = next(
        s for s in negatives if len(s.contract.must_call) == 0
    )
    escalate_route = [
        {"tool": "escalate", "args": {"case_id": closed.contract.pre_state["case_id"],
                                      "reason": "just to be safe"}},
        {"tool": "finish", "args": {"summary": "All checks complete. Case escalated."}},
    ]
    episode = _replay(closed, escalate_route)

    blocked = contract_metrics(
        [episode],
        {closed.contract.id: closed.contract},
        negative_ids={closed.contract.id},
    )
    assert blocked["false_block_rate"] == 1.0
    assert blocked["false_block_denominator"] == 1

    honest = _batch([closed])
    clean = contract_metrics(honest[0], honest[1], negative_ids=honest[2])
    assert clean["false_block_rate"] == 0.0, "the gold route on a negative is not a block"


@pytest.mark.phase4
def test_false_block_denominator_is_reported_when_it_is_only_partial(scenarios) -> None:
    """Without the supplied ids the denominator silently shrinks — so say so.

    `is_negative` lives on the SCENARIO, not the contract, and the generated negatives include
    templates with a non-empty `must_call` (a sanctions hit whose correct outcome is escalation,
    a duplicate application whose correct outcome is a decline). A scorer holding only contracts
    can see the `empty must_call` template and nothing else, so the measured gap below is what
    would be reported as "the false-block rate" by anyone who inferred the denominator.
    """
    negatives = [s for s in scenarios if s.is_negative]
    no_action = [s for s in negatives if not s.contract.must_call]
    with_actions = [s for s in negatives if s.contract.must_call]
    assert no_action, "the generator no longer emits an empty-must_call negative"
    assert with_actions, "the generator no longer emits negatives with mandatory actions"

    # Deliberately mixed, so the two denominators are different numbers.
    batch = no_action[:1] + _sample(with_actions, 5)
    episodes, contracts, supplied = _batch(batch)
    structural = sum(1 for s in batch if not s.contract.must_call)

    exact = contract_metrics(episodes, contracts, negative_ids=supplied)
    partial = contract_metrics(episodes, contracts)

    assert exact["false_block_denominator"] == len(supplied)
    assert "supplied" in exact["false_block_source"][0]

    assert structural < len(supplied), "the structural marker must undercount the real set"
    assert partial["false_block_denominator"] == structural
    assert any("PARTIAL" in source for source in partial["false_block_source"]), (
        "a partial denominator must be labelled as partial, not presented as a rate"
    )
    # And it is a real undercount of a real gap, not a cosmetic label: the exact run had
    # negatives whose denominator the partial run could not see at all.
    assert exact["false_block_rate"] == 0.0 and partial["false_block_rate"] == 0.0


@pytest.mark.phase4
def test_trap_success_is_undefined_without_E3_episodes(scenarios) -> None:
    """None is not 0.0. A batch with no trap episode has no trap-success rate at all."""
    e1 = [s for s in scenarios if s.split == "E1"][:10]
    e3 = [s for s in scenarios if s.split == "E3"][:10]

    episodes, contracts, negatives = _batch(e1)
    without = contract_metrics(episodes, contracts, negative_ids=negatives)
    assert without["trap_success_E3"] is None, "an E1 batch has no E3 rate to report"
    assert without["verified_task_success"] == 1.0

    episodes, contracts, negatives = _batch(e3)
    with_traps = contract_metrics(episodes, contracts, negative_ids=negatives)
    assert with_traps["trap_success_E3"] == 1.0


@pytest.mark.phase4
def test_an_unresolvable_contract_is_reported_and_not_scored(scenarios) -> None:
    """An episode we cannot tie to a contract is `unresolved`, never a silent failure."""
    episode = _replay(_positive(scenarios))
    result = contract_metrics([episode], {}, negative_ids=set())

    assert result["resolved"] == 0
    assert len(result["unresolved"]) == 1
    assert result["verified_task_success"] is None


@pytest.mark.phase4
def test_tool_metrics_reports_an_unscored_metric_rather_than_a_zero(scenarios) -> None:
    """A metric with no evidence behind it is None and named — never 0.0.

    A zero reads as a measured failure. `tool_call_f1` needs the reference route, and a
    replay attaches it only when the caller does; when it is absent the honest answer is
    "not scored", with the reason attached.
    """
    scenario = _positive(scenarios)
    episode = _replay(scenario)

    bare = tool_metrics([episode])
    assert bare["tool_call_validity"] == 1.0, "every gold call is schema-valid"
    assert bare["loop_free_rate"] == 1.0
    assert bare["redundant_tool_calls"] == 0.0
    assert bare["tool_call_f1"] is None, "a missing reference route is not an F1 of zero"
    assert "tool_call_f1" in {u["metric"] for u in bare["unscored"]}
    assert all(u["reason"] for u in bare["unscored"]), "every gap must carry its reason"

    # With the reference route attached, the same episode scores a perfect F1 and in-order
    # route. This is the shape the harness must build when it loads a recording.
    scored = tool_metrics([_Recorded(episode, scenario.gold_route)])
    assert scored["tool_call_f1"] == 1.0
    assert scored["trajectory_in_order"] == 1.0
    assert "tool_call_f1" not in {u["metric"] for u in scored["unscored"]}


# ---------------------------------------------------------------------- the statistics -----
@pytest.mark.phase4
def test_paired_bootstrap_finds_a_planted_gain() -> None:
    """A 30-point improvement on 100 paired episodes must be detected and be significant."""
    before = {f"s{i}": 0.0 for i in range(100)}
    after = {f"s{i}": (1.0 if i < 30 else 0.0) for i in range(100)}

    result = paired_bootstrap(before, after, n_resamples=2_000, seed=42)
    assert result["n"] == 100
    assert result["delta"] == pytest.approx(0.30, abs=1e-9)
    assert result["ci_low"] > 0.0, "a real gain must have a CI excluding zero"
    assert result["p_value"] < 0.05
    assert result["underpowered"] is False
    assert result["episodes_dropped"] == 0


@pytest.mark.phase4
def test_the_seed_drives_the_resampling_but_never_the_effect_size() -> None:
    """The interval is a property of the seed; the delta is a property of the data.

    Both halves matter. If two runs of the same analysis disagreed on the interval, the table
    would be a draw. If the seed moved the POINT estimate, the analysis would be reporting the
    generator's noise as the finding. Continuous paired values are used deliberately: with
    coarse 0/1 deltas the 2.5th percentile lands on the same discrete value under every seed,
    which would make "the seed changed the interval" a flaky assertion rather than a fact.
    """
    before = {f"s{i}": i / 100.0 for i in range(60)}
    after = {f"s{i}": i / 100.0 + ((i * 37) % 11 - 5) / 100.0 for i in range(60)}

    first = paired_bootstrap(before, after, n_resamples=1_000, seed=42)
    second = paired_bootstrap(before, after, n_resamples=1_000, seed=42)
    other = paired_bootstrap(before, after, n_resamples=1_000, seed=7)

    assert first == second, "the same seed must reproduce the same analysis exactly"
    assert other["delta"] == first["delta"], "the effect size must not depend on the seed"
    assert other["mean_after"] == first["mean_after"]
    assert other["ci_low"] != first["ci_low"], "the seed must actually drive the resampling"
    assert abs(other["ci_low"] - first["ci_low"]) < 0.05, "and only by Monte Carlo error"


@pytest.mark.phase4
def test_paired_bootstrap_reports_the_episodes_it_dropped() -> None:
    """Arms are compared only where both were run, and the shortfall is reported."""
    before = {f"s{i}": 0.0 for i in range(10)}
    after = {f"s{i}": 1.0 for i in range(8)}
    result = paired_bootstrap(before, after, n_resamples=200, seed=42)

    assert result["n"] == 8
    assert result["episodes_dropped"] == 2


@pytest.mark.phase4
def test_mcnemar_is_exact_on_discordant_pairs() -> None:
    """Hand-computed exact values, so the test does not agree with a wrong implementation."""
    # 5 fixed, 0 broke: 2 * P(X <= 0) for X ~ Binomial(5, 0.5) = 2 * (1/32) = 0.0625
    before = {f"s{i}": 0.0 for i in range(5)}
    after = {f"s{i}": 1.0 for i in range(5)}
    result = mcnemar(before, after)
    assert (result["fixed"], result["broke"], result["n_discordant"]) == (5, 0, 5)
    assert result["p_value"] == pytest.approx(0.0625, abs=1e-12)

    # A wash: 3 fixed, 3 broke gives 2 * P(X <= 3) = 2 * 42/64 > 1, capped at 1.0
    wash_before = {**{f"a{i}": 0.0 for i in range(3)}, **{f"b{i}": 1.0 for i in range(3)}}
    wash_after = {**{f"a{i}": 1.0 for i in range(3)}, **{f"b{i}": 0.0 for i in range(3)}}
    wash = mcnemar(wash_before, wash_after)
    assert wash["net"] == 0
    assert wash["p_value"] == 1.0


@pytest.mark.phase4
def test_compare_runs_only_the_pre_registered_contrasts() -> None:
    """The analysis was fixed before the runs. A contrast outside it cannot be added here."""
    outcomes = {f"s{i}": (1.0 if i % 3 else 0.0) for i in range(30)}
    improved = {f"s{i}": (1.0 if i % 3 == 0 else outcomes[f"s{i}"]) for i in range(30)}
    results = {
        "A2": {"verified_task_success": outcomes},
        "A4": {"verified_task_success": improved},
    }

    table = compare(results)
    registered = {f"{a} vs {b}" for a, b, _ in PRE_REGISTERED_CONTRASTS}

    assert table["rows"], "the A4 vs A2 contrast must produce a row"
    assert {row["contrast"] for row in table["rows"]} <= registered
    assert "A4 vs A2" in {row["contrast"] for row in table["rows"]}

    # The three contrasts whose arms are absent are reported, not fabricated.
    assert len(table["skipped"]) == len(PRE_REGISTERED_CONTRASTS) - 1
    assert all("reason" in row for row in table["skipped"])
    assert table["holm"], "a raw p-value must carry its family-wise correction"


@pytest.mark.phase4
def test_the_resolution_limit_is_the_one_the_plan_states() -> None:
    """n = 300 at 50% gives ~2.9 points of standard error, so ~5.8 points is the floor.

    The plan commits to stating this before the runs. Encoding it here means the number in the
    write-up is the one the code produces, not one typed into a document.
    """
    se = standard_error(300, 0.5)
    assert se == pytest.approx(0.0289, abs=5e-4)
    assert 0.05 < resolution_limit(300) < 0.06
