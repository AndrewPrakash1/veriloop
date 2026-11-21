"""Phase 5 (CPU side) — the SFT data path and the arm registry, with no GPU and no model.

Two invariants decide whether SFT can work at all, and both are cheap to check here:

  * the training prompt is built by `agents.policy.render_turn` — THE SAME function the policy
    uses at inference, so what the model is trained on is what it is scored on;
  * every target round-trips through `strategy.next_call_from_completion` back to the gold
    call, so the model is taught something the harness can actually consume.

Getting either wrong produces a model that looks broken for reasons living in this repo rather
than in the model — the most expensive kind of wrong, because the obvious response is to buy
more GPU hours.

The registry half asserts the project's thesis as a unit test: on the same episode, the naive
text reward and the state-verified reward must DISAGREE, and they must disagree in the
direction the whole project claims.
"""

from __future__ import annotations

import json

import pytest

from veriloop.env.db import open_world
from veriloop.env.episode import RoutePolicy, run_episode
from veriloop.gen.splits import build_scenarios
from veriloop.rewards.registry import (
    NO_TRAINING,
    claim_reward,
    random_reward,
    reset_random,
    reward_funcs,
    state_reward,
)

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
    contract = scenario.contract
    pre = dict(contract.pre_state)
    world = open_world(pre["seed"], pre, episode_id=f"sft-test-{contract.id}")
    try:
        return run_episode(
            world,
            RoutePolicy(list(scenario.gold_route if route is None else route)),
            contract.brief,
            contract.step_budget,
        )
    finally:
        world.close()


def _positive(scenarios, template=None):
    for scenario in scenarios:
        if not scenario.is_negative:
            if template is None or scenario.template == template:
                return scenario
    raise AssertionError(f"no positive scenario for template={template!r}")


# ------------------------------------------------------------------- the SFT data path ----
@pytest.mark.phase5
def test_one_example_per_gold_step(scenarios) -> None:
    """A route of 8 calls is 8 next-action decisions, not one episode."""
    from train.sft import build_examples

    batch = _sample(scenarios, 12)
    examples = build_examples(batch)

    by_scenario: dict[str, int] = {}
    for example in examples:
        by_scenario[example["scenario_id"]] = by_scenario.get(example["scenario_id"], 0) + 1

    for scenario in batch:
        expected = len(scenario.gold_route)
        got = by_scenario.get(scenario.scenario_id, 0)
        assert got == expected, (
            f"{scenario.scenario_id}: {expected} gold calls but {got} examples — "
            f"the loss would not cover the whole route"
        )


@pytest.mark.phase5
def test_the_training_prompt_is_the_inference_prompt(scenarios) -> None:
    """The load-bearing invariant: ONE renderer, used by both.

    Re-derives the observation independently (by replaying the route and capturing what the
    loop hands the policy) and requires the training prompt to match byte for byte. If a
    future SFT change introduces its own prompt builder, this fails — which is the point.
    """
    from agents.policy import render_turn
    from train.sft import RecordingPolicy, build_examples

    scenario = _positive(scenarios)
    contract = scenario.contract
    pre = dict(contract.pre_state)

    policy = RecordingPolicy(scenario.gold_route)
    world = open_world(pre["seed"], pre, episode_id=f"sft-invariant-{contract.id}")
    try:
        run_episode(world, policy, contract.brief, contract.step_budget)
    finally:
        world.close()

    examples = build_examples([scenario])
    assert len(examples) == len(policy.observations)

    for example, observation in zip(examples, policy.observations):
        assert example["prompt"] == render_turn(observation,
                                                max_steps=contract.step_budget), (
            "the SFT prompt is not the one the policy will send — the model would be trained "
            "on a format nobody scores"
        )


@pytest.mark.phase5
def test_every_target_round_trips_through_the_parser(scenarios) -> None:
    """A target must be readable by the same parser that scores a model turn."""
    from agents.strategy import next_call_from_completion
    from train.sft import build_examples

    names = {c.tool for s in scenarios for c in s.contract.must_call}
    for example in build_examples(_sample(scenarios, 8)):
        call = next_call_from_completion(example["target"], names)
        assert call != {"tool": "", "args": {}}, (
            f"target did not parse back: {example['target']!r}"
        )
        assert call["tool"], f"empty tool name in target {example['target']!r}"


@pytest.mark.phase5
def test_the_history_grows_with_each_step(scenarios) -> None:
    """Step k's prompt carries exactly k tool results — the model is never shown the future."""
    from train.sft import build_examples

    scenario = _positive(scenarios)
    examples = build_examples([scenario])
    assert len(examples) >= 4, "fixture too short to test history growth"

    for index, example in enumerate(examples):
        assert example["prompt"].count("Observation:") == index, (
            f"step {index} shows {example['prompt'].count('Observation:')} observations"
        )
    assert "No tool calls have been made yet." in examples[0]["prompt"]
    assert "No tool calls have been made yet." not in examples[1]["prompt"]


@pytest.mark.phase5
def test_negative_controls_are_in_the_training_data(scenarios) -> None:
    """The anti-escalate lesson has to be IN the data, or "always escalate" is unpunished.

    Asserted on a negative whose gold route performs no account decision, and on one whose gold
    route legitimately does (a decline, an escalation) — both must be present, because a
    negative control is not the same thing as a do-nothing case.
    """
    from train.sft import build_examples, target_text

    negatives = [s for s in scenarios if s.is_negative]
    no_action = [s for s in negatives if len(s.contract.must_call) == 0]
    with_action = [s for s in negatives if s.contract.must_call]
    assert no_action and with_action, "the generator changed shape; re-read templates/negatives"

    for scenario in (no_action[0], with_action[0]):
        examples = build_examples([scenario])
        assert examples, f"{scenario.scenario_id} produced no training examples"
        assert json.loads(examples[-1]["target"])["tool"] == "finish"
        expected_tools = [c["tool"] for c in scenario.gold_route]
        got_tools = [json.loads(e["target"])["tool"] for e in examples]
        assert got_tools == expected_tools
        assert target_text(scenario.gold_route[-1]) == examples[-1]["target"]


# ------------------------------------------------------------------------ the registry ----
@pytest.mark.phase5
def test_every_arm_maps_to_a_reward_or_is_declared_untrained() -> None:
    from veriloop.rewards.registry import ARMS

    for arm in ARMS:
        funcs = reward_funcs(arm)
        assert isinstance(funcs, list)
        if arm in NO_TRAINING:
            assert funcs == [], f"{arm} must have no training reward"
        else:
            assert funcs, f"{arm} is trainable but has no reward"
            assert all(callable(f) for f in funcs)


@pytest.mark.phase5
def test_an_unknown_arm_raises_rather_than_defaulting() -> None:
    with pytest.raises(KeyError, match="unknown arm"):
        reward_funcs("A9_bogus")
    with pytest.raises(ValueError, match="unknown state reward mode"):
        state_reward(None, None, mode="whatever")


@pytest.mark.phase5
def test_the_naive_and_state_rewards_disagree_on_a_shortcut(scenarios) -> None:
    """THE THESIS AS A UNIT TEST: same episode, opposite verdicts.

    Builds the evasion the project exists to catch — record the checks, skip the screens,
    announce success — and requires the naive text reward to PAY it while the state-verified
    reward does not. If these two ever agree here, the ablation measures nothing.
    """
    scenario = _positive(scenarios, template="kyc_clean_pass")
    assert any(c.tool.startswith("screen_") for c in scenario.contract.must_call)

    skipping = [c for c in scenario.gold_route if not c["tool"].startswith("screen_")]
    skipping[-1] = {"tool": "finish",
                    "args": {"summary": "All checks complete. Case escalated."}}
    episode = _replay(scenario, skipping)

    naive = claim_reward(episode, scenario.contract)
    state = state_reward(episode, scenario.contract)

    assert naive == 1.0, "the naive arm must pay a claim of success"
    assert state < 0.5, "the state-verified arm must not pay the same shortcut"


@pytest.mark.phase5
def test_mandatory_only_and_full_are_not_the_same_arm(scenarios) -> None:
    """A3 and A4 differ ONLY in the combination rule, and must actually differ in value."""
    scenario = _positive(scenarios)
    gold = _replay(scenario)

    full = state_reward(gold, scenario.contract, mode="full")
    mandatory = state_reward(gold, scenario.contract, mode="mandatory_only")

    assert full == pytest.approx(1.0), "a perfect episode scores exactly 1.0"
    assert mandatory < full, (
        "mode='mandatory_only' must pay less than the full combination on a perfect episode, "
        "or A3 vs A4 is not an ablation"
    )


@pytest.mark.phase5
def test_the_random_control_is_seeded_and_content_independent(scenarios) -> None:
    """A5 must not be able to see the episode — that is the only reason it is a control."""
    scenario = _positive(scenarios)
    episode = _replay(scenario)

    reset_random(42)
    first = [random_reward(episode, scenario.contract) for _ in range(5)]
    reset_random(42)
    again = [random_reward(episode, scenario.contract) for _ in range(5)]
    assert first == again, "the control must be reproducible from its seed"

    # Different content, same seed: an identical draw proves content cannot influence it.
    # (Not `kyc_sanctions_hit`: every instance of that template is a negative control with a
    # non-empty must_call, so it is not a "positive" to reach for.)
    others = [s for s in scenarios if not s.is_negative and s.template != scenario.template]
    assert others, "need a second positive template for this comparison"
    reset_random(42)
    other_episode = _replay(others[0])
    other = [random_reward(other_episode, scenario.contract) for _ in range(5)]
    assert other == first, "the control's draw must not depend on the episode"
    reset_random(42)


# -------------------------------------------------------------------- the model loader ----
@pytest.mark.phase5
def test_the_shipped_config_dtype_resolves_to_a_real_torch_attribute(repo_root) -> None:
    """The config says `bf16`; torch's attribute is `bfloat16`. **`torch.bf16` DOES NOT EXIST.**

    This is a regression test for a real failure: the pod ran
    `ValueError: unknown dtype 'bf16'; expected bf16, fp16 or float32` — an error message that
    rejected the very spelling it listed as valid. Tying `configs/base.yaml` to the loader's
    alias table means the config cannot drift away from what the loader accepts without failing
    here, on CPU, instead of on a rented GPU.
    """
    import yaml

    from agents.loader import resolve_dtype

    config = yaml.safe_load((repo_root / "configs" / "base.yaml").read_text())
    dtype = config["dtype"]

    class FakeTorch:
        bfloat16 = "BF16"
        float16 = "FP16"
        float32 = "FP32"

    assert resolve_dtype(FakeTorch, dtype) == "BF16", (
        f"the shipped config dtype {dtype!r} does not resolve — the loader would reject the "
        f"value the config ships with"
    )
    # Every spelling this project uses, and case-insensitivity.
    for name, expected in [("bf16", "BF16"), ("bfloat16", "BF16"), ("BF16", "BF16"),
                           ("fp16", "FP16"), ("float16", "FP16"), (" fp32 ", "FP32")]:
        assert resolve_dtype(FakeTorch, name) == expected, name


@pytest.mark.phase5
def test_an_unresolvable_dtype_names_the_valid_options() -> None:
    """The message must list the accepted names — the first version listed the name it refused."""
    from agents.loader import resolve_dtype

    class FakeTorch:
        bfloat16 = "BF16"

    with pytest.raises(ValueError, match="unknown dtype") as bad_name:
        resolve_dtype(FakeTorch, "bfloat")
    assert "bf16" in str(bad_name.value)

    # A valid name that this torch build happens not to expose is a different error.
    with pytest.raises(ValueError, match="no torch.float32"):
        resolve_dtype(FakeTorch, "fp32")
