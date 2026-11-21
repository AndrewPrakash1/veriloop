"""Phase 6 — the RL trainer, everything about it that can be checked without a GPU.

The generation and the backward pass cannot run here: there is no torch and no CUDA on this
machine. What CAN be checked is the arithmetic the whole run rests on — the RLOO advantage,
the log-prob span, the reward wiring, and the guard that stops a run from eating the held-out
split. Those are the parts where a mistake is invisible in the output: an off-by-one span or a
zero advantage both train happily and print a falling loss.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.phase6


# --------------------------------------------------------------------- the advantage ----
def test_rloo_advantage_is_leave_one_out_and_sums_to_zero() -> None:
    """`A_i = R_i - mean(other G-1)`, which is what makes RLOO critic-free."""
    from train.rl import advantages

    rewards = [1.0, 2.0, 3.0, 4.0]
    adv = advantages(rewards)

    # For i=0: 1.0 - mean(2,3,4) = 1.0 - 3.0 = -2.0
    assert adv[0] == pytest.approx(-2.0)
    # For i=3: 4.0 - mean(1,2,3) = 4.0 - 2.0 = 2.0
    assert adv[3] == pytest.approx(2.0)
    # Leave-one-out advantages always sum to zero: the baseline is the same mean in every
    # term, so the sum telescopes. If this ever fails, the baseline is not leave-one-out.
    assert sum(adv) == pytest.approx(0.0, abs=1e-9)


def test_a_group_that_scores_identically_has_no_signal() -> None:
    """All-equal rewards must give exactly zero advantage, not 'small'.

    This is the failure the whole probe mode exists to expose: if the G rollouts are the same
    episode (e.g. because the policy re-seeds before every generation), the reward is the same
    for all of them, every advantage is 0, and the run trains on nothing while printing a
    falling loss.
    """
    from train.rl import advantages

    assert advantages([0.5] * 8) == [0.0] * 8
    assert advantages([-1.0] * 3) == [0.0] * 3


def test_a_group_of_one_cannot_produce_a_signal() -> None:
    """No other rollout to compare against means no baseline, so no advantage."""
    from train.rl import advantages

    assert advantages([2.5]) == [0.0]
    assert advantages([]) == []


# ------------------------------------------------------------------- the scored span ----
def test_the_completion_span_starts_at_the_right_index() -> None:
    """`logits[:, i]` predicts `ids[i+1]`, so the completion begins at `n_prompt - 1`.

    An off-by-one here trains on the last prompt token as if it were the model's action. The
    run still works, the loss still falls, and the gradient is wrong — so it is pinned.
    """
    from train.rl import completion_slice_start

    assert completion_slice_start(10) == 9
    assert completion_slice_start(1) == 0
    # A truncated-away prompt must not index backwards.
    assert completion_slice_start(0) == 0
    assert completion_slice_start(-5) == 0


# ------------------------------------------------------------------- reward wiring ------
def test_the_arm_is_the_only_thing_that_changes() -> None:
    """A0/A1 carry no reward, A2..A5 each carry exactly one, and A3 is the mandatory_only
    variant of A4. This is the ablation's whole claim, so it is asserted rather than assumed."""
    from functools import partial

    from veriloop.rewards.registry import reward_funcs

    assert reward_funcs("A0") == [], "A0 is the base model: it is not trained on a reward"
    assert reward_funcs("A1") == [], "A1 is the SFT cold start: it is not trained on a reward"

    for arm in ("A2", "A3", "A4", "A5"):
        funcs = reward_funcs(arm)
        assert len(funcs) == 1, f"{arm} must carry exactly one reward function"

    a3 = reward_funcs("A3")[0]
    assert isinstance(a3, partial) and a3.keywords["mode"] == "mandatory_only", (
        "A3 asks whether the contract check is enough on its own; if it stops being the "
        "mandatory-only variant it is no longer that arm"
    )
    a4 = reward_funcs("A4")[0]
    assert not isinstance(a4, partial), "A4 is the shipped combination, unmodified"


def test_a_reward_less_arm_raises_rather_than_training_on_zero() -> None:
    """Training an A0/A1 run on a constant zero reward would look like a valid ablation."""
    from train.rl import total_reward

    with pytest.raises(ValueError, match="no reward functions"):
        total_reward([], episode=None, contract=None)


def test_rewards_are_summed_over_the_arm_s_terms() -> None:
    """One arm, one number: several reward functions must add up, not replace each other."""
    from train.rl import total_reward

    funcs = [lambda episode, contract: 0.25, lambda episode, contract: -1.0]
    assert total_reward(funcs, episode=None, contract=None) == pytest.approx(-0.75)


def test_an_empty_parameter_list_is_refused_rather_than_optimised() -> None:
    """A frozen adapter trains nothing and looks exactly like a successful run.

    `load_model(adapter=...)` wraps a checkpoint with `is_trainable=False`; `AdamW([])` accepts
    that, `.step()` is a no-op, and the forward pass still prints a falling loss. Nothing in
    the log distinguishes it from real training, so the empty case must be a hard stop.
    """
    from train.rl import require_trainable

    with pytest.raises(SystemExit, match="no trainable parameters"):
        require_trainable([])

    sentinel = [object()]
    assert require_trainable(sentinel) is sentinel


# ------------------------------------------------------------------ the split guard -----
@pytest.mark.parametrize("name", ["data/splits/E3.jsonl", "/workspace/veriloop/data/splits/E3.jsonl"])
def test_training_on_e3_is_refused(name: str) -> None:
    """E3 is the held-out adversarial split. Training on it destroys the only measurement of
    rewards without solver pressure that the project has, and the rule is in the module's own
    docstring — so it is enforced, not remembered at 04:00."""
    from train.rl import assert_trainable_split

    with pytest.raises(SystemExit, match="held-out eval split"):
        assert_trainable_split(name)


def test_training_on_the_train_rl_split_is_allowed() -> None:
    from train.rl import assert_trainable_split

    assert_trainable_split("data/splits/train_rl.jsonl")
    assert_trainable_split("data/splits/train.jsonl")


# ------------------------------------------------------------------- instrumentation ----
class _Record:
    def __init__(self, tool: str, args: dict) -> None:
        self.tool = tool
        self.args = args
        self.ok = True
        self.schema_valid = True
        self.step = 0


class _Episode:
    def __init__(self, records: list) -> None:
        self.log = type("L", (), {"records": records})()


def test_duplicate_rate_counts_repeated_calls_not_repeated_tools() -> None:
    """The reward's duplicate term is about the same CALL, so two `record_check`s with
    different `check_type` are not duplicates — and the log has to agree with the reward."""
    from train.rl import duplicate_rate

    records = [
        _Record("record_check", {"check_type": "identity"}),
        _Record("record_check", {"check_type": "sanctions"}),
        _Record("record_check", {"check_type": "identity"}),      # a real duplicate
        _Record("lookup_customer", {"customer_id": "c1"}),
    ]
    assert duplicate_rate(_Episode(records)) == pytest.approx(0.25)

    # Argument order must not matter, or the metric disagrees with a dict comparison.
    reordered = [_Record("lookup_customer", {"a": 1, "b": 2}),
                 _Record("lookup_customer", {"b": 2, "a": 1})]
    assert duplicate_rate(_Episode(reordered)) == pytest.approx(0.5)

    assert duplicate_rate(_Episode([])) is None, "an empty log is unmeasured, not zero"


def test_logged_duplicate_rate_flags_a_looping_policy() -> None:
    """A policy can shorten an episode to raise its reward while quietly looping. The logged
    rate is what makes that visible."""
    from train.rl import duplicate_rate

    looping = [_Record("lookup_customer", {"customer_id": "c1"})] * 4
    assert duplicate_rate(_Episode(looping)) == pytest.approx(0.75)


# ------------------------------------------------------------------------- the plan -----
def test_the_plan_is_the_episode_count_not_the_step_count() -> None:
    """Cost is episodes, not steps, and the configs' default is 300 x 2 x 8 = 4,800 per arm.
    Printed before a run starts, because reading it afterwards is too late."""
    from train.rl import plan_line, planned_episodes

    assert planned_episodes(300, 2, 8) == 4800
    assert planned_episodes(30, 2, 8) == 480
    assert "4800 episodes" in plan_line(300, 2, 8, seconds_per_episode=None)
    # With a measured per-episode cost the line also carries the wall clock and the rupees.
    sized = plan_line(30, 2, 8, seconds_per_episode=5.0)
    assert "480 episodes" in sized and "min" in sized and "Rs" in sized


def test_seconds_per_episode_scales_the_estimate() -> None:
    from train.rl import plan_line

    assert "40 min" in plan_line(30, 2, 8, seconds_per_episode=5.0)      # 480*5/60 = 40
    assert "80 min" in plan_line(30, 2, 8, seconds_per_episode=10.0)


# ------------------------------------------------------------------------- the CLI ------
def test_the_rl_cli_defaults_match_the_configs() -> None:
    """Defaults come from the arm config, so the CLI has to be able to express the run without
    contradicting it. `--probe` defaults to 0: a bare invocation is a training run."""
    from train.rl import _parser

    args = _parser().parse_args([])
    assert args.arm == "A3", "the state-verified arm is the one the thesis needs"
    assert args.probe == 0
    assert args.config.endswith("grpo_state.yaml")
    assert args.steps is None and args.group_size is None and args.batch_size is None, (
        "None means 'take it from the config', which is how the ablation stays one switch point"
    )

    probe = _parser().parse_args(["--arm", "A3", "--probe", "2"])
    assert probe.probe == 2 and probe.out is None


def test_a_probe_cannot_also_save_a_checkpoint() -> None:
    """The probe takes no optimizer, so an --out would write the SFT adapter back out
    unchanged and look like a trained arm."""
    from train.rl import main

    with pytest.raises(SystemExit, match="probe takes no optimizer"):
        main(["--arm", "A3", "--probe", "2", "--out", "outputs/A3_state"])


# ------------------------------------------------------------------------ the config ----
def test_arm_configs_resolve_their_extends_chain(repo_root) -> None:
    """Every arm `extends: base.yaml` so an ablation cannot differ in anything but the reward.
    If the resolution breaks, an arm silently trains with the wrong model or LoRA shape."""
    from train.rl import load_config

    config = load_config(str(repo_root / "configs" / "grpo_state.yaml"))
    assert config["model"] == "CohereLabs/tiny-aya-global"
    assert config["load_in_4bit"] is True
    assert config["lora"]["r"] == 16
    # The child's own keys survive, and its `data:` mapping merges over the parent's rather
    # than replacing it wholesale.
    assert config["stage"] == "rl"
    assert config["data"]["file"] == "data/splits/train_rl.jsonl"
    assert config["reward_funcs"].endswith("(mode=mandatory_only)")
    # A0/A1 come from the base config and must still be there.
    assert config["eval"]["splits"] == ["E1", "E2", "E3", "E4"]
