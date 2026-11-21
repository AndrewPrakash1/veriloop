"""Register rewards as callables for the rollout engine, and make the arm the only
thing that changes between runs.

`reward_funcs(arm) -> list[callable]`. One switch point, so an ablation cannot
accidentally differ in anything but the reward function.

THE SHAPE OF A REWARD HERE

    reward(episode, contract) -> float

`episode` is anything shaped like `EpisodeResult`: `log`, `snapshot`, `steps`, `terminated`,
`final_message`. That is what `run_episode` returns and what a rollout engine hands back, and
it is the same tuple `eval/metrics.py` consumes — so an arm's reward and its scorecard are
computed from one representation.

Only ONE of these functions reads the model's prose: `claim_verified`, the naive arm, whose
entire point is that it does. If the rollout engine wants a different signature, ONE adapter
translates it — the same rule as `strategy.to_agenttune_action`. Translate at the seam, once.

THE CONTROL ARM IS NOT DECORATION

A5 pays a coin flip. RLVR gains can come from clipping, output length or the base model, so
without a content-independent control an improvement is not attributable to the verifier —
which is the whole claim (H3). It is seeded so a run is reproducible, and `reset_random()`
re-seeds it at the top of a comparison so two arms see the same sequence of flips.
"""

from __future__ import annotations

import random
from functools import partial
from typing import Any, Callable

from veriloop.rewards.claim_verified import claim_verified
from veriloop.rewards.state_verified import W, reward_from_terms
from veriloop.rewards.terms import compute_terms

ARMS = {
    "A0": "base model, no training",
    "A1": "SFT cold start only",
    "A2": "claim-verified text reward (naive)",
    "A3": "state-verified, mandatory term only",
    "A4": "state-verified + shaping (S, E, D, L)",
    "A5": "random reward (content-independent control)",
    "A6": "A4 + self-healing rounds",
}

#: A0 and A1 are not trained against a reward — the base model and the SFT checkpoint. An
#: empty list is what the trainer should see. If something trains anyway, the arm is not the
#: ablation it claims to be, so this is stated rather than silently defaulted.
NO_TRAINING = ("A0", "A1")

#: Fixed so a run is reproducible. The sequence depends on call order, which depends on the
#: rollout order — deterministic for a given seed set, which is what the pre-registration needs.
RANDOM_SEED = 42
_RNG = random.Random(RANDOM_SEED)


def reset_random(seed: int = RANDOM_SEED) -> None:
    """Re-seed the A5 control. Call once at the top of a comparison, so both arms see the same
    sequence of coin flips and the difference between them cannot be the dice."""
    _RNG.seed(seed)


def _terms_of(episode: Any, contract: Any):
    return compute_terms(
        contract,
        episode.log,
        episode.snapshot,
        terminated=bool(episode.terminated),
        steps=int(episode.steps),
    )


def state_reward(episode: Any, contract: Any, *, mode: str = "full") -> float:
    """The proposed reward. `mode` is the ONLY thing that distinguishes A3 from A4.

    `full`            the shipped combination: gate, the `M = 1` cliff, final state, the
                      guarded efficiency bonus, minus duplicates and loops.
    `mandatory_only`  the completion cliff and the gate penalty, nothing else. This is arm A3,
                      and it asks a real question: is the contract check enough on its own, or
                      do the shaping terms buy something?

    Both modes read the SAME terms, so the two arms cannot differ in anything but the
    combination rule. Constants come from `rewards.state_verified.W`, so an arm cannot be
    tuned by editing this file.
    """
    if mode not in ("full", "mandatory_only"):
        raise ValueError(f"unknown state reward mode {mode!r}; expected 'full' or 'mandatory_only'")
    terms = _terms_of(episode, contract)
    if mode == "full":
        return reward_from_terms(terms)
    completion = W["mandatory"] * (1.0 if terms.M >= 1.0 else 0.0)
    return terms.gamma * completion - W["gate_penalty"] * (1 - terms.gamma)


def claim_reward(episode: Any, contract: Any) -> float:
    """The naive arm, trained as A2. TEXT ONLY — it never looks at the state or the log.

    `contract` is accepted and ignored so every reward shares one signature. That uniformity is
    deliberate: it means the swap between arms cannot change anything else, which is the whole
    point of the ablation.
    """
    return claim_verified(getattr(episode, "final_message", ""))


def random_reward(*_args: Any, **_kwargs: Any) -> float:
    """Content-independent. Seeded, so a run is reproducible."""
    return _RNG.random()


def reward_funcs(arm: str) -> list[Callable]:
    """Return the reward functions for `arm`. Raise on an unknown arm.

    A5 (random) pays a content-independent coin flip. It is not decoration: RLVR gains
    can come from clipping, output length or the base model, so without this control an
    improvement is not attributable to the verifier.
    """
    key = str(arm).upper()
    if key not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known arms: {sorted(ARMS)}")
    if key in NO_TRAINING:
        return []
    if key == "A2":
        return [claim_reward]
    if key == "A3":
        return [partial(state_reward, mode="mandatory_only")]
    if key in ("A4", "A6"):
        # A6 is A4 PLUS the healing loop. The loop changes the DATA, not the reward; if this
        # ever returns something else, A6 vs A4 stops measuring the loop.
        return [state_reward]
    if key == "A5":
        return [random_reward]
    raise AssertionError(f"arm {key!r} is in ARMS but has no reward — add it above")
