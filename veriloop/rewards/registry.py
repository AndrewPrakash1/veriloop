"""Register rewards as callables for the rollout engine, and make the arm the only
thing that changes between runs.

`reward_funcs(arm) -> list[callable]`. One switch point, so an ablation cannot
accidentally differ in anything but the reward.
"""

from __future__ import annotations

from typing import Callable

ARMS = {
    "A0": "base model, no training",
    "A1": "SFT cold start only",
    "A2": "claim-verified text reward (naive)",
    "A3": "state-verified, mandatory term only",
    "A4": "state-verified + shaping (S, E, D, L)",
    "A5": "random reward (content-independent control)",
    "A6": "A4 + self-healing rounds",
}


def reward_funcs(arm: str) -> list[Callable]:
    """Return the reward functions for `arm`. Raise on an unknown arm.

    A5 (random) pays a content-independent coin flip. It is not decoration: RLVR gains
    can come from clipping, output length or the base model, so without this control an
    improvement is not attributable to the verifier.
    """
    raise NotImplementedError("Phase 2")


def random_reward(*_args, **_kwargs) -> float:
    """Content-independent. Seeded, so a run is reproducible."""
    raise NotImplementedError("Phase 2")
