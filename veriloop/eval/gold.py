"""Verify a gold route by replaying it and scoring it with the reward.

This is the generator's strongest gate, and it lives in `eval/` rather than in `gen/` for a
structural reason: **`veriloop/gen/` may not import `veriloop/rewards/`**
(`tests/test_imports.py` enforces the dependency direction). That restriction is not
pedantry — a generator that can see the scorer gets tuned to the scorer, and the scenarios
stop being an independent draw.

`scripts/make_splits.py` joins the two: `gen` produces scenarios, `eval.gold` verifies them,
and the script owns the join.

Why replay the gold at all, when `validate()` already checks that the contract is
satisfiable? Because satisfiability is necessary and not sufficient. A gold route can satisfy
its contract by the letter and still score below 1.0 — a `min_steps` off by one, a
`final_state` assertion the gold route does not actually produce, a forbidden call the
template did not expect to make. Any of those becomes an **unwinnable episode** in the
rollout batch: every sample earns `Γ = 0`, the group advantage collapses to zero, and three
hours of GPU reads as *"RL did nothing"*. Rejecting the scenario at generation time is the
cheapest possible place to catch it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from veriloop.contracts.schema import Contract
from veriloop.env.db import open_world
from veriloop.env.episode import RoutePolicy, run_episode
from veriloop.rewards.state_verified import reward_from_terms
from veriloop.rewards.terms import compute_terms

#: A scenario is scorable only if its own gold route is a perfect episode.
PERFECT = 1.0


@dataclass(frozen=True)
class GoldVerdict:
    ok: bool
    reward: float
    steps: int
    terminated: bool
    detail: str
    terms: Mapping[str, Any]


def verify_gold(
    contract: Contract,
    gold_route: Sequence[Mapping[str, Any]],
    seed: int,
    pre_state: Mapping[str, Any],
    *,
    expect: float = PERFECT,
) -> GoldVerdict:
    """Replay `gold_route` against a fresh world and score it. Never mutates anything shared.

    `expect` is 1.0 for a scheme's gold route. Lowering it is a red flag: it means the
    scheme cannot produce a perfect episode, so no policy can ever be rewarded for doing the
    right thing and the arm will learn nothing.
    """
    world = open_world(seed, pre_state, episode_id=f"gold-{contract.id}")
    try:
        result = run_episode(
            world, RoutePolicy(gold_route), contract.brief, contract.step_budget
        )
        terms = compute_terms(
            contract, result.log, result.snapshot,
            terminated=result.terminated, steps=result.steps,
        )
        reward = reward_from_terms(terms)
    finally:
        world.close()

    ok = abs(reward - expect) < 1e-9
    if ok:
        detail = ""
    else:
        detail = (
            f"gold scored {reward:.4f}, expected {expect:.4f} — this scenario is "
            f"UNWINNABLE as written. gamma={terms.gamma} steps={result.steps}/"
            f"{contract.step_budget} terminated={result.terminated} "
            f"M={terms.M:.3f} F={terms.F} E={terms.E:.3f} D={terms.D:.3f} L={terms.L} "
            f"missing={list(terms.missing_mandatory)} "
            f"forbidden={list(terms.forbidden_seen)}"
        )

    return GoldVerdict(
        ok=ok,
        reward=reward,
        steps=result.steps,
        terminated=result.terminated,
        detail=detail,
        terms=terms.as_row(),
    )
