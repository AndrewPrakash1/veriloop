"""Flag the episodes worth learning from.

A `ScoredEpisode` carries the terms as well as the score, because the classifier needs them
and recomputing them downstream would let the two views drift. The score is the reward the arm
was TRAINED against — using it here is not circular: detection decides what to drill, and the
drill is then verified by `eval/gold.verify_gold` against the contract, not against the reward.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from veriloop.contracts.schema import Contract
from veriloop.eval.metrics import resolve_contract
from veriloop.rewards.state_verified import state_verified
from veriloop.rewards.terms import Terms, compute_terms


@dataclass(frozen=True)
class ScoredEpisode:
    """One episode plus everything the loop needs to decide what it was."""

    episode: Any
    contract: Contract | None
    terms: Terms | None
    reward: float
    reasons: tuple[str, ...]        # why it was flagged; empty means it was NOT flagged

    @property
    def episode_id(self) -> str:
        return str(getattr(self.episode, "episode_id", "<unknown>"))


def score_episode(episode: Any, contract: Contract | None) -> ScoredEpisode:
    """Score one episode. An unresolvable contract is a REASON, never an exception."""
    if contract is None:
        return ScoredEpisode(episode, None, None, 0.0, ("contract_unresolved",))
    terms = compute_terms(
        contract, episode.log, episode.snapshot,
        terminated=episode.terminated, steps=episode.steps,
    )
    reward = state_verified(
        contract, episode.log, episode.snapshot,
        terminated=episode.terminated, steps=episode.steps,
    )
    reasons: list[str] = []
    if not episode.terminated:
        reasons.append("not_terminated")
    if episode.steps >= contract.step_budget:
        reasons.append("budget_exhausted")
    if terms.gamma == 0:
        reasons.append("gate_failed")
    if terms.F < 1.0:
        reasons.append("final_state_failed")
    if terms.M < 1.0:
        # Flagged on its own, independent of the reward's sign. An episode that skipped
        # mandatory work can still score well — the shaping terms only pay 0.5 for M, and a
        # route that skips one screen of six keeps most of it. Gating detection on the reward
        # would therefore skip exactly the near-miss failures the drills exist to fix.
        reasons.append("mandatory_incomplete")
    return ScoredEpisode(episode, contract, terms, reward, tuple(reasons))


def detect_failures(
    episodes: Sequence[Any],
    contracts: Mapping[str, Contract] | dict,
    threshold: float = 0.0,
) -> list[ScoredEpisode]:
    """Return the episodes whose reward is at or below `threshold`, or that blew the budget.

    Returns `ScoredEpisode` wrappers rather than bare episodes: the classifier needs the
    terms, and re-deriving them at the next step is how two views of the same episode start
    disagreeing. `threshold` defaults to 0.0 because every failed episode is worth classifying
    — the round's DRILLS are then chosen by class frequency, not by severity.
    """
    out: list[ScoredEpisode] = []
    for ep in episodes:
        scored = score_episode(ep, resolve_contract(ep, contracts))
        if scored.reward <= threshold or scored.reasons:
            out.append(scored)
    return out


def class_counts(scored: Sequence[ScoredEpisode]) -> dict[str, int]:
    """Failure counts by class name — the round's drill plan."""
    from veriloop.heal.classify import classify

    counts: dict[str, int] = {}
    for s in scored:
        name = classify(s.terms, s.episode, contract=s.contract).value
        counts[name] = counts.get(name, 0) + 1
    return counts
