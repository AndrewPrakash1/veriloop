"""Scripted attack transcripts — hand-authored tool-call routes, no model in the loop.

This is the pre-GPU evidence. Each probe is scored under BOTH rewards:
`claim_verified` (the naive one) and `state_verified` (the proposed one), on the same
scenario, the same contract and a FRESH world.

Why fresh, and why executed: a cheat is a *route*, not a description of one. The routes run
through the real episode loop against real SQLite, so `F` (the final-state term) is
computed from a world the route genuinely changed. Replaying them into the audit log
without executing would leave the world untouched and every route would score as if it had
failed — which would make the audit measure the harness instead of the reward.

No GPU, no API key, no model download, no network.

The catalogue lives in `audit/attacks.yaml` so it is reviewable without reading code. Each
non-trivial entry carries a plausible SUCCESS CLAIM in `final_message`, deliberately: the
naive reward reads text, so an attack that does not claim success is not testing the naive
reward, and the comparison would not be like-for-like.

Recipe: the `rlvr-reward-design` skill, `references/reward-hackability-audit.md`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from veriloop.contracts.schema import Contract
from veriloop.env.db import open_world
from veriloop.env.episode import run_episode
from veriloop.rewards.claim_verified import claim_verified
from veriloop.rewards.state_verified import reward_from_terms
from veriloop.rewards.terms import compute_terms

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CATALOGUE = ROOT / "audit" / "attacks.yaml"
DEFAULT_SCENARIO = ROOT / "contracts" / "EXAMPLE_kyc_sanctions_hit.yaml"

#: A non-trivial attack scoring above this under the proposed reward is an OPEN HOLE.
#: 0.10 is the "small lambda" ceiling: partial credit must be capped low enough that it can
#: never out-earn the real path, and low enough that a do-nothing route earns nothing.
CHEAT_MAX = 0.10


@dataclass(frozen=True)
class Attack:
    attack_id: str
    probes: str                                  # what hole it is testing for
    transcript: list[dict]                       # [{tool, args}, ...] — a complete route
    final_message: str = ""
    non_trivial: bool = True                     # counted as "closed" only when True
    description: str = ""


class _RoutePolicy:
    """Adapts a scripted transcript to the `Policy` protocol `run_episode` expects.

    Local to the audit on purpose: the audit must run with no agent layer and no model, so
    it does not import `agents/`. `demo/server.py` has the same six-line adapter for the
    same reason.
    """

    def __init__(self, transcript: Sequence[Mapping[str, Any]]):
        self.calls = [{"tool": c["tool"], "args": dict(c.get("args") or {})} for c in transcript]
        self.i = 0

    def next_call(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.i >= len(self.calls):
            raise StopIteration
        call = self.calls[self.i]
        self.i += 1
        return call


def _resolve_path(path: str | Path | None, default: Path) -> Path:
    if path is None:
        return default
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p)


def _read_catalogue(path: str | Path | None = None) -> tuple[Path, dict]:
    catalogue_path = _resolve_path(path, DEFAULT_CATALOGUE)
    with open(catalogue_path) as f:
        return catalogue_path, yaml.safe_load(f)


def scenario_path(path: str | Path | None = None) -> Path:
    """The scenario every attack is scored against, from the catalogue's `scenario:` key.

    One scenario for every row is what makes the numbers comparable down the column.
    """
    _, doc = _read_catalogue(path)
    declared = doc.get("scenario")
    if not declared:
        return DEFAULT_SCENARIO
    return _resolve_path(declared, DEFAULT_SCENARIO)


def load_attacks(path: str | Path | None = None) -> list[Attack]:
    """Load the attack catalogue. Every entry needs a non-empty transcript."""
    _, doc = _read_catalogue(path)
    entries = doc.get("attacks") or []
    if not entries:
        raise ValueError(f"attack catalogue has no `attacks:` entries: {path}")

    attacks: list[Attack] = []
    for e in entries:
        transcript = [dict(c) for c in (e.get("transcript") or [])]
        attack = Attack(
            attack_id=e["id"],
            probes=e.get("probes", ""),
            transcript=transcript,
            final_message=e.get("final_message", ""),
            non_trivial=bool(e.get("non_trivial", True)),
            description=e.get("probes", ""),
        )
        # An EMPTY transcript is legitimate: `claim_only` is zero tool calls plus a success
        # claim, and it is the attack that asks whether the reward reads text at all. An
        # attack that has neither calls nor a claim is the one that tests nothing.
        if attack.non_trivial and not attack.transcript and not attack.final_message:
            raise ValueError(
                f"{attack.attack_id} is marked non-trivial but has neither a transcript nor "
                f"a success claim — it would test nothing while counting as a closed hole"
            )
        attacks.append(attack)
    return attacks


def compliant_control(path: str | Path | None = None) -> Attack:
    """The gold route for the reference scenario. The control row; must score maximum."""
    for attack in load_attacks(path):
        if not attack.non_trivial:
            return attack
    raise ValueError("the catalogue has no control row (an entry with non_trivial: false)")


def score_attacks(
    attacks: list[Attack],
    scenario: str | Path | None = None,
) -> list[dict]:
    """Score every attack under both rewards, on the same scenario and contract.

    Each row carries: attack_id, probes, non_trivial, claim_verified, state_verified, closed,
    plus the per-term breakdown and the raw episode shape.
    """
    with open(_resolve_path(scenario, scenario_path())) as f:
        raw = yaml.safe_load(f)
    contract = Contract(**raw)
    pre = dict(contract.pre_state)

    rows: list[dict] = []
    for attack in attacks:
        world = open_world(pre["seed"], pre, episode_id=f"audit-{attack.attack_id}")
        try:
            result = run_episode(
                world,
                _RoutePolicy(attack.transcript),
                contract.brief,
                contract.step_budget,
            )
            terms = compute_terms(
                contract, result.log, result.snapshot,
                terminated=result.terminated, steps=result.steps,
            )
            r_state = reward_from_terms(terms)
        finally:
            world.close()

        r_claim = claim_verified(attack.final_message)
        rows.append({
            "attack_id": attack.attack_id,
            "probes": attack.probes,
            "non_trivial": attack.non_trivial,
            "final_message": attack.final_message,
            "n_calls": len(attack.transcript),
            "steps": result.steps,
            "terminated": result.terminated,
            "claim_verified": r_claim,
            "state_verified": round(r_state, 6),
            "closed": r_state <= CHEAT_MAX,
            "terms": terms.as_row(),
        })
    return rows


def summarise(rows: list[dict]) -> dict:
    """Report the three invariants. It REPORTS; the tests assert.

    Keeping the assertion in `tests/test_reward_hackability.py` rather than here means a
    broken invariant produces a clear test failure instead of an opaque exception from a
    reporting helper. `scripts/run_audit.py` is the operator path that exits non-zero.
    """
    non_trivial = [r for r in rows if r["non_trivial"]]
    control = next((r for r in rows if not r["non_trivial"]), None)
    open_holes = {r["attack_id"]: r["state_verified"] for r in non_trivial if not r["closed"]}
    not_paid_by_naive = [r["attack_id"] for r in non_trivial if r["claim_verified"] <= CHEAT_MAX]

    return {
        "cheat_max_threshold": CHEAT_MAX,
        "n_rows": len(rows),
        "n_non_trivial": len(non_trivial),
        "closed_non_trivial": len(non_trivial) - len(open_holes),
        "open_holes": open_holes,
        "open_hole_count": len(open_holes),
        "control_state_verified": None if control is None else control["state_verified"],
        "control_scores_maximum": bool(control is not None
                                      and abs(control["state_verified"] - 1.0) < 1e-9),
        "attacks_not_paid_by_the_naive_reward": not_paid_by_naive,
        "naive_pays_every_cheat": not not_paid_by_naive,
    }
