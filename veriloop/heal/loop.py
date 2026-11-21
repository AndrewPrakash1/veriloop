"""One healing round, end to end, with an append-only audit record per round.

Retraining ALWAYS continues from the current adapter, never from base — otherwise a
healing round silently discards prior learning and the loop looks like progress while
losing ground.

WHAT THIS FILE DOES NOT DO. It does not train and it does not call a model. It takes
recorded episodes and two injected callables — `retrain` and `evaluate` — so the loop's
CONTROL FLOW is exercisable and testable with no GPU, and so the GPU side has exactly two
functions to supply. A loop that can only be tested by training cannot be tested.

`evaluate_with_policy` is the model-free stand-in: it drives any `Policy` over a scenario set
through the real episode loop and returns the three gate scores. That is what lets a round be
rehearsed on this laptop and what the demo uses. It is a real measurement of a scripted
policy — it is NOT a measurement of an untrained model, and any number it produces must be
labelled as a rehearsal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from veriloop.env.db import open_world
from veriloop.env.episode import Policy, run_episode
from veriloop.eval.metrics import contract_metrics
from veriloop.heal.classify import FailureClass
from veriloop.heal.detect import ScoredEpisode, class_counts, detect_failures
from veriloop.heal.gate import GateVerdict, RedeployGate
from veriloop.heal.synthesise import describe_plan, synthesis_plan

#: A retrain step takes the CURRENT adapter path and returns the new one.
RetrainFn = Callable[[str], str]

#: An evaluation step takes an adapter path (or a label) and returns the gate's scores.
EvaluateFn = Callable[[str], Mapping[str, float]]

#: Substrings that mark a path as the BASE model. A round that returns one of these as its
#: output has discarded every prior round.
_BASE_MARKERS = ("base", "pretrained", "preset", "original")


@dataclass
class RoundRecord:
    round_index: int
    counts_by_class: Mapping[str, int]
    synthesised: int
    verdict: GateVerdict
    promoted: bool
    # Added provenance, all defaulted so the originally declared fields are unchanged.
    seed: int = 0
    gate_margin: float = 0.0
    trap_tolerance: float = 0.0
    adapter_in: str = ""
    adapter_out: str | None = None
    plan: Mapping[str, Any] = field(default_factory=dict)
    flagged_episode_ids: tuple[str, ...] = ()
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        row = asdict(self)
        row["verdict"] = self.verdict.as_row()
        return row


def _looks_like_base(path: str) -> bool:
    low = str(path).lower()
    return any(m in low for m in _BASE_MARKERS)


def assert_continues_from_current(adapter_in: str, adapter_out: str | None) -> None:
    """The loop's second safety property, after the gate.

    Continuing from base is not a misconfiguration that shows up as a crash: it produces a
    model that is fine on this round's drills and has lost everything from every earlier one,
    and the round's numbers go UP. So it is refused, not warned about.
    """
    if not str(adapter_in or "").strip():
        raise ValueError(
            "adapter_in is empty. A healing round must be told which adapter is CURRENT: "
            "retraining from base silently discards prior learning and the round still looks "
            "like progress."
        )
    if adapter_out is not None and _looks_like_base(adapter_out) and not _looks_like_base(adapter_in):
        raise ValueError(
            f"retrain returned {adapter_out!r}, which names the base model, while the current "
            f"adapter is {adapter_in!r}. A round must continue from the current adapter."
        )


def _append_record(out_dir: str, record: RoundRecord) -> Path:
    """Append one JSON line. Append-only: an earlier round's record is never rewritten."""
    path = Path(out_dir) / "heal" / "rounds.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(record.as_row(), sort_keys=True, default=str) + "\n")
    return path


def heal_round(
    round_index: int,
    seed: int,
    out_dir: str,
    *,
    episodes: Sequence[Any],
    contracts: Mapping[str, Any],
    evaluate: EvaluateFn,
    retrain: RetrainFn | None = None,
    adapter_in: str = "",
    gate: RedeployGate | None = None,
    per_class: int = 8,
    note: str = "",
) -> RoundRecord:
    """detect -> classify -> synthesise -> retrain -> evaluate -> gate, recorded.

    `retrain` is optional. Without it the round measures the current adapter and still writes
    a record — useful for establishing the "before" row and for rehearsals, and it is honest
    about not having trained anything.
    """
    gate = gate or RedeployGate()

    flagged = detect_failures(episodes, contracts)
    counts = class_counts(flagged)
    plan = synthesis_plan(counts, per_class=per_class, seed=seed)
    synthesised = sum(len(v) for v in plan.values())

    adapter_out: str | None = None
    if retrain is not None:
        adapter_out = retrain(adapter_in)
        assert_continues_from_current(adapter_in, adapter_out)

    before = dict(evaluate(adapter_in or "current"))
    after = dict(evaluate(adapter_out or adapter_in or "current"))
    verdict = gate.evaluate(before, after)

    record = RoundRecord(
        round_index=round_index,
        counts_by_class=counts,
        synthesised=synthesised,
        verdict=verdict,
        promoted=verdict.promote,
        seed=seed,
        gate_margin=gate.margin,
        trap_tolerance=gate.trap_tolerance,
        adapter_in=adapter_in,
        adapter_out=adapter_out,
        plan=describe_plan(plan),
        flagged_episode_ids=tuple(s.episode_id for s in flagged),
        note=note,
    )
    _append_record(out_dir, record)
    return record


# ------------------------------------------------- model-free evaluation ---------------
def evaluate_with_policy(
    make_policy: Callable[[Any], Policy],
    scenarios: Sequence[Any],
    *,
    episode_prefix: str | None = None,
) -> dict[str, float]:
    """Drive a policy over scenarios through the REAL episode loop and return the gate scores.

    This is a real measurement OF THE POLICY IT IS GIVEN. Passing a scripted route measures the
    scripted route; it says nothing about a model, and a number from here must never be
    reported as a model result.
    """
    episodes: list[Any] = []
    contracts: dict[str, Any] = {}
    negative_ids: set[str] = set()
    for s in scenarios:
        contract = s.contract
        # The contract map is keyed by the id the EPISODE carries, so resolution cannot miss:
        # `eval.metrics.resolve_contract` looks for `episode.contract_id`, then `episode_id`,
        # then `episode_id` with one leading `tag-` stripped. A prefix the resolver does not
        # know about would silently resolve to None and every episode would score `unscored`.
        eid = f"{episode_prefix}-{contract.id}" if episode_prefix else contract.id
        contracts[eid] = contract
        if s.is_negative:
            negative_ids.add(eid)
        world = open_world(int(contract.pre_state["seed"]), dict(contract.pre_state),
                           episode_id=eid)
        try:
            episodes.append(run_episode(world, make_policy(s), contract.brief,
                                        contract.step_budget))
        finally:
            world.close()

    cm = contract_metrics(episodes, contracts, negative_ids=negative_ids or None)
    scores: dict[str, float] = {}
    for key, value in (("verified_success", cm.get("verified_task_success")),
                       ("trap_success", cm.get("trap_success_E3")),
                       ("shortcut_rate", cm.get("shortcut_rate"))):
        if value is not None:
            scores[key] = float(value)
    return scores


# --------------------------------------------------------------------- CLI -------------
def main(argv: list[str] | None = None) -> int:
    """`python -m veriloop.heal.loop --rounds 1` — the `make heal` target.

    Requires recorded episodes. Refuses rather than inventing them: the loop's input is an
    arm's rollouts, and no arm has been trained yet, so there is nothing to heal.
    """
    ap = argparse.ArgumentParser(description="One self-healing round, gated.")
    ap.add_argument("--rounds", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--episodes", default=None,
                    help="jsonl of recorded episodes from an arm (required)")
    ap.add_argument("--contracts", default="contracts")
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--trap-tolerance", type=float, default=0.01)
    ap.add_argument("--per-class", type=int, default=8)
    args = ap.parse_args(argv)

    if not args.episodes or not Path(args.episodes).exists():
        print(
            "make heal: no recorded episodes to learn from.\n"
            "\n"
            "The healing loop consumes an arm's rollouts. No arm has been trained yet, so\n"
            "there is nothing to classify and nothing to drill — this is a scheduling fact,\n"
            "not a bug. Two ways forward:\n"
            "\n"
            "  * run a rehearsal (no model, no GPU, scripted policies):\n"
            "      python scripts/heal_rehearsal.py --out outputs/rehearsal\n"
            "  * or train an arm first, then pass its rollouts:\n"
            "      make sft && make rl ARM=state\n"
            "      python -m veriloop.heal.loop --episodes outputs/A4_state/rollouts.jsonl",
            file=sys.stderr,
        )
        return 2

    raise NotImplementedError(
        "reading an arm's rollouts needs the AgentTune trajectory shape, which does not exist "
        "on this machine yet — see veriloop/eval/harness.py"
    )
