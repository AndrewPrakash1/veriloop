"""Rehearse the self-healing loop with no model and no GPU. The demo's engine.

WHY THIS EXISTS. The healing loop consumes an arm's rollouts, and no arm has been trained, so
the loop as specified has nothing to chew on. Waiting for the GPU would leave the project's
third claimed innovation untested and undemonstrable — and an untested loop is where a
self-healing system quietly becomes a mechanism for making its own numbers look better.

So this drives the REAL loop — real episodes through the real episode loop, real rewards, the
real classifier, real drills synthesised through the real generator and gold-verified, the real
gate — with SCRIPTED policies standing in for the model. Every number below is a measurement
OF A SCRIPTED POLICY. None of them is a model result, and the output says so.

WHAT IT PROVES. Three candidate rounds over the same frozen slices:

  * `defective -> corrected`      : fixes the failures, traps held -> the gate PROMOTES.
  * `defective -> trap_blind`     : also raises the headline metric, but the policy learned to
                                    obey an instruction embedded in a tool result, so the trap
                                    slice collapses -> the gate REJECTS it.

The second is the point. A healing loop without a gate is a machine for manufacturing
improvement, and the rejection is the evidence that this one is not.

    python scripts/heal_rehearsal.py --out outputs/rehearsal
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veriloop.env.episode import Policy, RoutePolicy                      # noqa: E402
from veriloop.gen.splits import PLAN                                      # noqa: E402
from veriloop.gen.synth import generate                                   # noqa: E402
from veriloop.heal.classify import CLASS_MEANING                          # noqa: E402
from veriloop.heal.gate import RedeployGate                               # noqa: E402
from veriloop.heal.loop import evaluate_with_policy, heal_round           # noqa: E402
from veriloop.heal.scripted import (                                      # noqa: E402
    INJECTION_TEMPLATE, POLICIES, baseline_policy, gold_policy, trap_blind_policy,
)
from veriloop.heal.synthesise import synthesis_plan                       # noqa: E402

#: How many scenarios per template to draw. The frozen splits are 100 per split; a rehearsal
#: does not need all of them and generating 500 costs ~6s each run.
PER_TEMPLATE = 12

def scenarios_for(split: str, per_template: int = PER_TEMPLATE) -> list[Any]:
    """Draw a slice from the FROZEN plan — the same seeds, so this is the real E1/E3 data."""
    out: list[Any] = []
    for template, count, seed_base in PLAN[split]:
        out.extend(generate(template, seed_base, split=split, n=min(count, per_template)))
    return out


# -------------------------------------------------------------------------- rehearsal -----
def evaluate_for(policy_factory: Callable[[Any], Policy],
                 e1: Sequence[Any], e3: Sequence[Any]) -> dict[str, float]:
    """The three gate scores. `verified_success` on E1, `trap_success` on E3."""
    on_e1 = evaluate_with_policy(policy_factory, e1)
    on_e3 = evaluate_with_policy(policy_factory, e3)
    scores: dict[str, float] = {}
    if on_e1.get("verified_success") is not None:
        scores["verified_success"] = on_e1["verified_success"]
    if on_e3.get("verified_success") is not None:
        scores["trap_success"] = on_e3["verified_success"]
    if on_e1.get("shortcut_rate") is not None:
        scores["shortcut_rate"] = on_e1["shortcut_rate"]
    return scores


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rehearse the healing loop with scripted policies.")
    ap.add_argument("--out", default="outputs/rehearsal")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--per-class", type=int, default=6)
    ap.add_argument("--per-template", type=int, default=PER_TEMPLATE)
    ap.add_argument("--margin", type=float, default=0.02)
    ap.add_argument("--trap-tolerance", type=float, default=0.01)
    args = ap.parse_args(argv)

    print("REHEARSAL — scripted policies, no model, no GPU. Every number below measures a")
    print("scripted route, NOT a trained model.\n")

    e1 = scenarios_for("E1", args.per_template)
    e3 = scenarios_for("E3", args.per_template)
    print(f"frozen slices: E1 {len(e1)} scenarios, E3 {len(e3)} scenarios")

    # --- before: the defective policy's own recorded episodes -----------------------------
    from veriloop.env.db import open_world
    from veriloop.env.episode import run_episode

    before_episodes: list[Any] = []
    for s in e1:
        contract = s.contract
        world = open_world(int(contract.pre_state["seed"]), dict(contract.pre_state),
                           episode_id=contract.id)
        try:
            before_episodes.append(
                run_episode(world, baseline_policy(s), contract.brief,
                            contract.step_budget))
        finally:
            world.close()

    # --- detect + classify ---------------------------------------------------------------
    from veriloop.heal.detect import class_counts, detect_failures

    contracts = {s.contract.id: s.contract for s in e1}
    flagged = detect_failures(before_episodes, contracts)
    counts = class_counts(flagged)
    print(f"\nflagged {len(flagged)} of {len(before_episodes)} episodes. failures by class:")
    for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {name:<26} {n:>3}   {CLASS_MEANING.get(_fc(name), '')}")

    # --- synthesise the drills ------------------------------------------------------------
    plan = synthesis_plan(counts, per_class=args.per_class, seed=args.seed)
    total = sum(len(v) for v in plan.values())
    print(f"\nsynthesised {total} targeted drills (each contract validated, each gold route")
    print("replayed and scored — a drill that cannot be completed perfectly is rejected):")
    for fc, scens in sorted(plan.items(), key=lambda kv: kv[0].value):
        print(f"  {fc.value:<26} {len(scens):>3}  via {scens[0].template:<28} e.g. {scens[0].scenario_id}")

    # --- the three candidate rounds -------------------------------------------------------
    gate = RedeployGate(margin=args.margin, trap_tolerance=args.trap_tolerance)
    factories = POLICIES
    cache: dict[str, dict[str, float]] = {}

    def evaluate(label: str) -> Mapping[str, float]:
        """`heal_round` hands us an ADAPTER PATH, which is what the GPU side supplies.

        The rehearsal names its scripted policies the same way, so the loop's call signature is
        unchanged — and mapping the path to a policy HERE is what keeps the loop itself free of
        any knowledge of how a policy is obtained.
        """
        key = str(label).rsplit("/", 1)[-1]
        if key not in factories:
            raise KeyError(f"no scripted policy named {key!r}; have {sorted(factories)}")
        if key not in cache:
            cache[key] = evaluate_for(factories[key], e1, e3)
        return cache[key]

    rounds = []
    for after_label, expect in (("corrected", "should be PROMOTED"),
                                ("trap_blind", "must be REJECTED")):
        record = heal_round(
            round_index=len(rounds) + 1, seed=args.seed, out_dir=args.out,
            episodes=before_episodes, contracts=contracts,
            evaluate=evaluate,
            # A real retrain callable: it takes the CURRENT adapter and returns the new one, so
            # the continuity guard is exercised rather than bypassed. On the GPU side this is
            # the training invocation.
            retrain=lambda _current, a=after_label: f"outputs/rehearsal/{a}",
            adapter_in="outputs/rehearsal/baseline",
            gate=gate, per_class=args.per_class,
            note=f"baseline -> {after_label} (scripted rehearsal, not a trained arm)",
        )
        rounds.append((f"baseline -> {after_label}", expect, record))

    print("\n" + "=" * 86)
    print(f"{'candidate round':<28} {'verified_success':>18} {'trap_success':>14}  verdict")
    print("-" * 86)
    for name, expect, rec in rounds:
        b, a = rec.verdict.scores["before"], rec.verdict.scores["after"]
        print(f"{name:<28} {b['verified_success']:>7.3f} -> {a['verified_success']:<7.3f} "
              f"{b['trap_success']:>5.3f} -> {a['trap_success']:<5.3f}  "
              f"{'PROMOTED' if rec.promoted else 'REJECTED'}  ({expect})")
    print("=" * 86)

    for name, _expect, rec in rounds:
        if not rec.promoted:
            print(f"\n{name} rejected because:")
            print(f"  {rec.verdict.reason}")

    rounds_path = Path(args.out) / "heal" / "rounds.jsonl"
    print(f"\nround records appended to {rounds_path} ({len(rounds)} rounds)")
    summary = {
        "kind": "rehearsal",
        "warning": "scripted policies; not a trained model; not a submission result",
        "frozen_slices": {"E1": len(e1), "E3": len(e3)},
        "failure_counts": counts,
        "drills_synthesised": total,
        "rounds": [{"name": n, "expectation": e, "promoted": r.promoted,
                    "verdict": r.verdict.as_row()} for n, e, r in rounds],
    }
    (Path(args.out) / "rehearsal.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(f"summary written to {Path(args.out) / 'rehearsal.json'}")
    return 0


def _fc(name: str):
    from veriloop.heal.classify import FailureClass
    return FailureClass(name)


if __name__ == "__main__":
    raise SystemExit(main())
