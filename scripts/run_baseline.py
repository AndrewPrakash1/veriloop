"""Arm A0 over E1-E4 against a served model, plus the failure-mode histogram.

The histogram is the input to the healing loop AND the "before" evidence, so produce it
even if nothing else downstream lands.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from typing import Any

# `python3 scripts/run_baseline.py` puts SCRIPTS/ on sys.path[0], not the repo root, so this
# import needs the repo root put back. Same shim as `scripts/verify_world.py`, derived from
# this file's location so it does not depend on the cwd.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Failure names come from the healing loop's enum, not from this script. The histogram is
#: Phase 7's INPUT, and a second set of names would not join up with it. `classify()` replaces
#: the rule set below once it lands; the names are already shared.
from veriloop.heal.classify import FailureClass


def load_scenarios(path: str, limit: int | None = None) -> list[dict]:
    """Read a split file. Each row is a scenario: contract, pre_state, gold_route, is_negative."""
    rows: list[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def failure_class_of(row: dict, terminated: bool) -> str:
    """Name the single dominant failure of one episode.

    Priority is deliberate and matches `heal/classify.py`: a malformed call beats everything
    (it changes what the later diagnostics even mean), then a forbidden call, then the rest.
    A clean episode is `success`; an unterminated one is `unfinished`.
    """
    if row["schema_invalid_calls"]:
        return FailureClass.SCHEMA_MISMATCH.value
    if row["forbidden_seen"]:
        return FailureClass.FORBIDDEN_ORDER.value
    if not terminated:
        return "unfinished"
    if row["M"] < 1.0:
        return FailureClass.SKIPPED_MANDATORY_STEP.value
    if row["oracle_verified"]:
        return "success"
    return "wrong_final_state"


def run_split(split_path: str, out_dir: str, *, model_label: str, limit: int | None = None,
              step_budget: int | None = None) -> dict:
    """Record and score one arm over one split. Returns the batch summary."""
    from veriloop.contracts.schema import Contract
    from veriloop.env.db import open_world
    from veriloop.env.episode import run_episode
    from veriloop.eval.metrics import contract_metrics, tool_metrics
    from veriloop.eval.oracle import replay_and_assert
    from veriloop.rewards.state_verified import reward_from_terms
    from veriloop.rewards.terms import compute_terms

    from agents.loader import build_policy

    scenarios = load_scenarios(split_path, limit)
    os.makedirs(out_dir, exist_ok=True)
    trajectories_path = os.path.join(out_dir, "trajectories.jsonl")
    rows: list[dict] = []
    episodes: list[Any] = []
    contracts: dict[str, Contract] = {}
    negative_ids: set[str] = set()

    policy = build_policy(step_budget or 12)

    with open(trajectories_path, "w") as traj:
        for scenario in scenarios:
            contract = Contract(**scenario["contract"])
            pre = dict(contract.pre_state)
            budget = step_budget or contract.step_budget
            world = open_world(pre["seed"], pre, episode_id=f"{model_label}-{contract.id}")
            try:
                started = time.time()
                result = run_episode(world, policy, contract.brief, budget, agent_name=model_label)
                elapsed = time.time() - started
                terms = compute_terms(contract, result.log, result.snapshot,
                                      terminated=result.terminated, steps=result.steps)
                reward = reward_from_terms(terms)
                oracle = replay_and_assert(
                    contract,
                    [{"tool": r.tool, "args": dict(r.args)} for r in result.log.records],
                    pre["seed"],
                )
                record = {
                    "scenario_id": scenario["scenario_id"],
                    "contract_id": contract.id,
                    "split": contract.split,
                    "is_negative": bool(scenario.get("is_negative")),
                    "reward": reward,
                    "terms": terms.as_row(),
                    "oracle": {"verified": oracle["verified"], "failures": oracle["failures"]},
                    "steps": result.steps,
                    "terminated": bool(result.terminated),
                    "seconds": round(elapsed, 3),
                    "final_message": result.final_message,
                    "calls": [
                        {"step": r.step, "tool": r.tool, "args": dict(r.args),
                         "ok": r.ok, "schema_valid": r.schema_valid}
                        for r in result.log.records
                    ],
                    "completions": list(getattr(policy, "completions", [])),
                }
                traj.write(json.dumps(record, default=str) + "\n")
            finally:
                world.close()

            if getattr(policy, "completions", None):
                policy.completions.clear()
            rows.append({
                "scenario_id": scenario["scenario_id"],
                "schema_invalid_calls": sum(1 for r in result.log.records if not r.schema_valid),
                "forbidden_seen": bool(terms.forbidden_seen),
                "M": terms.M,
                "oracle_verified": bool(oracle["verified"]),
                "reward": reward,
                "terminated": bool(result.terminated),
                "seconds": round(elapsed, 3),
            })
            episodes.append(result)
            contracts[contract.id] = contract
            if scenario.get("is_negative"):
                negative_ids.add(contract.id)

    histogram = Counter(failure_class_of(r, r["terminated"]) for r in rows)
    summary = {
        "arm": model_label,
        "split": os.path.basename(split_path),
        "n": len(rows),
        # Wall-clock across the batch, summed per episode. This is the number that converts
        # into "an arm costs N hours and therefore Rs N" — read it before starting a long run.
        "seconds": round(sum(r["seconds"] for r in rows), 3),
        "seconds_per_episode": round(sum(r["seconds"] for r in rows) / len(rows), 3) if rows else None,
        "failure_histogram": dict(histogram.most_common()),
        "contract_metrics": contract_metrics(episodes, contracts, negative_ids=negative_ids),
        "tool_metrics": tool_metrics(episodes),
        "mean_reward": (sum(r["reward"] for r in rows) / len(rows)) if rows else None,
    }
    with open(os.path.join(out_dir, "eval_results.json"), "w") as handle:
        json.dump(summary, handle, indent=2, default=str)
    return summary


def print_summary(summary: dict) -> None:
    metrics = summary["contract_metrics"]
    print(f"\n=== {summary['arm']} over {summary['split']} — {summary['n']} episodes ===")
    for name, direction in (metrics.get("direction") or {}).items():
        value = metrics.get(name)
        arrow = "up" if direction == "up" else "down"
        shown = "unmeasured" if value is None else f"{value:.3f}"
        print(f"  {name:24s} {shown:>9s}   ({arrow} is better)")
    print(f"  false-block denominator   {metrics.get('false_block_denominator')}"
          f"   source: {metrics.get('false_block_source')}")
    print(f"  mean reward                {summary['mean_reward']}")
    print(f"  failure histogram          {summary['failure_histogram']}")


def main(argv: list[str] | None = None) -> int:
    """`python scripts/run_baseline.py --split data/splits/E1.jsonl --out outputs/A0`."""
    parser = argparse.ArgumentParser(description="Record and score one arm over one split.")
    parser.add_argument("--split", required=True, help="path to a split JSONL")
    parser.add_argument("--out", required=True, help="output directory for this arm")
    parser.add_argument("--model-label", default="A0", help="arm label recorded in the rows")
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios only")
    parser.add_argument("--step-budget", type=int, default=None, help="override the contract's")
    args = parser.parse_args(argv)

    summary = run_split(args.split, args.out, model_label=args.model_label,
                        limit=args.limit, step_budget=args.step_budget)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
