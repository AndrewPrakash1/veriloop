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
              step_budget: int | None = None, adapter: str | None = None,
              model_id: str | None = None, temperature: float | None = None,
              seed: int | None = None) -> dict:
    """Record and score one arm over one split. Returns the batch summary.

    `adapter` is what makes A1 scorable at all. Without it every arm runs the BASE model, so a
    before/after table would compare A0 with itself and still produce a tidy number.
    """
    from veriloop.contracts.schema import Contract
    from veriloop.env.db import open_world
    from veriloop.env.episode import run_episode
    from veriloop.eval.metrics import contract_metrics, tool_metrics
    from veriloop.eval.oracle import replay_and_assert
    from veriloop.rewards.state_verified import reward_from_terms
    from veriloop.rewards.terms import compute_terms

    from agents.loader import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE, build_policy

    scenarios = load_scenarios(split_path, limit)
    os.makedirs(out_dir, exist_ok=True)
    trajectories_path = os.path.join(out_dir, "trajectories.jsonl")
    rows: list[dict] = []
    episodes: list[Any] = []
    contracts: dict[str, Contract] = {}
    negative_ids: set[str] = set()

    resolved_temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
    policy = build_policy(
        step_budget or 12,
        model_id=model_id or DEFAULT_MODEL_ID,
        adapter=adapter,
        temperature=resolved_temperature,
        seed=seed,
    )

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
        # Provenance: WHICH weights produced these rows. A before/after table whose two rows
        # cannot be told apart from its own JSON is not evidence — and an adapter forgotten on
        # the command line looks exactly like A1 scoring well.
        "model_id": model_id or DEFAULT_MODEL_ID,
        "adapter": adapter,
        "temperature": resolved_temperature,
        "seed": seed,
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


def _parser() -> argparse.ArgumentParser:
    """The CLI. Exposed as a function so a test can assert its defaults without a GPU.

    `--adapter` is the flag that makes an A0-vs-A1 comparison possible at all; without it
    `train/sft.py` could produce a checkpoint that nothing could ever score.
    """
    from agents.loader import DEFAULT_MODEL_ID, DEFAULT_TEMPERATURE

    parser = argparse.ArgumentParser(
        description="Record and score one arm over one split.",
        epilog=(
            "A0:  --model-label A0\n"
            "A1:  --model-label A1 --adapter outputs/A1      (the SFT checkpoint)\n"
            "Both arms must use the same --temperature and --seed, or the difference between "
            "them is not attributable to the training."
        ),
    )
    parser.add_argument("--split", required=True, help="path to a split JSONL")
    parser.add_argument("--out", required=True, help="output directory for this arm")
    parser.add_argument("--model-label", default="A0", help="arm label recorded in the rows")
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios only")
    parser.add_argument("--step-budget", type=int, default=None, help="override the contract's")
    parser.add_argument(
        "--adapter",
        default=None,
        help="path to a LoRA/PEFT checkpoint to score instead of the base model (this is A1)",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="HF repo id")
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="sampling temperature; identical across arms or the comparison is void",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="sampling seed; fixed by default so a rerun matches"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """`python scripts/run_baseline.py --split data/splits/E1.jsonl --out outputs/A0`."""
    args = _parser().parse_args(argv)

    summary = run_split(args.split, args.out, model_label=args.model_label,
                        limit=args.limit, step_budget=args.step_budget,
                        adapter=args.adapter, model_id=args.model_id,
                        temperature=args.temperature, seed=args.seed)
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
