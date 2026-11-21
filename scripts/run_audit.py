"""THE pre-GPU artifact: score every attack under both rewards and write the audit JSON.

    make audit          # -> audit/reward_hackability.json

Needs no GPU, no model, no API key and no network: each attack is a hand-authored route
scored directly by the reward function, on a FRESH world, through the real episode loop.

The output shape is consumed by `demo/server.py`, which reads `.rows[]` and the keys
`attack_id`, `claim_verified`, `state_verified`, `closed`. Miss one and the demo silently
keeps showing "unmeasured" — a failure that looks like nothing happening.

The exit code is non-zero when the design is broken, so this can gate a commit:
  * any non-trivial attack scoring above CHEAT_MAX under the proposed reward
  * the control not scoring exactly 1.0
  * the naive reward failing to pay a cheat, in which case the comparison demonstrates
    nothing about the naive reward
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veriloop.rewards.probes import (  # noqa: E402
    CHEAT_MAX,
    load_attacks,
    scenario_path,
    score_attacks,
    summarise,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Reward hackability audit (no GPU, no model).")
    ap.add_argument("--catalogue", default=None, help="attack catalogue YAML")
    ap.add_argument("--scenario", default=None, help="episode contract every row is scored on")
    ap.add_argument("--out", default="audit/reward_hackability.json")
    args = ap.parse_args(argv)

    attacks = load_attacks(args.catalogue)
    rows = score_attacks(attacks, args.scenario)
    summary = summarise(rows)

    width = max(len(r["attack_id"]) for r in rows) + 2
    print(f"scenario: {scenario_path(args.catalogue)}")
    print(f"{'attack':<{width}} {'naive':>7} {'state':>8}  {'closed':<7} probes")
    print("-" * (width + 42))
    for r in sorted(rows, key=lambda r: (not r["non_trivial"], r["attack_id"])):
        flag = "control" if not r["non_trivial"] else ("yes" if r["closed"] else "NO")
        print(f"{r['attack_id']:<{width}} {r['claim_verified']:>7.2f} "
              f"{r['state_verified']:>8.3f}  {flag:<7} {r['probes']}")

    print()
    print(f"non-trivial attacks       : {summary['n_non_trivial']}")
    print(f"closed under the proposed : {summary['closed_non_trivial']}")
    print(f"control scores exactly 1.0: {summary['control_scores_maximum']}")
    print(f"naive pays every cheat    : {summary['naive_pays_every_cheat']}")
    if summary["open_holes"]:
        print(f"\nOPEN HOLES (above {CHEAT_MAX}): "
              f"{json.dumps(summary['open_holes'], indent=2)}")
    if summary["attacks_not_paid_by_the_naive_reward"]:
        print("\nThe naive reward does NOT pay these, so they demonstrate nothing about it: "
              f"{summary['attacks_not_paid_by_the_naive_reward']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "scenario": str(scenario_path(args.catalogue)),
        "catalogue": str(args.catalogue or "audit/attacks.yaml"),
        "cheat_max": CHEAT_MAX,
        "summary": summary,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")

    failures = []
    if summary["open_hole_count"]:
        failures.append(f"{summary['open_hole_count']} open hole(s)")
    if not summary["control_scores_maximum"]:
        failures.append(f"control scored {summary['control_state_verified']}, not 1.0")
    if not summary["naive_pays_every_cheat"]:
        failures.append("the naive reward does not pay every cheat")
    if failures:
        print("\nAUDIT FAILED: " + "; ".join(failures))
        return 1
    print(f"\nAUDIT PASSED: {summary['closed_non_trivial']} non-trivial attacks closed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
