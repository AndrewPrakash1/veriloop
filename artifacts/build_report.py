"""Rebuild the before/after table from `outputs/` and render it to markdown.

Hard rule: a cell with no run behind it prints `unmeasured`. Predicted values go in a
SEPARATE, explicitly labelled column. Never emit a table of constructed or expected numbers
that a reader could mistake for results.

Which arms exist is NOT guessed from the plan. The arm list below is the plan's set, and every
one of them is looked up on disk; an arm with no `eval_results.json` renders as `unmeasured`
rather than being dropped from the table, because an absent row is exactly the thing a reader
needs to see.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from veriloop.eval.report import UNMEASURED, markdown_table

#: The arms PLAN.md §5 pre-registers. A0 is the base model, A1 the SFT adapter, A2/A2b the
#: claim-verified comparison, A3/A4 the state-verified arms under test.
ARMS: tuple[tuple[str, str], ...] = (
    ("A0_base", "base model, no tuning"),
    ("A1_sft", "SFT on gold trajectories"),
    ("A2_claim", "RL, naive claim-verified reward"),
    ("A2b_claim_shaped", "RL, claim-verified + shaping (naive comparison, fully tuned)"),
    ("A3_state", "RL, state-verified, mandatory term only"),
    ("A4_state_shaped", "RL, state-verified + shaping"),
    ("A5_random", "RL, random reward (negative control)"),
)

METRICS: tuple[tuple[str, str], ...] = (
    ("verified_task_success", "up"),
    ("mandatory_step_recall", "up"),
    ("shortcut_rate", "down"),
    ("forbidden_action_rate", "down"),
    ("false_block_rate", "down"),
    ("trap_success_E3", "up"),
)


def load_arm(outputs: Path, arm: str) -> dict[str, Any] | None:
    """Read one arm's `eval_results.json`, or None when the arm has not been run."""
    for name in ("eval_results.json", "eval/eval_results.json"):
        p = outputs / arm / name
        if p.exists():
            try:
                return json.loads(p.read_text())
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{p} is present but not valid JSON: {exc}")
    return None


def load_heal_rounds(outputs: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(outputs.rglob("rounds.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build(outputs: Path) -> str:
    parts: list[str] = []
    measured = 0

    rows: list[dict[str, Any]] = []
    for arm, description in ARMS:
        data = load_arm(outputs, arm)
        if data is None:
            rows.append({"metric": f"`{arm}`", "before": None, "after": None,
                         "note": f"no run — {description}"})
            continue
        measured += 1
        for key, direction in METRICS:
            value = (data.get(key) if isinstance(data, dict) else None)
            if value is None and isinstance(data, dict):
                value = (data.get("metrics") or {}).get(key)
            rows.append({"metric": f"`{arm}` · {key}", "before": value, "after": None,
                         "direction": direction,
                         "note": "" if value is not None else "missing from eval_results.json"})
    parts.append(markdown_table(rows, title="Arms", columns=("before",)))

    heal = load_heal_rounds(outputs)
    if heal:
        hrows = []
        for r in heal:
            v = r.get("verdict") or {}
            scores = v.get("scores") or {}
            delta = (scores.get("delta") or {})
            hrows.append({
                "metric": f"round {r.get('round_index')}",
                "before": (scores.get("before") or {}).get("verified_success"),
                "after": (scores.get("after") or {}).get("verified_success"),
                "delta": delta.get("verified_success"),
                "direction": "up",
                "note": ("PROMOTED" if r.get("promoted") else "rejected")
                        + (f" — {v.get('reason', '')[:80]}" if not r.get("promoted") else ""),
            })
        parts.append(markdown_table(hrows, title="Self-healing rounds"))

    header = (f"{measured} of {len(ARMS)} arms have a run behind them. "
              f"`{UNMEASURED}` means no run, not a zero.")
    return "\n\n".join([header, *parts]) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rebuild the before/after table from outputs/.")
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--out", default="artifacts/REPORT.md")
    args = ap.parse_args(argv)

    text = build(Path(args.outputs))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text)
    print(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
