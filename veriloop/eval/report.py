"""Render the scoreboard as markdown for the write-up.

Rule: an unmeasured cell prints `unmeasured` and the prediction is rendered in a SEPARATE
labelled column. Never a table of constructed or expected values a reader could mistake
for a run.

The rule is mechanical, not editorial: `_cell` has no branch that can turn a missing number
into a number, and a projection only ever appears under the header `Predicted` with the row's
own `prediction_source`. A reviewer can therefore tell, from the table alone, which cells came
from a run.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DIRECTION = {"up": "↑", "down": "↓"}

#: The value that stands in for "no run behind this cell". A constant so nothing can invent a
#: friendlier spelling of it in one place and not another.
UNMEASURED = "unmeasured"


def _cell(value: Any) -> str:
    """Format one measurement. There is deliberately no default that produces a number."""
    if value is None:
        return UNMEASURED
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _delta_cell(value: Any, direction: str | None) -> str:
    if value is None:
        return UNMEASURED
    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)
    if num == 0:
        return "0.000"
    arrow = DIRECTION.get(direction or "", "")
    return f"{num:+.3f}{arrow}"


def markdown_table(rows: Sequence[Mapping[str, Any]], *, title: str | None = None,
                   columns: Sequence[str] = ("before", "after")) -> str:
    """Render rows as a markdown table.

    Each row: `metric`, `direction` ("up"/"down"), the measurement columns, optional `delta`,
    optional `prediction` + `prediction_source`, optional `note`.

    A `prediction` always renders in its own `Predicted` column, and the column is only emitted
    when at least one row carries one — so a table with no predictions does not advertise an
    empty column, and a table WITH predictions cannot hide them next to a measurement.
    """
    has_prediction = any(r.get("prediction") is not None for r in rows)
    header = ["Metric", *[c.capitalize() for c in columns]]
    if "delta" in columns or any(r.get("delta") is not None for r in rows):
        if "delta" not in columns:
            header.append("Δ")
    if has_prediction:
        header.append("Predicted (not a run)")
    if any(r.get("note") for r in rows):
        header.append("Note")

    lines = [f"### {title}" if title else "", ""] if title else []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for r in rows:
        cells = [str(r.get("metric", ""))]
        for c in columns:
            cells.append(_cell(r.get(c)))
        if "delta" in header:
            cells.append(_delta_cell(r.get("delta") if "delta" in r else _auto_delta(r),
                                     r.get("direction")))
        if has_prediction:
            pred = r.get("prediction")
            if pred is None:
                cells.append("—")
            else:
                src = r.get("prediction_source")
                cells.append(f"{_cell(pred)}" + (f" ({src})" if src else ""))
        if "Note" in header:
            cells.append(str(r.get("note") or ""))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(line for line in lines if line is not None)


def _auto_delta(row: Mapping[str, Any]) -> Any:
    before, after = row.get("before"), row.get("after")
    if before is None or after is None:
        return None
    try:
        return float(after) - float(before)
    except (TypeError, ValueError):
        return None


def significance_note(p_value: float | None, n: int, *, seed: int | None = None) -> str:
    """One line a reader can check. `n` is REQUIRED: a p-value without its n is not evidence."""
    if p_value is None:
        return f"{UNMEASURED} (n={n})"
    verdict = "significant" if p_value < 0.05 else "not significant"
    tail = f", seed {seed}" if seed is not None else ""
    return f"p = {p_value:.4f} ({verdict}), n = {n}{tail}"
