"""Render the scoreboard as markdown for the write-up.

Rule: an unmeasured cell prints `unmeasured` and the prediction is rendered in a SEPARATE
labelled column. Never a table of constructed or expected values a reader could mistake
for a run.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

DIRECTION = {"up": "↑", "down": "↓"}


def markdown_table(rows: Sequence[Mapping[str, Any]]) -> str:  # type: ignore[name-defined]
    raise NotImplementedError("Phase 4")
