"""Rebuild the before/after table from `outputs/` and render it to markdown.

Hard rule: a cell with no run behind it prints `unmeasured`. Predicted values go in a
SEPARATE, explicitly labelled column. Never emit a table of constructed or expected
numbers that a reader could mistake for results.
"""

from __future__ import annotations


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 4")


if __name__ == "__main__":
    raise SystemExit(main())
