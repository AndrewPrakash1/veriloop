"""Flag episodes worth learning from: gate failures and budget blowouts."""

from __future__ import annotations

from typing import Any


def detect_failures(episodes: list[Any], contracts: dict, threshold: float = 0.0) -> list[Any]:
    """Return episodes whose reward is at or below `threshold`, or that exhausted budget."""
    raise NotImplementedError("Phase 7")
