"""Statistics. Pre-registered, paired, and honest about the resolution limit.

  * Unit of analysis is the EPISODE, not the token and not the step.
  * Paired on the same scenario set: exact McNemar per seed, paired bootstrap across seeds.
  * 10,000 resamples, fixed seed (42), Holm correction across the pre-registered contrasts.
  * Report the step-count checkpoint, not the best checkpoint.
  * Never a mean without its across-seed spread; three seeds estimate seed variance poorly
    and that must be written down, not hidden.
  * The resolution limit is stated up front: at n = 300 and 50% accuracy the standard
    error is ~2.9 points, so an unpaired gap under ~6 points cannot be resolved.
"""

from __future__ import annotations

from typing import Any, Sequence

#: Named before running, so the analysis is not a fishing trip.
PRE_REGISTERED_CONTRASTS: Sequence[tuple[str, str, str]] = (
    ("A4", "A2", "verified success and shortcut rate under shaped vs naive reward"),
    ("A4", "A5", "attribution: the verifier, not spurious RLVR gain"),
    ("A2", "A5", "does anything non-random help at all"),
    ("A6", "A4", "does the self-healing loop add anything"),
)


def paired_bootstrap(before: dict, after: dict, n_resamples: int = 10_000, seed: int = 42) -> dict:
    raise NotImplementedError("Phase 4")


def mcnemar(before: dict, after: dict) -> dict:
    raise NotImplementedError("Phase 4")


def compare(results: dict[str, dict]) -> dict:
    """The before/after table: per metric, per contrast, with CI, p-value, fixed/broke."""
    raise NotImplementedError("Phase 4")
