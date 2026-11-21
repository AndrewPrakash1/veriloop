"""Statistics. Pre-registered, paired, and honest about the resolution limit.

  * Unit of analysis is the EPISODE, not the token and not the step.
  * Paired on the same scenario set: exact McNemar per seed, paired bootstrap across seeds.
  * 10,000 resamples, fixed seed (42), Holm correction across the pre-registered contrasts.
  * Report the step-count checkpoint, not the best checkpoint.
  * Never a mean without its across-seed spread; three seeds estimate seed variance poorly
    and that must be written down, not hidden.
  * The resolution limit is stated up front: at n = 300 and 50% accuracy the standard
    error is ~2.9 points, so an unpaired gap under ~6 points cannot be resolved.

HOW THE INPUTS ARE SHAPED

Every function here takes `{episode_id: outcome}` for ONE metric and ONE arm — a 0/1 for a
success or claim metric, or a float for a rate. Keying by episode is what makes the
comparison PAIRED: it compares the same scenarios under two arms, so a difference in
scenario difficulty cancels instead of inflating the variance. Two arms are only compared on
the episodes present in BOTH, and the number dropped is reported rather than quietly ignored.

WHY THE BOOTSTRAP IS IMPLEMENTED HERE IN STDLIB

`auditkit` supplies `paired_bootstrap`, and this is a re-implementation of it, deliberately:
the headline contrast must not become uncomputable because a dependency failed to import on
the machine that has to produce the number. It is twenty lines of `random.Random` and
`math.comb`, it is seeded, and it is checked against hand-computed exact values in
`tests/test_eval.py`. The AuditKIT figure is still the one that goes in the write-up, and the
two agreeing is another independent check — the same argument as `eval/oracle.py`.
"""

from __future__ import annotations

import math
import random
from typing import Any, Mapping, Sequence

#: Named before running, so the analysis is not a fishing trip.
PRE_REGISTERED_CONTRASTS: Sequence[tuple[str, str, str]] = (
    ("A4", "A2", "verified success and shortcut rate under shaped vs naive reward"),
    ("A4", "A5", "attribution: the verifier, not spurious RLVR gain"),
    ("A2", "A5", "does anything non-random help at all"),
    ("A6", "A4", "does the self-healing loop add anything"),
)

DEFAULT_RESAMPLES = 10_000
DEFAULT_SEED = 42

#: The plan's stated resolution rule, as a constant rather than a sentence somebody has to
#: remember: an unpaired gap smaller than twice the standard error is not a finding. Pairing
#: buys more resolution than this, which is exactly why the paired CI is the decider.
_RESOLUTION_Z = 2.0


def standard_error(n: int, p: float = 0.5) -> float:
    """Standard error of a proportion, at the worst case `p` by default."""
    if n <= 0:
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def resolution_limit(n: int, p: float = 0.5) -> float:
    """The smallest unpaired gap this sample size can resolve, as a fraction.

    At n = 300 this is ~0.058, about six points — the number the plan commits to stating up
    front rather than discovering after the fact.
    """
    return _RESOLUTION_Z * standard_error(n, p)


def _paired(
    a: Mapping[Any, Any], b: Mapping[Any, Any]
) -> tuple[list[Any], list[float], list[float]]:
    """Episode ids present in both arms, and the two aligned value lists."""
    shared = sorted(set(a) & set(b), key=str)
    return shared, [float(a[k]) for k in shared], [float(b[k]) for k in shared]


def paired_bootstrap(before: dict, after: dict, n_resamples: int = 10_000, seed: int = 42) -> dict:
    """Paired bootstrap CI for `after - before`, plus a two-sided bootstrap p-value.

    Resampling is over EPISODES, with the same index drawn for both arms — that is what makes
    it paired. The p-value is the two-sided bootstrap tail: twice the smaller share of
    resampled deltas on the wrong side of zero, capped at 1.0. It is a bootstrap p, not an
    exact test; `mcnemar` is the exact one for 0/1 outcomes, and both are reported.
    """
    shared, a_vals, b_vals = _paired(before, after)
    n = len(shared)
    if n == 0:
        return {
            "n": 0,
            "episodes_dropped": len(set(before) ^ set(after)),
            "mean_before": None,
            "mean_after": None,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "p_value": None,
            "n_resamples": n_resamples,
            "seed": seed,
            "resolution_limit": None,
            "underpowered": None,
        }

    deltas = [b - a for a, b in zip(a_vals, b_vals)]
    point = sum(deltas) / n

    rng = random.Random(seed)
    boot: list[float] = []
    for _ in range(n_resamples):
        idx = rng.choices(range(n), k=n)
        boot.append(sum(deltas[i] for i in idx) / n)
    boot.sort()

    low = boot[int(0.025 * n_resamples)]
    high = boot[min(n_resamples - 1, int(0.975 * n_resamples))]
    left = sum(1 for d in boot if d <= 0.0) / n_resamples
    right = sum(1 for d in boot if d >= 0.0) / n_resamples

    return {
        "n": n,
        "episodes_dropped": len(set(before) ^ set(after)),
        "mean_before": sum(a_vals) / n,
        "mean_after": sum(b_vals) / n,
        "delta": point,
        "ci_low": low,
        "ci_high": high,
        "p_value": min(1.0, 2.0 * min(left, right)),
        "n_resamples": n_resamples,
        "seed": seed,
        "resolution_limit": resolution_limit(n),
        # The CI, not the resolution rule, decides this — pairing resolves more than the
        # unpaired rule assumes. The flag exists so a null result is never read as a measured
        # absence of an effect when the batch was simply too small.
        "underpowered": not (low > 0.0 or high < 0.0),
    }


def _binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial tail: 2 * P(X <= min(k, n - k)) for X ~ Binomial(n, p)."""
    if n == 0:
        return 1.0
    tail = sum(
        math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
        for i in range(0, min(k, n - k) + 1)
    )
    return min(1.0, 2.0 * tail)


def mcnemar(before: dict, after: dict) -> dict:
    """Exact McNemar over the discordant pairs. No SciPy, no normal approximation.

    Only the pairs that CHANGED carry information, and both directions matter: a round that
    fixes 40 episodes while breaking 35 is not an improvement, and the exact test says so.
    """
    shared, a_vals, b_vals = _paired(before, after)
    fixed = sum(1 for a, b in zip(a_vals, b_vals) if a == 0.0 and b == 1.0)
    broke = sum(1 for a, b in zip(a_vals, b_vals) if a == 1.0 and b == 0.0)
    discordant = fixed + broke

    return {
        "n": len(shared),
        "fixed": fixed,
        "broke": broke,
        "n_discordant": discordant,
        "net": fixed - broke,
        "p_value": _binomial_two_sided(min(fixed, broke), discordant),
        "method": "exact_two_sided_binomial",
    }


def holm(pvalues: Mapping[str, float]) -> dict:
    """Holm step-down correction. Family-wise error control over the pre-registered set.

    Holm rather than Bonferroni because it is uniformly at least as powerful and no harder to
    compute; the ordering is by p-value, and the adjusted values are made monotone so a reader
    cannot see a smaller raw p with a larger adjusted one.
    """
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    adjusted: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        value = min(1.0, max(running, (m - i) * p))
        adjusted[key] = value
        running = value
    return adjusted


def compare(results: dict[str, dict]) -> dict:
    """The before/after table: per metric, per contrast, with CI, p-value, fixed/broke.

    `results` is `{arm: {metric: {episode_id: outcome}}}`. Only the PRE-REGISTERED contrasts
    are run — the analysis was fixed before the runs, so a contrast that is not in that tuple
    cannot be added here without the correction above having been designed for a different
    family. A contrast whose arms are not both present is reported under `skipped` with the
    reason; it is never filled in with a fabricated row.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    pvalues: dict[str, float] = {}

    for left_arm, right_arm, description in PRE_REGISTERED_CONTRASTS:
        if left_arm not in results or right_arm not in results:
            missing = [a for a in (left_arm, right_arm) if a not in results]
            skipped.append({
                "contrast": f"{left_arm} vs {right_arm}",
                "description": description,
                "reason": f"no results for {missing}",
            })
            continue

        shared_metrics = sorted(set(results[left_arm]) & set(results[right_arm]))
        if not shared_metrics:
            skipped.append({
                "contrast": f"{left_arm} vs {right_arm}",
                "description": description,
                "reason": "the two arms share no metric",
            })
            continue

        for metric in shared_metrics:
            boot = paired_bootstrap(results[left_arm][metric], results[right_arm][metric])
            exact = mcnemar(results[left_arm][metric], results[right_arm][metric])
            key = f"{left_arm}->{right_arm}:{metric}"
            if boot["p_value"] is not None:
                pvalues[key] = boot["p_value"]
            rows.append({
                "contrast": f"{left_arm} vs {right_arm}",
                "description": description,
                "metric": metric,
                "direction": "down" if metric == "shortcut_rate" else "up",
                **boot,
                "fixed": exact["fixed"],
                "broke": exact["broke"],
                "mcnemar_p": exact["p_value"],
            })

    return {
        "rows": rows,
        "skipped": skipped,
        "holm": holm(pvalues) if pvalues else {},
        "n_resamples": DEFAULT_RESAMPLES,
        "seed": DEFAULT_SEED,
        "contrasts": [f"{a} vs {b}" for a, b, _ in PRE_REGISTERED_CONTRASTS],
    }


def across_seeds(deltas: Mapping[Any, float]) -> dict:
    """The spread of a per-seed delta. Never report the mean without this.

    Three seeds estimate seed variance poorly, and that sentence belongs in the write-up next
    to the number rather than in a footnote nobody reads.
    """
    values = [float(v) for v in deltas.values()]
    if not values:
        return {
            "seeds": 0, "mean": None, "min": None, "max": None, "range": None,
            "note": "no seeds reported",
        }
    return {
        "seeds": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
        "note": (
            f"{len(values)} seed(s): this estimates seed variance poorly. Report the range "
            f"next to the mean, and say so in the write-up."
        ),
    }
