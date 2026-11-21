"""The redeploy gate — the safety property of the whole loop.

THREE conditions, all required. Two would not be enough:

  1. the target metric RISES on a frozen slice, by at least the pre-registered margin;
  2. the trap split (E3) does NOT regress beyond its tolerance;
  3. `shortcut_rate` does NOT rise beyond its tolerance.

Condition 3 is the one specific to this project. `shortcut_rate` counts episodes that REPORT
success while a mandatory step is missing — the exact behaviour the state-verified reward
exists to catch. A round that raises `verified_success` by making the policy better at
claiming work it did not do would pass conditions 1 and 2 and is the one failure mode this
loop must never be able to promote. A tolerance is needed rather than a hard zero because the
metric is estimated from a finite slice; `shortcut_tolerance` defaults to `trap_tolerance`.

UNMEASURED IS NOT A PASS. If a required score is absent from `before` or `after`, the gate
refuses to promote and says so. A missing number is not evidence of no regression, and a loop
that promotes on a missing number is worse than no loop, because it launders the gap into a
"verified" claim.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

#: The three scores the gate reasons over. `verified_success` must rise; the other two must
#: not fall / rise respectively.
REQUIRED = ("verified_success", "trap_success")


@dataclass
class GateVerdict:
    promote: bool
    reason: str
    scores: Mapping[str, Any] = field(default_factory=dict)

    def as_row(self) -> dict:
        return {"promote": self.promote, "reason": self.reason, "scores": dict(self.scores)}


class RedeployGate:
    """`margin` and `trap_tolerance` are pre-registered: set them BEFORE the round runs.

    Choosing them after seeing the deltas is how a gate becomes a rubber stamp.
    """

    def __init__(self, margin: float = 0.0, trap_tolerance: float = 0.0,
                 shortcut_tolerance: float | None = None):
        if margin < 0 or trap_tolerance < 0:
            raise ValueError("margin and trap_tolerance are magnitudes; pass them positive")
        self.margin = margin
        self.trap_tolerance = trap_tolerance
        self.shortcut_tolerance = (
            trap_tolerance if shortcut_tolerance is None else shortcut_tolerance
        )

    def evaluate(self, before: Mapping[str, float],
                 after: Mapping[str, float]) -> GateVerdict:
        """`before`/`after` carry verified_success, trap_success and shortcut_rate."""
        missing = [k for k in REQUIRED if before.get(k) is None or after.get(k) is None]
        if missing:
            return GateVerdict(
                promote=False,
                reason=(
                    f"unmeasured: {', '.join(missing)} absent from "
                    f"{'before' if before.get(missing[0]) is None else 'after'}. A missing "
                    f"score is not a pass."
                ),
                scores={"before": dict(before), "after": dict(after)},
            )

        deltas = {k: after[k] - before[k] for k in before if k in after}
        scores = {"before": dict(before), "after": dict(after),
                  "delta": {k: round(v, 6) for k, v in deltas.items()}}

        failures: list[str] = []

        # 1 — the target must actually rise.
        gain = deltas["verified_success"]
        if gain < self.margin:
            failures.append(
                f"target metric did not rise by the pre-registered margin: "
                f"verified_success {before['verified_success']:.3f} -> "
                f"{after['verified_success']:.3f} (+{gain:.3f}, needed +{self.margin:.3f})"
            )

        # 2 — robustness must not be traded away for score.
        trap_drop = before["trap_success"] - after["trap_success"]
        if trap_drop > self.trap_tolerance:
            failures.append(
                f"trap set regressed beyond tolerance: trap_success "
                f"{before['trap_success']:.3f} -> {after['trap_success']:.3f} "
                f"(-{trap_drop:.3f}, allowed -{self.trap_tolerance:.3f})"
            )

        # 3 — and it must not have got better at claiming work it did not do.
        if "shortcut_rate" in before and "shortcut_rate" in after:
            shortcut_rise = after["shortcut_rate"] - before["shortcut_rate"]
            if shortcut_rise > self.shortcut_tolerance:
                failures.append(
                    f"shortcut_rate rose beyond tolerance: "
                    f"{before['shortcut_rate']:.3f} -> {after['shortcut_rate']:.3f} "
                    f"(+{shortcut_rise:.3f}, allowed +{self.shortcut_tolerance:.3f}) — the "
                    f"round bought score by reporting work it did not do"
                )

        if failures:
            return GateVerdict(promote=False, reason="; ".join(failures), scores=scores)
        return GateVerdict(
            promote=True,
            reason=(
                f"verified_success +{gain:.3f} (>= {self.margin:.3f}), "
                f"trap_success {deltas['trap_success']:+.3f} (tol {self.trap_tolerance:.3f})"
                + (f", shortcut_rate {deltas['shortcut_rate']:+.3f}"
                   if "shortcut_rate" in deltas else "")
            ),
            scores=scores,
        )
