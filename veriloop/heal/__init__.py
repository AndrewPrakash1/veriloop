"""The self-healing closed loop: detect -> classify -> synthesise -> retrain -> gate.

The gate is the safety property, not a formality: a round is promoted only if the target
metric rises on a frozen slice AND the trap set does not regress. A round that trades
robustness for score is rejected by construction.

Two properties hold the loop together, and both are asserted rather than trusted:

  * **The gate can reject.** `RedeployGate` refuses on a trap regression, on a
    `shortcut_rate` rise, and on a missing score — unmeasured is not a pass.
  * **A round continues from the CURRENT adapter.** Retraining from base produces a model
    that is fine on this round's drills and has lost every earlier round, and the round's
    numbers go UP. `assert_continues_from_current` refuses it.

The loop is deliberately model-free at its core: `heal_round` takes injected `retrain` and
`evaluate` callables, so the control flow is testable without a GPU, and the GPU side has
exactly two functions to supply.
"""

from veriloop.heal.classify import CLASS_MEANING, FailureClass, classify
from veriloop.heal.detect import ScoredEpisode, class_counts, detect_failures, score_episode
from veriloop.heal.gate import GateVerdict, RedeployGate
from veriloop.heal.loop import (
    RoundRecord,
    assert_continues_from_current,
    evaluate_with_policy,
    heal_round,
)
from veriloop.heal.synthesise import (
    HEAL_SPLIT,
    TEMPLATES_BY_CLASS,
    synthesis_plan,
    synthesise,
    template_for,
)

__all__ = [
    "detect_failures", "score_episode", "class_counts", "ScoredEpisode",
    "FailureClass", "classify", "CLASS_MEANING",
    "synthesise", "synthesis_plan", "template_for", "TEMPLATES_BY_CLASS", "HEAL_SPLIT",
    "RedeployGate", "GateVerdict", "heal_round", "RoundRecord",
    "assert_continues_from_current", "evaluate_with_policy",
]
