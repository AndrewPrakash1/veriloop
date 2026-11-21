"""The self-healing closed loop: detect → classify → synthesise → retrain → gate.

The gate is the safety property, not a formality: a round is promoted only if the target
metric rises on a frozen slice AND the trap set does not regress. A round that trades
robustness for score is rejected by construction.
"""

from veriloop.heal.detect import detect_failures
from veriloop.heal.classify import FailureClass, classify
from veriloop.heal.synthesise import synthesise
from veriloop.heal.gate import RedeployGate
from veriloop.heal.loop import heal_round

__all__ = ["detect_failures", "FailureClass", "classify", "synthesise",
           "RedeployGate", "heal_round"]
