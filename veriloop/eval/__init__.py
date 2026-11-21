"""Evaluation: AuditKIT's episode layer, our contract metrics, and the statistics.

The separation is the credibility argument. The reward we TRAIN against and the verdict
AuditKIT gives are two independent code paths over the same recorded episodes. A bug in
our reward therefore surfaces as a disagreement between them, instead of as silent
agreement with ourselves.
"""

from veriloop.eval.harness import load_episodes, run_eval
from veriloop.eval.metrics import contract_metrics, tool_metrics
from veriloop.eval.stats import compare, paired_bootstrap

__all__ = ["load_episodes", "run_eval", "contract_metrics", "tool_metrics",
           "compare", "paired_bootstrap"]
