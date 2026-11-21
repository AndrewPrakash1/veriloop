"""Scenario generation: gold by construction, never by an LLM label.

The generator emits, per scenario: the natural-language brief, the world pre-state, the
contract, the ground-truth call route, and the required final state. Gold comes from the
same artefact that defines the check, which is what makes it free — and also what makes
it dangerous: a generator and a checker that share a bug produce consistent-but-wrong
gold that passes every automated test. `scripts/make_splits.py` therefore re-derives a
sample with an independent implementation and hand-audits 50 scenarios.
"""

from veriloop.gen.synth import Scenario, generate
from veriloop.gen.splits import SPLITS, write_splits

__all__ = ["Scenario", "generate", "SPLITS", "write_splits"]
