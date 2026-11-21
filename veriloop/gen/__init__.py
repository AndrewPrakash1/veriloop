"""Scenario generation: gold by construction, verified before it is written.

The generator emits, per scenario: the natural-language brief, the world pre-state, the
contract, and the gold route — and then **replays the gold route and scores it** before the
scenario is accepted. Gold comes from the same artefact that defines the check, which is what
makes it free — and also what makes it dangerous: a generator and a checker that share a bug
produce consistent-but-wrong gold that passes every automated test.
`tests/test_gen_splits.py` therefore re-derives a sample with an independent implementation.

**`gen` does not import `rewards`.** `tests/test_imports.py` enforces the direction, and the
reason is structural: a generator that can see the scorer gets tuned to the scorer, and the
scenarios stop being an independent draw. The verification lives in `veriloop/eval/gold.py`
and the join lives in `scripts/make_splits.py`.
"""

from veriloop.gen.splits import (
    NEGATIVE_BAND,
    PLAN,
    SPLITS,
    TRAINING_SPLITS,
    build_scenarios,
    check_leakage,
    write_splits,
)
from veriloop.gen.synth import GenerationError, Scenario, derive_gold, generate, template_names

__all__ = [
    "Scenario", "generate", "derive_gold", "template_names", "GenerationError",
    "SPLITS", "PLAN", "TRAINING_SPLITS", "NEGATIVE_BAND",
    "build_scenarios", "check_leakage", "write_splits",
]
