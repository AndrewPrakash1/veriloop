"""Seven NAMED failure classes. The class SELECTS the generator.

Named, not free-text: a label a model invents cannot choose a synthesis template, so
free-text classification turns a deterministic loop into a guess. Every class below maps
to exactly one entry in `synthesise.TEMPLATES_BY_CLASS`.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from veriloop.rewards.terms import Terms


class FailureClass(str, Enum):
    SCHEMA_MISMATCH = "schema_mismatch"
    WRONG_TOOL = "wrong_tool"
    SKIPPED_MANDATORY_STEP = "skipped_mandatory_step"
    FORBIDDEN_ORDER = "forbidden_order"
    EXECUTION_ERROR = "execution_error"
    LOOP = "loop"
    FALSE_BLOCK = "false_block"


def classify(terms: Terms, episode: Any) -> FailureClass:
    """Pick the single dominant class for one failed episode.

    Priority matters and must be documented in the write-up: a forbidden violation beats a
    loop beats a missing step, because fixing the schema error changes what the later
    diagnostics even mean.
    """
    raise NotImplementedError("Phase 7")
