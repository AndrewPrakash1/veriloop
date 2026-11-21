"""`$case.*` substitution, shared by the generator, the world and the reward.

One implementation, used everywhere. A second, subtly different resolver is how a
contract and its gold route silently disagree.
"""

from __future__ import annotations

from typing import Any, Mapping


def resolve(value: Any, pre_state: Mapping[str, Any]) -> Any:
    """Recursively substitute `"$case.<key>"` strings from `pre_state`.

    Raise `KeyError` on an unresolvable reference — never substitute a placeholder,
    because a literal `"$case.missing"` reaching the reward engine would silently score
    as a wrong argument instead of as the data bug it is.
    """
    raise NotImplementedError("Phase 2")


def eval_predicate(expr: str, world_facts: Mapping[str, Any]) -> bool:
    """Evaluate a `when` predicate over factual quantities only.

    The predicate language is deliberately tiny and non-Turing-complete (comparisons,
    `and`/`or`/`not`, membership, `all(...)`/`any(...)` over named lists). It is
    resolved from facts the world exposes — never from the model's text.
    """
    raise NotImplementedError("Phase 2")
