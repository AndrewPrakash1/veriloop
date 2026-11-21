"""Targeted synthesis: one deterministic template per failure class.

Feeds new scenarios back through `veriloop.gen`, so the new contracts are validated by the
same code path as the original ones, and gold is still produced by construction.

WHY THE SPLIT IS `H` AND NOT `train`. The heal loop's drills must not be written into the
frozen splits: `data/manifest.json` and `data/checksums.txt` are the leakage record for
train/E1-E4, and appending to them after the fact would invalidate that record for every run
that quoted it. `scripts/make_splits.py` iterates its own `PLAN`, so an `H` scenario can never
reach a split file. It also must never be `E3` — training on the trap probe is the one thing
that would make every robustness number in the write-up circular.

SYNTHESIS IS DETERMINISTIC IN `(class, seed, n)`. A healing round has to be reproducible or
its gate verdict is not auditable: two people running round 2 must be able to get the same
drills, and the round record stores the seeds that produced them.
"""

from __future__ import annotations

from typing import Any

from veriloop.heal.classify import CLASS_MEANING, FailureClass
from veriloop.gen.synth import Scenario, generate

#: class -> generator template name in veriloop.gen.templates
TEMPLATES_BY_CLASS: dict[FailureClass, str] = {
    FailureClass.SCHEMA_MISMATCH:          "drill_argument_shapes",
    FailureClass.WRONG_TOOL:               "drill_near_miss_tools",
    FailureClass.SKIPPED_MANDATORY_STEP:   "same_contract_longer_horizon",
    FailureClass.FORBIDDEN_ORDER:          "tempting_shortcut_variants",
    FailureClass.EXECUTION_ERROR:          "schema_error_recovery",
    FailureClass.LOOP:                     "budget_tightened",
    FailureClass.FALSE_BLOCK:              "correct_no_action_cases",
}

#: The split label for heal-generated drills. Never a frozen split, never E3.
HEAL_SPLIT = "H"

#: Seed stride between classes. Distinct strides keep two classes from drawing the same
#: scenario even when a caller passes the same `seed` for both.
_CLASS_STRIDE = 100_000


def class_index(failure_class: FailureClass) -> int:
    """Stable index for a class, so a seed is a function of the class's IDENTITY and not of
    its position in a dict literal that someone may reorder."""
    return list(FailureClass).index(failure_class)


def template_for(failure_class: FailureClass) -> str:
    """The generator template a class selects. Raises rather than defaulting: a class with no
    template turns the loop into a guess, which is the failure this table prevents."""
    try:
        return TEMPLATES_BY_CLASS[failure_class]
    except KeyError:                                            # pragma: no cover
        raise KeyError(
            f"{failure_class!r} has no synthesis template, so a round could not act on it. "
            f"Every class in classify.FailureClass needs exactly one entry here."
        ) from None


def synthesise(failure_class: FailureClass, n: int, seed: int) -> list[Scenario]:
    """Emit `n` new scenarios isolating one class. Deterministic in `(class, seed, n)`.

    The scenarios are generated with `generate(...)`, which validates each contract and
    replays and scores each gold route, so a drill that cannot be completed perfectly is
    rejected here rather than fed into a rollout batch.
    """
    if n <= 0:
        return []
    template = template_for(failure_class)
    base = seed + class_index(failure_class) * _CLASS_STRIDE
    return generate(template, base, split=HEAL_SPLIT, n=n)


def synthesis_plan(counts_by_class: dict[str, int], *, per_class: int,
                   seed: int) -> dict[FailureClass, list[Scenario]]:
    """Drills for one round: `per_class` scenarios for every class that was OBSERVED.

    Classes that did not occur are skipped rather than drilled speculatively. A round that
    drills a failure it never saw is spending training time on a guess, and `per_class` is a
    cap on the round's size, not a target.
    """
    plan: dict[FailureClass, list[Scenario]] = {}
    for name, count in sorted(counts_by_class.items()):
        if count <= 0:
            continue
        fc = FailureClass(name)
        plan[fc] = synthesise(fc, per_class, seed)
    return plan


def describe_plan(plan: dict[FailureClass, list[Scenario]]) -> dict[str, Any]:
    """The round record's view of what was drilled, including why each class exists."""
    return {
        fc.value: {
            "template": template_for(fc),
            "n": len(scens),
            "meaning": CLASS_MEANING[fc],
            "scenario_ids": [s.scenario_id for s in scens],
        }
        for fc, scens in plan.items()
    }
