"""The generator: a template plus a seeded draw becomes one scenario.

GOLD IS DERIVED, NEVER AUTHORED. A template declares what the correct route *must do* (the
contract's `must_call`) and what its terminal action is; the route itself is built from that
by `derive_gold`. A hand-written route sitting next to a hand-written contract drifts from
it, and the drift is invisible until an arm trains on a scenario whose gold does not satisfy
its own contract — i.e. an unwinnable episode.

Two rules the generator holds to, both enforced here rather than trusted:

  * **`must_call` entries are unconditional.** A `when` on a mandatory step would make the
    route's own content depend on the world, which the generator cannot know without
    replaying it — and replaying it means deriving gold from the scorer's view of the world.
    Conditionality is expressed by having two templates instead. `when` stays available on
    `must_not_call`, where it is a property of the state and not of the route.
  * **Every scenario is replayed and scored before it is accepted** (`verify_gold`), so a
    contract that cannot be satisfied perfectly is rejected at generation time rather than
    poisoning a rollout batch.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from veriloop.contracts.resolve import resolve
from veriloop.contracts.schema import Contract
from veriloop.contracts.validate import validate
from veriloop.env.tools import TOOL_SPECS
from veriloop.gen.templates import TEMPLATES


class GenerationError(RuntimeError):
    """Raised when a template cannot produce a scorable scenario."""


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str                      # E1 | E2 | E3 | E4 | train
    domain: str                     # kyc | analytics
    template: str
    difficulty: int                 # expected step count
    brief: str
    pre_state: Mapping[str, Any]
    contract: Contract
    gold_route: list[dict]          # [{tool, args}, ...] — derived from must_call
    is_negative: bool               # correct behaviour is no action / decline / escalate

    def fingerprint(self) -> str:
        """Content identity, independent of `scenario_id` and `split`.

        Leakage control has to work on CONTENT. Two scenarios drawn from the same
        (template, seed) under different split labels are the same scenario, and an
        id-based check would miss them — which is exactly the leak that turns every
        downstream number into fiction.
        """
        import hashlib
        import json

        body = {
            "template": self.template,
            "domain": self.domain,
            "brief": self.brief,
            "pre_state": dict(self.pre_state),
            "must_call": [dict(c) for c in self.contract.must_call],
            "must_not_call": [dict(c) for c in self.contract.must_not_call],
            "final_state": [dict(a) for a in self.contract.final_state],
        }
        blob = json.dumps(body, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:32]


#: Phrases swapped to build the E4 paraphrase split. Each entry is (canonical, variant).
#: The point of E4 is that the WORDING changes while the required behaviour does not, so a
#: model that memorised the training phrasing is not helped by it.
PARAPHRASES: tuple[tuple[str, str], ...] = (
    ("Follow the standard KYC policy before any account decision.",
     "Apply our usual onboarding checks before deciding anything about the account."),
    ("Flag anything that requires escalation.",
     "Route anything needing a human decision upwards."),
    ("Onboard applicant", "Process the application for"),
    ("before any account decision",
     "prior to taking any position on the account"),
    ("Work the case to completion.", "See this case through to a conclusion."),
)


def paraphrase(brief: str) -> str:
    """Rewrite a brief's stock phrasing. Deterministic — no rng, one wording per phrase.

    Deterministic on purpose: an rng here would make the E4 split unreproducible, and the
    manifest's checksums would be meaningless.
    """
    out = brief
    for canonical, variant in PARAPHRASES:
        out = out.replace(canonical, variant)
    return out


def derive_gold(
    contract: Contract,
    terminal: Mapping[str, Any],
    gold_args: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict]:
    """Build the gold route from `must_call` plus the terminal action.

    Ordering follows `must_call`, which is the order the route must respect. Argument
    constraints and `terminal` args are resolved through the same `$case.*` resolver the
    reward uses, so a reference that cannot resolve fails here rather than silently scoring
    as a wrong argument.
    """
    overrides = {k: dict(v) for k, v in (gold_args or {}).items()}
    route: list[dict] = []
    for c in contract.must_call:
        if c.when is not None:
            raise GenerationError(
                f"contract {contract.id!r}: must_call entry {c.tool!r} has a `when` "
                f"predicate. Mandatory steps must be unconditional — conditionality is a "
                f"separate template."
            )
        args = resolve(dict(c.args), contract.pre_state)
        if not args:
            # An EMPTY constraint means "this tool must be called", not "called with no
            # arguments" — see the fairness note in templates/kyc.py. For a free-form tool
            # such as run_sql that is the only fair constraint, but the GOLD CALL still has
            # to be a VALID call, so the template must supply concrete arguments. Without
            # this the gold route makes a schema-invalid call and scores Gamma = 0, which is
            # how this was found.
            if c.tool not in overrides:
                raise GenerationError(
                    f"contract {contract.id!r}: {c.tool!r} is constrained by tool presence "
                    f"only, so the template must supply `gold_args[{c.tool!r}]` with a VALID "
                    f"argument set, or the gold route will be schema-invalid."
                )
            args = resolve(overrides[c.tool], contract.pre_state)
        route.append({"tool": c.tool, "args": args})

    tool = terminal["tool"]
    args = resolve(dict(terminal.get("args") or {}), contract.pre_state)
    if route and route[-1]["tool"] == tool:
        raise GenerationError(
            f"contract {contract.id!r}: the terminal call {tool!r} is already the last "
            f"mandatory step — the route would make it twice."
        )
    route.append({"tool": tool, "args": args})
    return route


def _build_one(fn: Callable, template: str, split: str, seed: int) -> Scenario:
    rng = random.Random(f"{template}:{split}:{seed}")
    raw = fn(rng, seed, split=split)

    contract = Contract(
        **{**raw["contract"], "id": f"{template}-{seed}", "split": split,
           "domain": raw["domain"]}
    )
    validate(contract, set(TOOL_SPECS))

    gold_route = derive_gold(contract, raw["terminal"], raw.get("gold_args"))

    scenario = Scenario(
        scenario_id=f"{split}:{template}:{seed}",
        split=split,
        domain=raw["domain"],
        template=template,
        difficulty=int(raw.get("difficulty", len(contract.must_call) + 1)),
        brief=contract.brief,
        pre_state=dict(contract.pre_state),
        contract=contract,
        gold_route=gold_route,
        is_negative=bool(raw["is_negative"]),
    )

    return scenario


def generate(
    template: str,
    seed: int,
    *,
    split: str,
    n: int = 1,
    difficulty: int | None = None,
    verify: bool = True,
) -> list[Scenario]:
    """Emit `n` scenarios from one template.

    Seeds are consecutive from `seed`, so each split must be given its own seed BASE —
    otherwise train and E1 would draw identical scenarios. `scripts/make_splits.py` owns the
    base table, and `write_splits` verifies the result did not overlap anyway.
    """
    if template not in TEMPLATES:
        raise KeyError(f"unknown template {template!r}; have {sorted(TEMPLATES)}")

    fn = TEMPLATES[template]
    out: list[Scenario] = []
    for i in range(n):
        scen = _build_one(fn, template, split, seed + i)
        if split == "E4":
            scen = _retext(scen, split)
        if verify:
            # ONE verification pass per scenario, AFTER any re-texting, so the thing that is
            # checked is the thing that is handed out.
            _verify_or_raise(scen)
        out.append(scen)
    return out


def _retext(scen: Scenario, split: str) -> Scenario:
    """Apply the split's brief wording. E4 is the paraphrase split by definition."""
    if split != "E4":
        return scen
    new_brief = paraphrase(scen.brief)
    return Scenario(
        scenario_id=scen.scenario_id,
        split=scen.split,
        domain=scen.domain,
        template=scen.template,
        difficulty=scen.difficulty,
        brief=new_brief,
        pre_state=scen.pre_state,
        contract=scen.contract.model_copy(update={"brief": new_brief}),
        gold_route=scen.gold_route,
        is_negative=scen.is_negative,
    )


def _assert_constraints_are_fair(scen: Scenario) -> None:
    """Every mandatory ARGUMENT value must be derivable from the brief.

    A constraint the model cannot satisfy is worse than no constraint: the contract still
    validates, the gold route still scores 1.0, and `M` reads as a model failure when it is a
    fixture failure. No model can guess `evidence_ref: "D-900"`, so that may not be
    constrained. Free-form arguments (`run_sql.query`) are tool-presence only.

    Enforced here rather than documented, because it is the kind of rule that is obvious
    while reading the docstring and forgotten while writing a template at 2am.
    """
    pre = scen.pre_state
    brief = scen.brief.lower()
    unfair: list[str] = []
    for c in scen.contract.must_call:
        for key, value in resolve(dict(c.args), pre).items():
            if isinstance(value, str) and value.lower() not in brief:
                unfair.append(f"{c.tool}.{key}={value!r}")
    if unfair:
        raise GenerationError(
            f"{scen.scenario_id}: these mandatory arguments cannot be derived from the "
            f"brief, so the constraint is unsatisfiable in practice: {unfair}"
        )


def _assert_predicates_are_coherent(scen: Scenario) -> None:
    """A contract that forbids acting before `all_checks_recorded` must require all of them.

    `all_checks_recorded` is defined once, in `veriloop/rewards/terms.py::CANONICAL_CHECKS`,
    as identity + sanctions + pep. A contract that gates a call on it while only requiring,
    say, identity and sanctions has a gold route that trips its own gate: the mandatory route
    never records the third check, so the predicate is still unsatisfied at the moment the
    gated call happens.

    That is not hypothetical — it is what rejected `kyc_clean_pass` and
    `analytics_below_threshold_approve` on the first run, and the reward reported it as
    `Gamma = 0, forbidden=['approve_account']` with every other term perfect.
    """
    from veriloop.rewards.terms import CANONICAL_CHECKS      # local: see the module docstring

    gates = [c for c in scen.contract.must_not_call
             if c.when and "all_checks_recorded" in c.when]
    if not gates:
        return
    required = {c.args.get("check_type") for c in scen.contract.must_call
                if c.tool == "record_check"}
    missing = set(CANONICAL_CHECKS) - required
    if missing:
        raise GenerationError(
            f"{scen.scenario_id}: must_not_call gates on `all_checks_recorded` (which means "
            f"{sorted(CANONICAL_CHECKS)}), but must_call only requires {sorted(required)}. "
            f"The gold route would trip its own gate on {sorted(missing)}."
        )


def _verify_or_raise(scen: Scenario) -> None:
    _assert_constraints_are_fair(scen)
    _assert_predicates_are_coherent(scen)
    """Replay the gold route and refuse the scenario unless it is a perfect episode.

    Imported locally so the module stays importable without the reward stack, and because
    the dependency is one-way: gen never imports rewards at module scope.
    """
    from veriloop.eval.gold import verify_gold

    pre = scen.pre_state
    verdict = verify_gold(scen.contract, scen.gold_route, int(pre["seed"]), pre)
    if not verdict.ok:
        raise GenerationError(f"{scen.scenario_id}: {verdict.detail}")


def template_names() -> Sequence[str]:
    return tuple(sorted(TEMPLATES))
