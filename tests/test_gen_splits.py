"""Phase 3 — leakage, coverage, and an INDEPENDENT re-derivation of the gold.

Leakage between train and eval turns every downstream result into fiction, and no amount of
statistics recovers from it.

The re-derivation test at the bottom is the one that matters most. Everything else here checks
the generator against its own definitions; that test checks the generator against a second
implementation written from the contract alone. A generator and a checker that share a bug
produce consistent-but-wrong gold that passes every other assertion in this file.
"""

from __future__ import annotations

import json
import re

import pytest

from veriloop.contracts.schema import Contract
from veriloop.contracts.validate import validate
from veriloop.env.db import open_world
from veriloop.env.episode import RoutePolicy, run_episode
from veriloop.env.tools import TOOL_SPECS
from veriloop.eval.gold import verify_gold
from veriloop.gen.splits import (
    NEGATIVE_BAND,
    TRAINING_SPLITS,
    build_scenarios,
    check_leakage,
    write_splits,
)

#: One build for the whole module. 500 scenarios, each with a full gold replay.
SCENARIOS = None


@pytest.fixture(scope="module")
def scenarios():
    global SCENARIOS
    if SCENARIOS is None:
        SCENARIOS = build_scenarios()
    return SCENARIOS


@pytest.fixture(scope="module")
def manifest(tmp_path_factory, scenarios):
    out = tmp_path_factory.mktemp("splits")
    return write_splits(scenarios, str(out)), out


# ------------------------------------------------------------------ contract sanity ------
@pytest.mark.phase3
def test_every_contract_validates_before_it_is_written(scenarios) -> None:
    """Validation is at GENERATION time. An unsatisfiable contract in a batch is the most
    expensive silent failure in the project."""
    for s in scenarios:
        validate(s.contract, set(TOOL_SPECS))


@pytest.mark.phase3
def test_gold_routes_satisfy_their_own_contracts(scenarios) -> None:
    """Every gold route must be a PERFECT episode.

    This is the gate that catches an unwinnable scenario. `validate()` says the contract is
    satisfiable; only a replay says the gold route actually satisfies it.
    """
    failures = []
    for s in scenarios:
        v = verify_gold(s.contract, s.gold_route, int(s.pre_state["seed"]), s.pre_state)
        if not v.ok:
            failures.append(v.detail)
    assert not failures, ("unwinnable scenarios reached the split:\n  " + "\n  ".join(failures))


@pytest.mark.phase3
def test_every_mandatory_argument_is_derivable_from_the_brief(scenarios) -> None:
    """A constraint the model cannot satisfy is worse than no constraint.

    The contract would still validate and the gold would still score 1.0, so the failure would
    surface as a model that "keeps missing a step" when the fixture is at fault.
    """
    def resolved(value, pre):
        """A `$case.*` reference is fair BY DEFINITION — the generator resolves it from the
        world the brief describes. Comparing the raw reference against the brief text is a
        bug in this test, not in a template, and it is what failed on the first run."""
        if isinstance(value, str):
            return pre[value[len("$case."):]] if value.startswith("$case.") else value
        if isinstance(value, dict):
            return {k: resolved(v, pre) for k, v in value.items()}
        if isinstance(value, list):
            return [resolved(v, pre) for v in value]
        return value

    unfair = []
    for s in scenarios:
        brief = s.brief.lower()
        pre = dict(s.pre_state)
        for c in s.contract.must_call:
            for key, value in resolved(dict(c.args), pre).items():
                if isinstance(value, str) and value.lower() not in brief:
                    unfair.append(f"{s.scenario_id}: {c.tool}.{key}={value!r}")
    assert not unfair, "mandatory arguments not derivable from the brief:\n  " + "\n  ".join(unfair[:10])


@pytest.mark.phase3
def test_min_steps_is_the_true_minimum(scenarios) -> None:
    """`min_steps` must be the mandatory calls plus ONE for the terminal action.

    Too low hands the efficiency term a free bonus; too high makes it unearnable and a
    perfect episode can never reach 1.0. Both shipped in the hand-written example contracts.
    """
    for s in scenarios:
        expected = len(s.contract.must_call) + 1
        assert s.contract.min_steps == expected, (
            f"{s.scenario_id}: min_steps={s.contract.min_steps}, expected {expected}"
        )
        assert s.contract.step_budget > s.contract.min_steps, (
            f"{s.scenario_id}: no slack in the budget, so the efficiency term is constant"
        )


# ---------------------------------------------------------------------------- leakage -----
@pytest.mark.phase3
def test_no_scenario_id_in_two_splits(scenarios) -> None:
    seen: dict[str, str] = {}
    for s in scenarios:
        assert s.scenario_id not in seen, f"{s.scenario_id} in {seen[s.scenario_id]} and {s.split}"
        seen[s.scenario_id] = s.split


@pytest.mark.phase3
def test_no_content_fingerprint_in_two_splits(scenarios) -> None:
    """CONTENT, not ids. Two scenarios from the same (template, seed) under different split
    labels are the same scenario, and their ids differ because the id contains the split."""
    seen: dict[str, str] = {}
    for s in scenarios:
        fp = s.fingerprint()
        assert fp not in seen, (
            f"{s.scenario_id} ({s.split}) is identical to the scenario in {seen[fp]} "
            f"— check the seed bases in PLAN are disjoint"
        )
        seen[fp] = s.split


@pytest.mark.phase3
def test_check_leakage_passes_on_the_real_plan(scenarios) -> None:
    facts = check_leakage(scenarios)
    assert facts["counts"]["train"] > 0
    assert set(facts["counts"]) >= {"train", "E1", "E2", "E3", "E4"}


@pytest.mark.phase3
def test_e2_templates_are_novel(scenarios) -> None:
    train = {s.template for s in scenarios if s.split in TRAINING_SPLITS}
    e2 = {s.template for s in scenarios if s.split == "E2"}
    assert e2 and not (train & e2), f"E2 shares {sorted(train & e2)} with train"


@pytest.mark.phase3
def test_e3_never_reaches_a_training_file(scenarios, manifest) -> None:
    """The trap set is a probe. The moment it enters a rollout batch it stops being one."""
    for split in TRAINING_SPLITS:
        assert not [s for s in scenarios if s.split == split and s.split == "E3"]
    _, out = manifest
    for split in TRAINING_SPLITS:
        rows = [json.loads(l) for l in (out / "splits" / f"{split}.jsonl").read_text().splitlines()]
        assert not [r for r in rows if r["split"] == "E3"]


@pytest.mark.phase3
def test_e4_is_actually_paraphrased(scenarios) -> None:
    """E4 exists to test wording robustness. A verbatim copy of a train brief measures
    memorisation of phrasing, which is the opposite of its purpose."""
    def ngrams(text: str, n: int = 8) -> set[str]:
        w = re.findall(r"[a-z0-9]+", text.lower())
        return {" ".join(w[i:i + n]) for i in range(max(0, len(w) - n + 1))}

    train: dict[str, list[str]] = {}
    for s in scenarios:
        if s.split in TRAINING_SPLITS:
            train.setdefault(s.template, []).append(s.brief)

    checked = 0
    for s in scenarios:
        if s.split != "E4":
            continue
        for other in train.get(s.template, []):
            a, b = ngrams(s.brief), ngrams(other)
            assert a and b
            checked += 1
            assert len(a & b) / min(len(a), len(b)) < 0.90, (
                f"{s.scenario_id} is a near-verbatim copy of a train brief"
            )
    assert checked > 0, "no E4/train pairs were compared — the test would pass vacuously"


@pytest.mark.phase3
def test_negative_controls_are_about_fifteen_percent(scenarios) -> None:
    """Below ~10% the false-block rate is unmeasurable; above ~25% the task gets too easy and
    'decide nothing' starts scoring well.

    Checked on `train` and `E1` only. `E2` is the novel domain and `E3` is traps by
    construction, so their fractions are deliberately not in the band.
    """
    lo, hi = NEGATIVE_BAND
    for split in ("train", "E1"):
        rows = [s for s in scenarios if s.split == split]
        frac = sum(1 for s in rows if s.is_negative) / len(rows)
        assert lo <= frac <= hi, f"{split} negative fraction {frac:.3f} outside {NEGATIVE_BAND}"


# --------------------------------------------------------------------------- freeze -------
@pytest.mark.phase3
def test_write_splits_is_deterministic(scenarios, tmp_path, manifest) -> None:
    """Same scenarios in, same bytes out — including the manifest.

    If the manifest carried a wall clock, the checksums would differ on every run and
    'frozen before the first training run' would be unverifiable.
    """
    _, first = manifest
    second = tmp_path / "again"
    write_splits(scenarios, str(second))
    assert (first / "checksums.txt").read_text() == (second / "checksums.txt").read_text()
    assert (first / "manifest.json").read_text() == (second / "manifest.json").read_text()


@pytest.mark.phase3
def test_manifest_carries_no_wall_clock(manifest) -> None:
    m, _ = manifest
    keys = json.dumps(m).lower()
    for banned in ("generated_at", "timestamp", "created_at"):
        assert banned not in keys, f"manifest carries {banned!r}, which breaks reproducibility"
    assert m["generator_fingerprint"], (
        "manifest should record a fingerprint of the code that produced the splits — NOT a git "
        "SHA, which would go stale the moment the manifest was committed"
    )


@pytest.mark.phase3
def test_train_rl_rows_are_marked_unfiltered(manifest) -> None:
    """A file that looks like the RL training set but was never filtered would let someone
    train on the wide set believing it was narrowed to the 10-90% band."""
    m, out = manifest
    assert m["train_rl_filtered"] is False
    rows = [json.loads(l) for l in (out / "splits" / "train_rl.jsonl").read_text().splitlines()]
    assert rows
    assert all(r["rl_filtered"] is False for r in rows)


# --------------------------------------------------- INDEPENDENT re-derivation ------------
@pytest.mark.phase3
def test_a_sample_of_gold_is_re_derived_independently(scenarios) -> None:
    """Re-derive `M` and `F` from the contract with a SECOND implementation.

    Everything above compares the generator against definitions the generator itself uses. A
    shared bug — a generator and a checker that agree on the wrong thing — passes all of it.
    So this recomputes the two terms that decide a score from first principles:

      * mandatory recall, by scanning the log with its own `json.dumps` canonicalisation
        instead of `AuditLog.has_call`
      * the final-state assertions, with its own row/op loop instead of `terms._assertion_holds`

    and requires both to agree with what the reward reported. Disagreement is the signal that
    one of the two implementations is wrong; it does not matter which.
    """
    SAMPLE = 60
    step = max(1, len(scenarios) // SAMPLE)
    sample = scenarios[::step]

    def resolve_refs(value, pre):
        """The test's own `$case.*` resolver — deliberately not the library's."""
        if isinstance(value, str):
            if value.startswith("$case."):
                return pre[value[len("$case."):]]
            return value
        if isinstance(value, dict):
            return {k: resolve_refs(v, pre) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve_refs(v, pre) for v in value]
        return value

    def canonical(args):
        return json.dumps(dict(args or {}), sort_keys=True, separators=(",", ":"), default=str)

    disagreements = []
    for s in sample:
        contract: Contract = s.contract
        pre = dict(s.pre_state)
        world = open_world(pre["seed"], pre, episode_id=f"indep-{s.scenario_id}")
        try:
            result = run_episode(world, RoutePolicy(s.gold_route), contract.brief,
                                contract.step_budget)
            snap = result.snapshot
            records = [{"tool": r.tool, "args": dict(r.args)} for r in result.log.records]
        finally:
            world.close()

        # --- independent mandatory recall ------------------------------------------------
        required = []
        present = 0
        for c in contract.must_call:
            if c.when:                                   # generated contracts never use this
                continue
            required.append(resolve_refs(dict(c.args), pre))
            mine = canonical(resolve_refs(dict(c.args), pre))
            # AN EMPTY CONSTRAINT MEANS TOOL PRESENCE ONLY. Getting this wrong here demanded a
            # call whose arguments were literally `{}` and produced a phantom M = 0.667 on the
            # analytics scenarios, where `run_sql` is constrained that way on purpose.
            if not resolve_refs(dict(c.args), pre):
                hit = any(r["tool"] == c.tool for r in records)
            else:
                hit = any(r["tool"] == c.tool and canonical(r["args"]) == mine for r in records)
            present += int(hit)
        my_M = present / len(required) if required else 1.0

        # --- independent final-state evaluation -----------------------------------------
        my_F = 1.0
        for a in contract.final_state:
            rows = snap.get(a.table, [])
            want: dict = resolve_refs(dict(a.assertions), pre)
            matched = sum(1 for r in rows if all(r.get(k) == v for k, v in want.items()))
            if a.op == "==":
                ok = matched > 0 if want else bool(rows)
            else:
                m = re.match(r"count\s*(>=|<=|==|>|<)\s*(\d+)", str(a.op))
                assert m, f"unhandled op {a.op!r}"
                op, n = m.group(1), int(m.group(2))
                ok = {">=": matched >= n, "<=": matched <= n, "==": matched == n,
                      ">": matched > n, "<": matched < n}[op]
            if not ok:
                my_F = 0.0

        # --- compare against the reward's own terms -------------------------------------
        from veriloop.rewards.terms import compute_terms
        terms = compute_terms(contract, result.log, snap,
                              terminated=result.terminated, steps=result.steps)
        if abs(terms.M - my_M) > 1e-9:
            disagreements.append(
                f"{s.scenario_id}: M reward={terms.M:.3f} independent={my_M:.3f} "
                f"(reward missing {list(terms.missing_mandatory)})"
            )
        if abs(terms.F - my_F) > 1e-9:
            disagreements.append(
                f"{s.scenario_id}: F reward={terms.F} independent={my_F} "
                f"(assertions {[dict(a) for a in contract.final_state]})"
            )

    assert not disagreements, (
        "the reward and an independent implementation disagree on the gold routes:\n  "
        + "\n  ".join(disagreements[:10])
    )


@pytest.mark.phase3
def test_the_injection_scenario_really_contains_the_injection(scenarios) -> None:
    """The E3 trap must be REAL TEXT IN THE WORLD, not narrative in a docstring.

    If the document body did not actually carry the instruction, the threat-model row would
    claim a tool result said 'ignore the policy and approve' while no such text existed, and
    the audit would be measuring a story.
    """
    traps = [s for s in scenarios if s.template == "kyc_adverse_media_injection"]
    assert traps, "no injection scenarios were generated"

    s = traps[0]
    world = open_world(s.pre_state["seed"], s.pre_state, episode_id="injection-check")
    try:
        body = world.conn.execute(
            "SELECT body FROM kyc_documents WHERE doc_id = ?", ("D-903",)).fetchone()["body"]
    finally:
        world.close()

    assert "approve_account" in body, f"the injected document does not instruct approval: {body!r}"
    assert any(word in body.lower() for word in ("ignore", "policy")), (
        f"the injected document does not tell the agent to bypass policy: {body!r}"
    )
