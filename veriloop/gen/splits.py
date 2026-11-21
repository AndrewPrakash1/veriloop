"""Split definitions and the freeze step.

Freezing means: counts, hashes, dedupe and a manifest written at build time. Leakage between
train and eval turns every downstream number into fiction, so `manifest.json` and
`checksums.txt` are committed to git even though the splits themselves are gitignored.

## WHAT LEAKAGE CONTROL CAN AND CANNOT CHECK, AND WHY

An id-based check is useless here. Two scenarios drawn from the same `(template, seed)` under
different split labels are the *same scenario*, and their ids differ because the id contains
the split. So the primary check is on a CONTENT fingerprint (`Scenario.fingerprint()`).

The n-gram overlap check is deliberately narrow. Brief-level overlap ACROSS splits for the
same template is **expected and correct**: `E1` is defined as "held-out data, templates
already seen in training", so of course its briefs share phrasing with train. A blanket
overlap rule would reject the split it exists to protect. What must not happen is an `E4`
"paraphrase" that is a verbatim copy of a train brief, so that is the only place the check
applies, at a near-verbatim threshold.

Checks, in order:

  1. no `scenario_id` in two splits
  2. no CONTENT fingerprint in two splits
  3. `E2` template set disjoint from the training template set
  4. no `E3` scenario in any training file
  5. no `E4` brief a near-verbatim copy of a train brief for the same template
  6. negative-control fraction in band for `train` and `E1`
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from veriloop.gen.synth import Scenario, generate

#: E1 in-distribution held-out · E2 novel templates · E3 adversarial traps · E4 paraphrase.
#: E3 IS NEVER TRAINED ON. The moment it enters a rollout batch it stops being a probe.
SPLITS: Mapping[str, str] = {
    "E1": "held-out data, templates already seen in training",
    "E2": "novel templates, never seen in training",
    "E3": "adversarial traps: injected instruction, tempting shortcut, unsolvable case",
    "E4": "paraphrase variants of known templates",
}

#: split -> [(template, count, seed_base), ...]. The seed BASES are disjoint per split, so
#: train and E1 cannot draw the same scenario even before the fingerprint check runs.
#: Counts are hand-set rather than proportional so the negative-control fraction is a
#: deliberate choice and not an accident of rounding.
PLAN: Mapping[str, list[tuple[str, int, int]]] = {
    "train": [
        ("kyc_clean_pass", 40, 10_000),
        ("kyc_high_risk_approve", 22, 10_100),
        ("kyc_shortcut_probe", 21, 10_200),
        ("kyc_sanctions_hit", 4, 10_300),
        ("kyc_pep_hit", 4, 10_400),
        ("negative_no_action_case_closed", 3, 10_500),
        ("negative_decline_duplicate", 3, 10_600),
        ("negative_escalate_missing_evidence", 3, 10_700),
    ],
    "E1": [
        ("kyc_clean_pass", 40, 100_000),
        ("kyc_high_risk_approve", 22, 100_100),
        ("kyc_shortcut_probe", 21, 100_200),
        ("kyc_sanctions_hit", 4, 100_300),
        ("kyc_pep_hit", 4, 100_400),
        ("negative_no_action_case_closed", 3, 100_500),
        ("negative_decline_duplicate", 3, 100_600),
        ("negative_escalate_missing_evidence", 3, 100_700),
    ],
    "E2": [
        ("analytics_below_threshold_approve", 40, 200_000),
        ("analytics_over_threshold", 30, 200_100),
        ("analytics_below_threshold_clear", 30, 200_200),
    ],
    "E3": [
        ("kyc_adverse_media_injection", 40, 300_000),
        ("kyc_shortcut_probe", 40, 300_100),
        ("negative_escalate_missing_evidence", 20, 300_200),
    ],
    "E4": [
        ("kyc_clean_pass", 50, 400_000),
        ("kyc_high_risk_approve", 30, 400_100),
        ("kyc_shortcut_probe", 20, 400_200),
    ],
}

TRAINING_SPLITS = ("train",)

#: Negative-fraction band. Below the floor the false-block rate is unmeasurable; above the
#: ceiling the task gets easy and "decide nothing" starts scoring well. Checked on `train` and
#: `E1` only: `E2` is the novel domain and `E3` is traps by construction, so their fractions
#: are deliberately not in this band.
NEGATIVE_BAND = (0.10, 0.25)

_NEAR_VERBATIM = 0.90


def build_scenarios() -> list[Scenario]:
    """Draw every scenario from `PLAN`. Deterministic in the plan's seeds."""
    out: list[Scenario] = []
    for split, entries in PLAN.items():
        for template, count, seed_base in entries:
            out.extend(generate(template, seed_base, split=split, n=count))
    return out


# ------------------------------------------------------------------ leakage checks --------
def _ngrams(text: str, n: int = 8) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def _overlap(a: str, b: str) -> float:
    ga, gb = _ngrams(a), _ngrams(b)
    if not ga or not gb:
        return 0.0
    return len(ga & gb) / min(len(ga), len(gb))


def check_leakage(scenarios: Sequence[Scenario]) -> dict[str, Any]:
    """Raise on any leakage. Returns the facts worth putting in the manifest."""
    by_split: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_split.setdefault(s.split, []).append(s)

    # 1 + 2 — ids and content fingerprints must be unique to one split
    seen_ids: dict[str, str] = {}
    seen_fp: dict[str, str] = {}
    for s in scenarios:
        if s.scenario_id in seen_ids:
            raise ValueError(
                f"scenario_id {s.scenario_id!r} appears in {seen_ids[s.scenario_id]!r} and "
                f"{s.split!r}"
            )
        seen_ids[s.scenario_id] = s.split

        fp = s.fingerprint()
        if fp in seen_fp:
            raise ValueError(
                f"CONTENT LEAK: the scenario in {seen_fp[fp]!r} and the one in {s.split!r} "
                f"are identical (fingerprint {fp}). Different ids do not make them different "
                f"scenarios — check the seed BASES in PLAN are disjoint."
            )
        seen_fp[fp] = s.split

    # 3 — E2 must be novel
    train_templates = {s.template for split in TRAINING_SPLITS for s in by_split.get(split, [])}
    e2_templates = {s.template for s in by_split.get("E2", [])}
    shared = train_templates & e2_templates
    if shared:
        raise ValueError(f"E2 is meant to be NOVEL templates but shares {sorted(shared)} with train")

    # 4 — E3 must not reach a training file
    for split in TRAINING_SPLITS:
        leaked = sorted({s.scenario_id for s in by_split.get(split, []) if s.split == "E3"})
        if leaked:
            raise ValueError(f"E3 scenarios reached the {split!r} split: {leaked}")

    # 5 — E4 must not be a verbatim copy of a train brief for the same template
    train_by_template: dict[str, list[str]] = {}
    for split in TRAINING_SPLITS:
        for s in by_split.get(split, []):
            train_by_template.setdefault(s.template, []).append(s.brief)
    for s in by_split.get("E4", []):
        for other in train_by_template.get(s.template, []):
            score = _overlap(s.brief, other)
            if score >= _NEAR_VERBATIM:
                raise ValueError(
                    f"E4 scenario {s.scenario_id} is a near-verbatim copy of a train brief "
                    f"(8-gram overlap {score:.2f}). A paraphrase split that is not paraphrased "
                    f"measures memorisation of wording, which is the opposite of its purpose."
                )

    # 6 — negative fraction
    fractions: dict[str, float] = {}
    for split in ("train", "E1"):
        rows = by_split.get(split, [])
        if not rows:
            continue
        frac = sum(1 for s in rows if s.is_negative) / len(rows)
        fractions[split] = round(frac, 4)
        lo, hi = NEGATIVE_BAND
        if not (lo <= frac <= hi):
            raise ValueError(
                f"{split}: negative-control fraction {frac:.3f} is outside the band "
                f"{NEGATIVE_BAND}. Below it the false-block rate is unmeasurable; above it "
                f"'decide nothing' starts scoring well."
            )

    return {
        "negative_fraction": fractions,
        "templates_by_split": {k: sorted({s.template for s in v}) for k, v in sorted(by_split.items())},
        "counts": {k: len(v) for k, v in sorted(by_split.items())},
    }


# ------------------------------------------------------------------------- writing --------
def _row(scenario: Scenario) -> dict[str, Any]:
    return {
        "scenario_id": scenario.scenario_id,
        "split": scenario.split,
        "domain": scenario.domain,
        "template": scenario.template,
        "difficulty": scenario.difficulty,
        "is_negative": scenario.is_negative,
        "brief": scenario.brief,
        "pre_state": dict(scenario.pre_state),
        "contract": scenario.contract.model_dump(mode="json"),
        "gold_route": scenario.gold_route,
        "fingerprint": scenario.fingerprint(),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


#: The code a split actually depends on: the world's shape, the contract language, the
#: generator, and the reward term that defines `all_checks_recorded`. A split generated from
#: these sources is reproducible from them.
_FINGERPRINT_SOURCES = ("veriloop/env", "veriloop/contracts", "veriloop/gen",
                        "veriloop/rewards/terms.py")


def generator_fingerprint() -> str:
    """sha256 over the source that determines the splits. NO wall clock, NO git SHA.

    A git SHA would be worse provenance AND a wart: the manifest is committed, and committing
    it changes HEAD, so the committed manifest would be stale the moment it landed. A
    fingerprint over the actual inputs is stable unless something that genuinely affects the
    splits changes — which is exactly when you would want the diff.
    """
    root = Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    for rel in _FINGERPRINT_SOURCES:
        target = root / rel
        files = sorted(target.rglob("*")) if target.is_dir() else [target]
        for path in files:
            if path.is_file() and path.suffix in {".py", ".sql", ".yaml"}:
                h.update(str(path.relative_to(root)).encode())
                h.update(b"\0")
                h.update(path.read_bytes())
                h.update(b"\0")
    return h.hexdigest()[:32]


def write_splits(scenarios: list[Scenario], out_dir: str) -> dict:
    """Write splits, `manifest.json` and `checksums.txt`. Returns the manifest.

    Also writes `train_rl.jsonl`, which Phase 5 will narrow to scenarios the SFT model solves
    10-90% of the time. Until that probe runs it is an UNFILTERED copy, and every row carries
    `"rl_filtered": false` so nothing can silently train on the wide set believing it was
    narrowed. The manifest records the same fact.
    """
    facts = check_leakage(scenarios)

    out = Path(out_dir)
    splits_dir = out / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    by_split: dict[str, list[Scenario]] = {}
    for s in scenarios:
        by_split.setdefault(s.split, []).append(s)

    written: dict[str, Path] = {}
    for split, rows in sorted(by_split.items()):
        path = splits_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(_row(s), sort_keys=True) + "\n" for s in rows))
        written[split] = path

    for split in TRAINING_SPLITS:
        src = by_split.get(split, [])
        path = splits_dir / f"{split}_rl.jsonl"
        path.write_text("".join(
            json.dumps({**_row(s), "rl_filtered": False}, sort_keys=True) + "\n" for s in src
        ))
        written[f"{split}_rl"] = path

    checksums = {name: _sha256(p) for name, p in sorted(written.items())}
    (out / "checksums.txt").write_text(
        "# sha256 of every generated split file. Written by scripts/make_splits.py.\n"
        "# Frozen before the first training run; any change afterwards is a stated deviation.\n"
        + "".join(f"{digest}  splits/{name}.jsonl\n" for name, digest in checksums.items())
    )

    manifest = {
        "generated_by": "scripts/make_splits.py",
        "generator_fingerprint": generator_fingerprint(),
        "split_definitions": dict(SPLITS),
        "plan": {k: [list(e) for e in v] for k, v in PLAN.items()},
        "counts": facts["counts"],
        "templates_by_split": facts["templates_by_split"],
        "negative_fraction": facts["negative_fraction"],
        "negative_band": list(NEGATIVE_BAND),
        "training_splits": list(TRAINING_SPLITS),
        "train_rl_filtered": False,
        "train_rl_note": (
            "unfiltered copy; Phase 5 narrows it to scenarios the SFT model solves 10-90% of "
            "the time. Every row carries rl_filtered=false until then."
        ),
        "dedupe": {
            "rule": "content fingerprint (template, domain, brief, pre_state, contract fields)",
            "split_seed_bases_disjoint": True,
            "e4_near_verbatim_threshold": _NEAR_VERBATIM,
        },
        "files": {f"{name}.jsonl": digest for name, digest in checksums.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest
