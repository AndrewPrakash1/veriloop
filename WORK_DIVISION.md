# Work division — four members

This is the canonical ownership document. `CODEOWNERS` mirrors it, and `STRUCTURE.md` §2
points here rather than repeating it. If the two ever disagree, this file wins and the
other gets fixed in the same PR.

**Phase 1 (the world) is already built and verified**, so this plan starts from what is
actually left, not from zero. `veriloop/env/` is done — nobody needs to touch it except to
fix a bug, and a bug there needs the lead's sign-off because every number downstream
depends on it.

> **Team size, once.** The Lexsi rules cap teams at **1–3 members**. Four of you can build
> this; only three can be on the submission form. Settle that before Wave 2, not on Nov 24.

---

## 1. The four roles

| | Role | Mandate in one line | Owns these paths |
|---|---|---|---|
| **M1** | Reward & Audit | *Prove the reward resists gaming.* The thesis of the project lives in this directory. | `veriloop/rewards/` · `audit/` · `configs/audit.yaml` · `scripts/run_audit.py` · `tests/test_reward_*.py` |
| **M2** | Contracts & Data | *Turn one workflow into thousands of scenarios with gold nobody had to label.* | `veriloop/contracts/` · `veriloop/gen/` · `contracts/` · `data/` · `scripts/make_splits.py` · `tests/test_contracts.py` · `tests/test_gen_splits.py` |
| **M3** | Model & Evaluation | *Get a base model rolling, then prove the tuned one is better — with a control arm.* GPU owner. | `agents/` · `train/` · `configs/base.yaml` · `configs/sft.yaml` · `configs/grpo_*.yaml` · `veriloop/eval/` · `scripts/{smoke,serve,run_baseline}.py` |
| **M4** | Demo, Healing & Delivery | *Make it visible and make it inspectable.* Judges see this directory; the healing loop feeds it. | `demo/` · `veriloop/heal/` · `artifacts/` · `submission/` · `STRUCTURE.md` · `PLAN.md` |

Each person owns a **disjoint set of paths**. That is what keeps four branches conflict-free.
The only shared files are `Makefile`, `pyproject.toml`, `README.md` and `.github/` — touch
those only with a note in the PR body.

---

## 2. Wave 1 — every path exists *(phases 0, 2, 3, 4-start)*

**All four start today. Nobody is blocked on anybody.** That is the whole point of this wave:
the reward audit and the demo need no model, the generator needs no world, and the base-model
rollout only needs the contract that already exists.

### M1 — Reward & Audit

| Order | File | What lands in it |
|---|---|---|
| 1 | `veriloop/rewards/terms.py` | `compute_terms()` — `Γ, M, S, F, E, D, L`, each separately inspectable |
| 2 | `veriloop/rewards/state_verified.py` | the combination rule. `Γ` gates, `1[M=1]` is a cliff, `E` sits behind the `M=S=F=1` guard, `D`/`L` subtract |
| 3 | `veriloop/rewards/claim_verified.py` | the naive arm. **Keep it embarrassingly simple** |
| 4 | `veriloop/rewards/probes.py` | `load_attacks`, `score_attacks`, `summarise`, `compliant_control` |
| 5 | `scripts/run_audit.py` | writes `audit/reward_hackability.json` |
| 6 | `audit/attacks.yaml` | extend only if a hole is found; every new row is a threat-model entry |
| 7 | `Makefile` | add `-m phase2` to the `ci` target, **in the same commit** |

**Gate:** `make test-reward` green, then `make audit` writes the JSON.

**Done when:** every non-trivial cheat scores ≤ 0.1 under `state_verified`, the control scores
1.0, the naive reward *does* pay the cheats (otherwise the comparison demonstrates nothing),
and at least three non-trivial attacks are closed.

**The trap to avoid:** do not make `state_verified`'s constants look good. Swap every weight
and the structure must still hold. `tests/test_reward_terms.py` is written to survive that.

### M2 — Contracts & Data

| Order | File | What lands in it |
|---|---|---|
| 1 | `veriloop/contracts/validate.py` | generation-time validation. **Blocks M2's own step 2 and M3's splits** |
| 2 | `veriloop/contracts/resolve.py` | `$case.*` substitution + the tiny `when` predicate language. One resolver, used everywhere |
| 3 | `veriloop/gen/synth.py` | template + seed → scenario, gold derived from the contract |
| 4 | `veriloop/gen/templates/kyc.py` | the workflow families, cheapest first |
| 5 | `veriloop/gen/templates/negatives.py` | the ~15% correct-no-action cases. **Not filler** — see below |
| 6 | `veriloop/gen/templates/analytics.py` | the DB-query domain (PS 3.6) |
| 7 | `veriloop/gen/splits.py` | freeze: manifest, checksums, dedupe, no `E3` in any training file |
| 8 | `scripts/make_splits.py` | the operator entry point |
| 9 | `contracts/*.yaml` | more hand-written shapes; `contracts/README.md` documents them |

**Gate:** `make test-contracts` then `make test-gen`, then `make splits`.

**Done when:** no scenario appears in two splits, every contract validates *before* it is
written, and `E3` cannot reach a training file.

**The trap to avoid:** the negative-control templates are the reason the false-block rate is
measurable. Without them a do-nothing agent looks perfect on every scenario where no action
was needed. Do not treat that file as filler when the clock gets tight — cut a KYC template
first.

### M3 — Model & Evaluation

| Order | File | What lands in it |
|---|---|---|
| 1 | `scripts/smoke.py` | the phase 0 gate. **Requires Python 3.12+** — start here |
| 2 | `agents/toolspec.py` | generate the tool schemas from `TOOL_SPECS`. Never hand-write them twice |
| 3 | `agents/prompts.py` | **one** system prompt, hashed into the run config |
| 4 | `agents/strategy.py` | `parse` a turn into `{tool, args}`; unparseable → a schema-invalid call, never an exception |
| 5 | `agents/harness.py` | `build_harness`, `ScriptedPolicy` |
| 6 | `scripts/serve.sh` | serve the model, OpenAI-compatible |
| 7 | `scripts/run_baseline.py` | arm A0 over `EXAMPLE_kyc_sanctions_hit.yaml` + the failure-mode histogram |
| 8 | `veriloop/eval/oracle.py` | the independent replay-and-assert. **Must not call `state_verified`** |
| 9 | `veriloop/eval/{harness,metrics,stats,report}.py` | AuditKIT wiring, the scoreboard, paired bootstrap |

**Gate:** `make smoke` exits 0, then `make baseline` writes a base row.

**Done when:** the base model produces recorded episodes on the existing contract, and
AuditKIT's verdict agrees with ours on 100% of them. A disagreement is a bug in one of the
two implementations and is worth reporting, not hiding.

**Do this first:** get the model loading and serving on the GPU laptop before anyone types
the letters R, L. Environment work always takes longer than expected and it is the only thing
that can stop the whole team.

### M4 — Demo, Healing & Delivery

| Order | File | What lands in it |
|---|---|---|
| 1 | `demo/index.html` | **already runs.** Wire the reward columns, never compute them |
| 2 | `demo/server.py` | add a live-model mode alongside the scripted one |
| 3 | `submission/writeup.md` | fill the skeleton; the related-work section needs no results |
| 4 | `artifacts/build_report.py` | the before/after table; a cell with no run behind it prints `unmeasured` |
| 5 | `veriloop/heal/detect.py`, `classify.py` | seven **named** classes; the class selects the generator |
| 6 | `veriloop/heal/synthesise.py`, `gate.py`, `loop.py` | synthesis per class, and the two-condition redeploy gate |

**Gate:** `python demo/server.py` → the page renders, and `make heal` produces one round.

**Done when:** the page shows the money shot (`no_op_or_blanket_escalate` reaching the same
end state with zero checks recorded) and a healing round is promoted or rejected *with a
logged reason*.

**The rule for the demo:** it never computes a score. Reward columns come from
`audit/reward_hackability.json`. Two implementations of the score that disagree would put a
number on screen contradicting the write-up, in front of judges.

---

## 3. Wave 2 — get the numbers *(phases 4–6)*

| | Work | Files |
|---|---|---|
| **M3** | base rollout on all four splits, failure histogram | `scripts/run_baseline.py`, `veriloop/eval/` |
| **M3** | SFT cold start, format only | `train/sft.py`, `configs/sft.yaml` |
| **M3** | filter the training set to 10–90% pass rate | `scripts/`, `data/` (with M2) |
| **M3** | arms A2, A3, A4, A5 — **only `reward_funcs` changes** | `train/rl.py`, `configs/grpo_*.yaml` |
| **M4** | wire the measured reward columns into the demo | `demo/` |
| **M2** | freeze the splits; `E2` novel templates, `E4` paraphrases | `veriloop/gen/` |
| **M1** | hand-review the audit JSON against a live run | `audit/`, `scripts/run_audit.py` |

**This wave is where the schedule can break**, for one reason: M3 has all the GPU work and
everyone else is waiting on the numbers. Two mitigations, both cheap:

- M3 lands **A2 and A4 at seed 42 first**, then A5, then more seeds. A partial table with a
  control arm beats a complete table without one.
- M4's healing work needs no GPU, and M1's audit needs no GPU. If M3 slips, the submission
  still has two real artifacts.

**Never train on `E3`.** It is the probe. The moment it enters a rollout batch, every
robustness claim in the write-up becomes circular.

---

## 4. Wave 3 — close the loop and ship *(phases 7–8)*

| | Work | Files |
|---|---|---|
| **M4** | two healing rounds, per-class counts before/after | `veriloop/heal/` |
| **M4** | the video / slides; the healing round is the best beat in the deck | `submission/` |
| **M1** | rewrite the threat model and the evasion table with **measured** numbers | `ARCHITECTURE.md` §9 |
| **M2** | the dataset card; leakage and coverage statement | `data/`, `submission/` |
| **M3** | final eval, the before/after table with CIs, fixed/broke counts | `veriloop/eval/` |
| **all** | HF datasets + models collections, repro commands, the pre-hackathon declaration | `artifacts/` |

**Gate:** `make ci` green, every linked artifact opens from a clean browser, and the
"what is not claimed" section exists.

---

## 5. Day 1, concretely

| | Command | Expect |
|---|---|---|
| **M1** | `make test-reward` | red on `NotImplementedError: Phase 2` — then make it green |
| **M2** | `make test-contracts` | same, for `validate.py` |
| **M3** | `make smoke` | **after** recreating the venv on 3.12: `brew reinstall python@3.13` |
| **M4** | `python demo/server.py` | already green — open `http://127.0.0.1:8765` |

Then each person appends their own line to the `ci` target:

```make
ci:
	@$(PY) -m pytest tests/test_imports.py -p no:cacheprovider
	@$(PY) scripts/verify_world.py
	@$(PY) -m pytest tests/test_env.py -p no:cacheprovider
	# M1: @$(PY) -m pytest tests/test_reward_terms.py tests/test_reward_hackability.py -m phase2 -p no:cacheprovider
	# M2: @$(PY) -m pytest tests/test_contracts.py tests/test_gen_splits.py -m phase2 -p no:cacheprovider
```

Uncomment yours **in the same PR that lands your phase.** A permanently-red CI gets ignored,
which is worse than no CI.

---

## 6. The critical path

```
M1 reward ──► the audit ──► demo reward columns ──► the headline table
M2 validate ──► splits ──► M3 rollouts ──► SFT ──► the RL arms ──► before/after
                        └──► M4 healing rounds
```

Two bottlenecks, named so they can be watched:

1. **`veriloop/contracts/validate.py`** (M2) blocks M3's rollout data. It is a few hundred
   lines. M2 does it first, before any template work.
2. **M3 is the only person with a GPU**, so Wave 2 has one lane. If M3 finishes early, M1
   should pick up `veriloop/heal/` — the failure classes read the reward terms directly, so
   it is the natural hand-off.

## 7. Rules of engagement

Fuller version in `CONTRIBUTING.md`. The four that matter:

1. **One directory.** A PR that touches someone else's module gets sent back and split.
2. **Rebase, never merge.** Four merge commits on four branches makes the history unreadable.
3. **Paste your gate output in the PR body.** The lead merges on evidence.
4. **Never weaken an assertion to make a test pass.** If the test is wrong, say so and
   explain why in the PR body. That is a legitimate PR; editing the assertion is not.
