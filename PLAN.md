# VeriLoop — detailed build plan

Scope: Lexsi AI Safety & Alignment Hackathon, Track 3 (Agentic Fine-tuning), Nov 2026.
Companion docs: `README.md` (overview), `ARCHITECTURE.md` (components and data flow).

> **Status convention used below.** `[ ]` = not built. `[~]` = partially built / needs
> confirmation. Nothing in this plan has been trained or measured at the time of writing.
> Every predicted number is labelled *prediction*, every unrun check *unmeasured*.

---

## 0. The thesis, in a form that can be wrong

> **Claim.** From the same SFT checkpoint, the same prompt, the same rollout budget and
> the same optimiser, training against a **state-verified** reward produces a higher
> verified task success rate *and* a lower shortcut rate than training against a
> **claim-verified (outcome-text)** reward — and the gap survives a random-reward control.

Three falsifiable sub-claims, pre-registered:

1. **H1 — hacking gap.** On the scripted attack suite, the claim-verified reward pays a
   skipping trajectory ≥ 0.9 of the reward it pays a compliant one; the state-verified
   reward pays it ≤ 0.1. *(Static measurement, CPU-only, no model.)*
2. **H2 — real gain.** Post-training raises held-out verified task success over the base
   agent by a reportable margin, with a paired bootstrap 95% CI that excludes 0. *(The
   headline before → after number.)*
3. **H3 — no spurious gain.** The state-verified arm beats the random-reward arm, so the
   improvement is attributable to the verifier and not to clipping, longer outputs, or
   anything else RLVR does on its own.

And one negative result we commit to publishing if it happens: **H4** — if 300 steps
change nothing, we ship the training curves and say so.

## 1. Scope lock

**Ships (the one demonstration):**
one brief → agent works the case → reward engine inspects DB + audit log → a
*verified / not-verified* verdict on a case card, with the failing check named.

**Ships as evidence, not UI:** the reward hackability audit table, the base-vs-tuned
metric table with a confidence interval, the failure-mode breakdown before and after the
healing loop.

**Does NOT ship:** a web app with auth, a multi-tenant service, real bank integrations,
a vector database, a second model family, or fine-tuning more than one base model.

**Frozen after Phase 2:** the environment, the contracts, and the reward. Changing the
reward mid-experiment breaks H2's attribution, which is the point of the experiment.

## 2. Components and ownership

| # | Component | Module | Owner |
|---|---|---|---|
| C1 | Enterprise environment: SQLite schema, tool surface, append-only audit log | `veriloop/env/` | A |
| C2 | Contract schema + validation + the scenario grammar | `veriloop/contracts/`, `contracts/*.yaml` | A |
| C3 | Scenario generator (deterministic, gold-by-construction) + frozen eval splits | `veriloop/gen/`, `data/` | A |
| C4 | Reward engine (pure function of contract + audit log + DB snapshot) | `veriloop/rewards/` | B |
| C5 | Attack suite + reward hackability audit | `veriloop/rewards/probes.py`, `audit/reward_hackability.py` | B |
| C6 | Training: SFT cold start, GRPO/RLOO arms, configs | `train/`, `configs/` | B |
| C7 | Self-healing loop: detect → classify → synthesise → retrain → gate | `veriloop/heal/` | B |
| C8 | Evaluation harness (AuditKIT `agent_eval`) + metric computation + stats | `veriloop/eval/` | C |
| C9 | Artifacts: AgentTune trajectories, HF datasets/models collections, repro scripts | `artifacts/`, `scripts/` | C |
| C10 | Write-up, deck, video | `submission/` | C (A+B review) |

With a 2-person team, merge C8 into B and C10 into both. With 1 person, order is
C4 → C5 → C1/C2 → C3 → C8 → C6 → C7 → C10 — the reward audit is the cheapest thing that
can win, so it goes first.

## 3. Phases

Each phase ends with a **gate**: an artifact that either exists or doesn't.
Phases 0–3 need no GPU. That is deliberate: the highest-value evidence in this project is
CPU-only and can be produced even if GPU access collapses.

---

### Phase 0 — Skeleton and stack smoke test
**Deliverable:** repo that imports both kits and runs end-to-end on a fixture.

- [ ] `pip install auditkit` (core is zero-dep); AgentTune from source:
      `pip install --no-deps "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git"`
      — a full install pins torch, which is heavy; the agentic spine imports torch at
      package import time, so budget for that.
- [ ] Confirm Python ≥ 3.12 for AgentTune, ≥ 3.10 for AuditKIT → **one venv at 3.12**.
- [ ] `Project(strategy=..., harness=...)` + `proj.infer("...")` runs with a stub tool.
- [ ] `auditkit eval --model echo <<< "What is 2+2?"` runs (zero-cost smoke test).
- [ ] A one-command `make smoke` / `scripts/smoke.py` that runs both.
**Gate:** `scripts/smoke.py` exits 0 and prints both versions.

### Phase 1 — Environment + contracts (C1, C2)
**Deliverable:** a deterministic enterprise world and the contract language.

- [ ] `veriloop/env/db.py` — SQLite schema: `customers`, `kyc_documents`, `accounts`,
      `transactions`, `audit_log`, `cases`. Seeded from a frozen seed.
- [ ] Tool surface (each call appends to `audit_log` with args, step index, result hash):
      `lookup_customer`, `fetch_document`, `run_sql` (read-only, enforced),
      `record_check(check_type, evidence_ref)`, `screen_sanctions`, `screen_pep`,
      `escalate(case_id, reason)`, `file_sar(reason)`, `approve_account`, `reject_account`,
      `finish(summary)`.
- [ ] `contracts/schema.py` — Pydantic models for the contract (see `ARCHITECTURE.md §4`).
- [ ] Two worked contracts by hand: `kyc_clean_pass`, `kyc_sanctions_hit`.
- [ ] `pytest` proving: read-only SQL really is read-only; the audit log is append-only;
      replaying the same seed reproduces the same DB byte-for-byte.
**Gate:** `pytest -q veriloop/env` green; two hand-written contracts validate.

### Phase 2 — Reward engine + hackability audit (C4, C5) ★ highest value
**Deliverable:** `veriloop.rewards.state_verified(contract, audit_log, db) -> float`, plus
the reward hackability table.

- [ ] Implement the graded terms `Γ, M, S, F, E, D, L` and the combination rule
      (`ARCHITECTURE.md §5`), as a **pure function with no model and no randomness**.
- [ ] Implement the deliberately naive `reward_claim_verified(...)` for the comparison arm:
      it pays 1.0 when the model's final message asserts the workflow completed.
- [ ] Write the attack suite as **scripted transcripts** (hand-authored tool-call
      sequences), so the comparison needs no model:
      `claim_only`, `skip_sanctions_then_approve`, `loop_12_lookups`, `schema_fuzz`,
      `no_op_escalate_everything`, `injection_in_document`, `sql_error_retry_forever`.
- [ ] Score every attack under both rewards → `audit/reward_hackability.json`.
- [ ] Unit tests: every attack scores ≤ 0.1 under state-verified; the compliant gold
      trajectory scores 1.0; a compliant run with a broken final state scores < 1.0.
**Gate:** the hackability table exists with a real denominator, and at least three
non-trivial attacks are shown to be closed. *This table is a submission artifact on its
own, and it is produced before any GPU hour.*

### Phase 3 — Scenario generator + frozen splits (C3)
**Deliverable:** datasets with gold-by-construction, checksummed and split.

- [ ] `veriloop/gen/synth.py` — template grammar → (NL brief, DB pre-state, contract,
      ground-truth call sequence, required final state, gold trajectory). Gold comes from
      the contract, never from an LLM.
- [ ] Domain A: **KYC / compliance workflow** (identity, sanctions, PEP, adverse media,
      risk tiering, escalation, SAR filing, account open/decline).
- [ ] Domain B: **analytical querying** (business question → SQL → verified result row).
- [ ] Difficulty tiers: 2–4 tool calls, 5–8, 9–12; plus a **negative tier** (~15%) where
      the correct behaviour is *no action* / *escalate* / *decline*.
- [ ] Splits, frozen before any training and never used for tuning:
      **E1** in-distribution held-out (new data, same templates) ·
      **E2** novel templates (unseen workflow shapes) ·
      **E3** adversarial/trap (hidden instruction in a document field, a tempting shortcut,
      an unsolvable case that must be escalated) ·
      **E4** paraphrase variants.
- [ ] Leakage control: hash + n-gram dedupe across splits; write `data/manifest.json`
      with counts, seed and `checksums.txt`.
**Gate:** manifest + checksums committed; no scenario appears in two splits.

### Phase 4 — Baseline measurement (no training yet)
**Deliverable:** the "before" column.

- [ ] Agent scaffold: AgentTune `ReActStrategy` + `DictToolHarness` over the tool surface
      with the contract schema as the tool JSON schemas.
- [ ] Run the base model (zero-shot) over E1–E4, recorded via AgentTune
      `trajectory_writer`, scored by AuditKIT `agent_eval` in **recorded mode** with a
      `FinalStateAssertion` / `CustomPredicate` oracle.
- [ ] Report the **failure-mode histogram** — this is the input to Phase 7 and it is also
      useful "before" evidence on its own.
**Gate:** `eval_results.json` with a base row on all four splits and a failure breakdown.

### Phase 5 — SFT cold start (arm A1)
**Deliverable:** a checkpoint that speaks the format.

- [ ] Distil/SFT on gold trajectories (teacher = the generator's own gold, hand-checked on
      a 50-scenario sample). Assistant-only loss masking; tool-call segments weighted up.
- [ ] Re-run Phase 4's harness on the SFT checkpoint.
- [ ] Probe the pass rate per scenario; keep scenarios the SFT model solves 10–90% of the
      time into a `train_filtered` file (RL needs headroom, not free wins).
**Gate:** SFT checkpoint beats base on verified success and produces parseable calls > 95%
of the time.

### Phase 6 — The RL comparison (arms A2–A5) ★ the experiment
**Deliverable:** the headline table.

All arms: same checkpoint, same prompt, same tool set, same group size, same steps,
same seed set. **Only `reward_funcs` changes.**

| Arm | Reward | What it tests |
|---|---|---|
| **A0** | none (base) | the floor |
| **A1** | none (SFT) | format learned, behaviour not |
| **A2** | claim-verified text | the naive baseline the problem statement describes; the *attack surface* |
| **A3** | state-verified, `M`-only | is the contract check enough? |
| **A4** | state-verified + shaping (`S,E,D,L`) | does efficiency shaping help without hurting success? |
| **A5** | random (50% coin flip) | is any gain from the verifier at all? |
| **A6** | A4 + self-healing rounds (Phase 7) | does the closed loop add anything? |

- [ ] `train/rl.py` on `create_agentic_trainer(algorithm="grpo", tools=..., train_dataset=..., reward_funcs=[...])`.
      Keep RLOO as a fallback if GRPO is unstable at 3B.
- [ ] Run order if the budget is tight: **A2, A4 (seed 42)** → **A5** → **A1 vs A0** →
      seeds 43/44 for A2 and A4 → A3 → A6.
- [ ] Log per-step reward, mean steps, duplicate rate and GitSHA + config fingerprint.
**Gate:** four arms trained at one seed, each with an eval row on E1–E4.

### Phase 7 — Self-healing loop (C7, arm A6)
**Deliverable:** two healing rounds with a visible before/after.

- [ ] **Detect:** flag episodes failing the gate or exceeding the step budget.
- [ ] **Classify** into: schema mismatch · wrong tool · skipped mandatory step · forbidden
      order · SQL error · loop · false-block/over-refusal. (Named classes, not a free-text
      label — the class chooses the generator.)
- [ ] **Synthesise:** for each class, emit N new scenario/contract pairs that isolate that
      class, with gold trajectories. Deterministic templates per class.
- [ ] **Retrain:** append to the buffer, continue from the current adapter (not from base).
- [ ] **Gate redeployment:** a frozen held-out slice; promote only if verified success
      rises by the pre-registered margin **and** no regression on the trap set (E3).
- [ ] Log every round as an audit record so the loop is inspectable, not a black box.
**Gate:** two rounds run; per-class failure counts before vs after.

### Phase 8 — Artifacts, write-up, video
**Deliverable:** the submission.

- [ ] HF **datasets** collection: scenario splits, gold trajectories, AgentTune
      `trajectories.jsonl` for every arm, eval logs, AuditKIT reports.
- [ ] HF **models** collection: the LoRA adapter(s) + `train_metrics.json` + config.json.
      Add a GGUF/4-bit quant if time permits (bonus for T3, required for T2b).
- [ ] GitHub repo: code, configs, notebooks if used, `make` targets, repro commands.
- [ ] Write-up following the organiser's content list: title + one-line claim · track and
      PS numbers · abstract (<500 words) · **headline before→after number** · "for judges:
      one minute" box · why this problem / real use case · pre-registered hypotheses ·
      **threat model [T3]** · related work · kits used · before/after table with CIs ·
      what didn't work · sample inputs/outputs with summarised logs · artifacts ·
      documentation & reproduction · **declaration of work done outside the hackathon** ·
      discussion, implications, future work.
- [ ] Video/slides (final round): hook → problem → mechanism → demo → numbers → limits.
**Gate:** submission form accepted; every linked artifact opens from a clean browser.

## 4. Metrics — the scoreboard

Headline is **verified task success**, not accuracy-of-text.

| Group | Metric | Definition | Better | Source |
|---|---|---|---|---|
| Task | **Verified task success** | contract gate passes *and* final-state assertions hold | ↑ | AuditKIT `FinalStateAssertion` / custom predicate |
| Safety | **Shortcut rate** | episodes reported successful *with* a mandatory step missing | ↓ | ours (contract eval) |
| Safety | **Forbidden-action rate** | any call the contract forbids in that state | ↓ | ours |
| Safety | **Mandatory-step recall** | required calls present with valid args / required calls | ↑ | ours |
| Tool use | Tool-call validity | known tool + schema-valid args | ↑ | AuditKIT `tool_call_validity` |
| Tool use | Tool-call F1 vs gold | overlap with the reference route | ↑ | AuditKIT `tool_call_f1` |
| Tool use | Redundant calls | share of repeated identical calls | ↓ | AuditKIT `redundant_tool_calls` |
| Tool use | Loop-free rate | episodes with no detected loop | ↑ | AuditKIT `agent_loop_detection` |
| Protocol | Trajectory in order | compliant call ordering | ↑ | AuditKIT `trajectory_in_order` |
| Protocol | Finished rate | episodes ending in a terminal action | ↑ | ours |
| Efficiency | Steps / tokens / latency per episode | mean, p50, p95 | ↓ | AuditKIT performance metrics |
| Robustness | **Trap-set success (E3)** | success on the adversarial split | ↑ | ours |
| Robustness | False-block rate | correct no-action cases refused | ↓ | ours |
| Statistics | Paired bootstrap p-value, 95% CI | 10,000 resamples, seed 42, per scenario | — | AuditKIT `paired_bootstrap` |
| Statistics | Fixed / broke counts | scenarios wrong→right and right→wrong | more fixed, fewer broke | AuditKIT |

Every metric above is direction-aware: state which way is better next to the number, and
never report a mean without its across-seed spread. **Report step 300, not the best
checkpoint.**

## 5. Pre-registered statistics

- **Unit of analysis:** the scenario, not the token and not the step.
- **Paired** comparisons on the same scenario set: exact McNemar per seed, paired bootstrap
  across seeds, Holm correction across the pre-registered contrasts.
- **Pre-registered contrasts:** A4 vs A2 (verified success, shortcut rate) · A4 vs A5
  (verified success) · A2 vs A5 (does *anything* non-random help?) · A6 vs A4 (healing).
- **Resolution, stated up front:** at n = 300 and 50% accuracy the standard error is ~2.9
  points, so unpaired gaps under ~6 points cannot be resolved. Say so rather than
  over-reading a 3-point difference.
- **Seeds:** 3 minimum, and say in the write-up that 3 seeds estimate seed variance poorly.
- **Fixed after Phase 2:** reward, contracts, splits, contrasts. Anything changed later
  gets reported as a deviation with its reason.

## 6. Compute budget

Prediction, not measurement — time the first 20 steps before committing the rest.

| Arm | What runs | Rough GPU time (24 GB card, 4-bit 3B + LoRA) |
|---|---|---|
| A1 SFT | 1 epoch on ~1.5k gold trajectories | 0.5–1 h |
| A2–A5 RL | 4 arms × 300 steps × group 8 × ~12 steps/episode | 12–30 h |
| A0/A1/A4 eval | 4 splits × 2 models, recorded scoring on a served model | 3–6 h |
| A6 healing | 2 rounds × (synthesise + retrain + eval) | 6–14 h |
| Extra seeds | 2 more seeds on A2, A4 | +6–15 h |
| **Total** | | **~28–66 h of GPU** |

Levers if the clock is short: fewer arms (A2, A4, A5), 150 steps, one eval split (E1 + E3),
group size 4, and CPU-only for the reward audit and all statistics (both are genuinely
free — the audit is a pure function and AuditKIT's core has zero required dependencies).

## 7. Risk register

| Risk | Why it kills the project | Mitigation |
|---|---|---|
| Reward is gamed anyway (Goodhart) | the whole thesis collapses | attack suite is a *product*; keep E3 out of any training loop; report evasions found even if unfixed |
| Reward-floor exploit: a "safe wrong" trajectory banks partial credit | the model learns to do nothing | `Γ` gives no credit for omission; ~15% correct-no-action scenarios; mean-centred advantages |
| "Do nothing" wins on negative scenarios | false confidence | state-check alone is not enough — score coverage of mandatory-action scenarios separately; that is a *measured lesson*, not a guess |
| Model too small for multi-step reasoning | near-zero reward → no gradient | measure base pass rate on a 100-scenario probe first; if < 5%, do more SFT; keep difficulty tiering |
| AgentTune/TRL install friction | days lost to environment | Phase 0 gate; AuditKIT core needs nothing; keep a HuggingFace-`Trainer`+PEFT fallback for SFT |
| GPU quota exhaustion | no trained model | everything in Phases 0–3 is CPU-only; A0–A2 at least fit in Colab/Kaggle free tiers |
| Reward changed mid-run | H2/H3 lose attribution | freeze after Phase 2; deviations reported |
| Two writers in one repo | commits land under the wrong message; files vanish | one owner per component; commit-and-push in the same breath; `git pull --rebase` before push |
| Held-out set too easy / same templates | gains don't generalise | E2 novel templates + E4 paraphrases exist precisely for this |
| Leakage between train and eval | numbers are fiction | hash + n-gram dedupe, manifest with checksums, frozen splits |
| Judges ask "how is this different from AuditKIT's state check?" | novelty questioned | answer with the *hard gate + omission floor + shaping-with-precedence* structure and the hackability table, and say plainly that AuditKIT supplies the state assertion and the tool metrics |

## 8. Calendar

### Compressed track — target: submit by **Nov 24, 23:59 AoE** (≈ Nov 25 05:29 IST)

From a Nov 21 start that is ~3.5 days. Cut to the evidence that cannot be faked, and cut
arms before you cut rigour.

| When | Do | Ship |
|---|---|---|
| **Day 1 (Nov 21)** | Phase 0 smoke · env + 2 hand contracts · reward engine + **hackability audit** · validator tests | the audit table |
| **Day 2 (Nov 22)** | Scenario generator + frozen E1/E3 · base-model run (A0) · SFT cold start · start A2 | base row + SFT checkpoint |
| **Day 3 (Nov 23)** | A2 and A4 (seed 42) · A5 random control · eval all arms on E1 + E3 | headline table + CI |
| **Day 4 (Nov 24, morning)** | Write-up, HF artifacts, video · one healing round only if a run is already green | submission |
| **Hard stop** | nothing new starts after **12:00 IST on Nov 24** | |

If the window turns out to be longer (the plan is being read outside the finalist round),
stretch the same order: insert A3, seeds 43/44, E2/E4, and two healing rounds, and give
the write-up a full week.

### Definition of done
- [ ] Reward hackability table with at least 3 closed attacks and the numbers to prove it
- [ ] A0 vs A1 vs A2 vs A4 vs A5 on E1 and E3, each with a paired CI
- [ ] At least one self-healing round, with per-class failure counts before/after
- [ ] HF datasets collection + HF models collection + repo, all opening from a clean browser
- [ ] Write-up with the threat model, the negative results, and the reproduction commands
- [ ] Declaration of pre-hackathon work stated honestly
