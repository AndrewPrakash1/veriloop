# VeriLoop — repo structure

This file is the map. It says what every directory owns and what order to fill them in.
The rule of thumb: **the tree and the interfaces are built; the logic is yours to write.**
Every module in `veriloop/`, `agents/` and `train/` is a *stub* — a docstring stating the
single responsibility, the signatures, and `raise NotImplementedError`.

> **`make test` fails on purpose right now.** The tests in `tests/` are the executable
> spec, and they are RED until the phase named in each test module lands. Do not "fix"
> them by weakening an assertion; that is the whole artifact.

---

## 1. The tree

```
veriloop/
├── STRUCTURE.md              ← you are here
├── WORK_DIVISION.md          four roles, file-level, wave by wave
├── README.md                 overview: the problem, the mechanism, the status
├── PLAN.md                   phases, arms, metrics, risks, calendar
├── ARCHITECTURE.md           components, contracts, reward maths, threat model
├── NOTION.md                 pointer to the Notion page
├── pyproject.toml            one venv, Python 3.12 (AgentTune needs ≥3.12)
├── requirements-dev.txt      CPU-only dev + test deps
├── requirements-gpu.txt      torch/vllm/peft — only for a GPU box
├── Makefile                  smoke · test · audit · splits · baseline · train · report
│
├── contracts/                episode contracts as DATA (.yaml)
├── configs/                  one YAML per training arm and per tool run
│
├── veriloop/                 the library — import it, don't script it
│   ├── env/                  the deterministic world: SQLite, tools, audit log
│   ├── contracts/            contract schema, validation, $case resolution
│   ├── gen/                  the scenario generator (gold by construction)
│   ├── rewards/              the reward engine + the naive comparison arm
│   ├── eval/                 AuditKIT harness, metrics, statistics
│   └── heal/                 detect → classify → synthesise → gate
│
├── agents/                   policy-side glue: strategy, harness, prompt, tool specs
├── train/                    the three entry points that actually run things
├── scripts/                  one-shot operators, each with a single job
├── audit/                    the pre-GPU reward hackability audit
├── data/                     splits, manifest, checksums (generated, mostly gitignored)
├── outputs/                  per-arm adapters, trajectories, logs
├── artifacts/                HF upload + report building
├── tests/                    the executable spec
└── submission/               write-up, deck, video script
```

## 2. Who owns what

**Per-person ownership lives in `WORK_DIVISION.md`** (four roles, file-level, phased). This
section is the technical version: what each directory may do, and — more usefully — what it
must never do.

| Dir | Owns | Must NOT do |
|---|---|---|
| `veriloop/env/` | the world: tables, the eleven tools, the append-only log, episode stepping | know anything about rewards or training |
| `veriloop/contracts/` | the shape of a contract, validating it, resolving `$case.*` | touch the DB or the model |
| `veriloop/gen/` | turning a contract template into (brief, pre-state, gold route) | call an LLM to write gold |
| `veriloop/rewards/` | the pure reward function and the attack probes | mutate the world, or read the model's text |
| `veriloop/eval/` | AuditKIT wiring, metric computation, bootstrap | compute the training reward |
| `veriloop/heal/` | failure classes, targeted synthesis, the redeploy gate | retrain without the gate |
| `agents/` | how the model is prompted and stepped | contain any task logic |
| `train/` | entry points (`sft.py`, `rl.py`, `eval_arms.py`) | contain reward logic |
| `scripts/` | one-shot CLI operators | import each other |
| `tests/` | the spec | be weakened to pass |

## 3. Dependency direction (no cycles)

```
env  ←─ contracts ←─ gen ←─ rewards ←─ train ─→ agents
 │                             ↑                    │
 └─────────────→ eval ─────────┘                    │
                  │                                 │
                  └────────→ heal ──────────────────┘
```

Concretely: `rewards/` may import `env/` and `contracts/`; nothing in `env/` may import
`rewards/`. A cycle here is a design bug — the world must not know how it is scored.

## 4. Fill order (the order that keeps you unblocked)

Each step is runnable before the next one exists.

| # | Do this | Then run | It should |
|---|---|---|---|
| 0 | `make smoke` | `scripts/smoke.py` | print both kit versions, exit 0 |
| 1 | `env/schema.sql`, `env/db.py`, `env/audit_log.py`, `env/tools.py` | `make test-env` | replay a seed twice → identical bytes |
| 2 | `contracts/schema.py`, `validate.py`, `resolve.py` | `make test-contracts` | the two example contracts validate; a bad one is rejected |
| 3 | `rewards/terms.py`, `state_verified.py`, `claim_verified.py` | `make test-reward` | the compliant transcript scores 1.0; every cheat ≤ 0.1 |
| 4 | `audit/attacks.yaml` → `scripts/run_audit.py` | `make audit` | write `audit/reward_hackability.json` |
| 5 | `gen/synth.py` + `gen/templates/*` | `make splits` | write `data/` + manifest + checksums |
| 6 | `eval/harness.py`, `metrics.py`, `stats.py` | `make baseline` | write the base row for E1–E4 |
| 7 | `train/sft.py` | `make sft` | a checkpoint that emits parseable calls >95% |
| 8 | `train/rl.py` + `configs/grpo_*.yaml` | `make rl ARM=state` | one arm with an eval row |
| 9 | `heal/*` | `make heal` | one round, gate verdict logged |

**Steps 0–4 need no GPU at all.** That is deliberate: the reward audit is the cheapest
artifact that can carry the project, so it is built before any GPU hour is spent.

## 5. Environment — what was actually verified on this Mac

| Thing | State |
|---|---|
| `~/Documents/veriloop/.venv` | created on Python **3.11.5** (`/usr/local/bin/python3.11`) with `pytest` 9.2, `pydantic` 2.14, `pyyaml` 6.0 — via `virtualenv`, because `python -m venv` on this machine cannot bootstrap pip (see below) |
| `pytest --collect-only` | **54 tests collect, 0 errors** |
| `make verify-world` | **GREEN** — the gold route produces the contract's `final_state` (account `escalated`, case `pending_review`, 3 `checks` rows), and every guard holds |
| `make test-env` | **GREEN** — 8 pass, 1 skip (`tests/test_env.py`) |
| `make test` | **still RED by design** — phases 2, 3, 5, 6, 7 are unimplemented; those tests fail on `NotImplementedError: Phase N`. Do not weaken them to make the bar green |
| homebrew `python3.14` / `python3.15` | **BROKEN here**: `pyexpat` fails to dlopen — `Symbol not found: _XML_SetAllocTrackerActivationThreshold` against `/usr/lib/libexpat.1.dylib`. `pip` cannot even start under them, so neither can build a venv |
| Python 3.12 / 3.13 | **not installed** |

**Phase 0 blocker, fix before anything else.** `pyproject.toml` and AgentTune both require
**≥ 3.12**, and the only working interpreter on this machine is 3.11. Cheapest first:

1. `brew reinstall python@3.13` (the 3.14/3.15 kegs carry the broken expat link), then
   `virtualenv -p /opt/homebrew/bin/python3.13 .venv`
2. `pyenv install 3.13.1 && pyenv virtualenv 3.13.1 veriloop`
3. conda / miniforge: `conda create -n veriloop python=3.13`

**Do not let this block the work.** Phases 0–4 need no torch: pytest, pyyaml and pydantic
are genuinely enough to build and verify the environment, the contracts, the reward engine
and the whole hackability audit. Only importing AgentTune itself (and every `train/` entry
point) needs 3.12+. So fill in phases 1–4 in the 3.11 venv and recreate it on 3.13 before
`make smoke`.

## 6. Conventions

- **Contracts are data.** Never encode a workflow as Python control flow — it cannot be
  generated, counted or validated at scale.
- **The reward is pure.** `state_verified(contract, audit_log, snapshot) -> float`, no I/O,
  no model, no randomness. If it needs a fact, that fact comes in as an argument.
- **Log first, judge second.** Every tool call appends to the audit log *before* it
  mutates state, so a crash mid-episode still leaves evidence.
- **Never train on `E3`.** The trap split is a probe. The moment it enters a rollout batch
  it stops being one.
- **Freeze before the first run.** Reward, contracts, splits and the pre-registered
  contrasts. Any later change is reported as a deviation with its reason.
- **One owner per directory.** Two writers in one tree get each other's files committed
  under the wrong message.
- **Direction next to every number.** `up` or `down`, in the same table cell.
