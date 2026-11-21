# VeriLoop — project overview

**One line:** small open-weights agents skip mandatory compliance steps and claim success; VeriLoop pays them only when the *database and the audit log* say the work actually happened, then trains them again on their own failures.

> **Status — read this first.** This folder is a *plan*, not a report.
> Nothing in VeriLoop has been trained at the time of writing. Where a number would
> normally go, this document says *unmeasured* and states the prediction instead.
> Treat every figure as a hypothesis until a run replaces it.

---

## 1. The problem, stated as a measurement

Small open-weights models (1.5B–8B) fail in multi-step enterprise workflows in four
repeatable ways, and each one is *measurable*:

| Failure | What it looks like | How you would measure it |
|---|---|---|
| **Hallucinated tool schemas** | invents an argument name, passes a string where an int is required | schema-valid call rate, format-error rate |
| **Redundant call loops** | calls `lookup_customer` four times, or re-runs the same query until the step cap | duplicate-call rate, loop-free rate, steps per episode |
| **Skipped mandatory steps** | approves an account without ever running the sanctions check | mandatory-step recall, forbidden-action rate |
| **Reward hacking** | learns that *saying* "all checks passed" scores as well as doing them | reward earned by a scripted cheat transcript |

The fourth is the root cause of the third, and it is the one this project attacks.

A naive reward — *"did the model say the task is done?"* — pays a shortcut exactly as
much as the compliant path, and pays it faster. That is why the shortcut is what the
model learns.

## 2. The mechanism

The reward never reads the model's prose. At the end of every rollout the reward engine
reads two things instead:

1. **The final database state** — did the rows the workflow promises actually get written?
2. **The append-only audit log** — which tool was called, with which arguments, at which
   step index?

Each scenario ships a machine-readable **episode contract** (`contracts/*.yaml`) that
states which calls are mandatory, which are forbidden in which order, and what the final
state must be. The reward is a deterministic function of (contract, audit log, DB
snapshot). It is pure, side-effect-free, and runs on the CPU with no model and no GPU —
so it can be unit-tested and attacked before a single GPU hour is spent.

Three design rules follow from that, and they are the whole idea:

- **Γ, a hard gate.** If a forbidden call happened, or a call was schema-invalid, or the
  episode never terminated, the episode's reward is `≤ 0`. There is no partial credit
  for *omission* — a trajectory that skips the work cannot bank a "safe wrong" score.
- **One graded term, one shape term.** `M` (mandatory-step recall) is hard (0/1) and pays
  most of the reward; the efficiency bonus `E` is paid **only** when `M = S = F = 1`, so
  "be fast" can never outbid "be correct".
- **Negative controls in the data.** ~15% of scenarios are *correct no-actions* (nothing
  to flag). Without them, "always escalate" is a winning policy — a lesson that came out
  of scoring a state-check-only agent.

Then, because optimising against *any* fixed check teaches evasion, the loop closes:

**Run → evaluate → classify the failure → synthesise a targeted scenario → retrain →
gate redeployment on a frozen held-out set.** That is the self-healing half.

## 3. Why the stack fits

VeriLoop is built on **[AgentTune](https://github.com/Lexsi-Labs/AgentTune)** (agent
construction + RL training + the closed-loop healing primitives) and
**[AuditKIT](https://github.com/Lexsi-Labs/AuditKIT)** (evaluation + the verified-outcome
episode layer). The seam between them is the trajectory: AgentTune writes
`trajectories.jsonl`, AuditKIT's `agent_eval` reads it and scores the verified outcome
with tool-call diagnostics on top. Neither library is re-implemented here; VeriLoop is
the environment, the contracts, the reward, and the closed loop that joins them.

## 4. Problem statements covered

Track 3 (Agentic Fine-tuning), in priority order:

- **3.9 Reward design and reward-hacking defences** — the headline contribution.
- **3.7 Business workflow and decision agents** — the KYC/compliance workflow.
- **3.6 Data and database agents** — the analytical querying branch.
- **3.8 Self-healing agents** — the retraining loop.
- **3.1 / 3.2** (reliable tool calling, efficient trajectories) — the measured outcomes.

## 5. Files here

| File | What it is |
|---|---|
| `STRUCTURE.md` | the repo map: what every directory owns, dependency direction, fill order |
| `README.md` | this overview |
| `PLAN.md` | the detailed build plan: phases, ownership, metrics, ablations, risks, calendar |
| `ARCHITECTURE.md` | components, data flow, contracts, reward maths, repo layout |
| `NOTION.md` | where the Notion page lives and what it contains |

## 6. Constraints worth keeping in view

- **Event:** Lexsi — AI Safety and Alignment Hackathon (Lexsi Labs × Cohere Labs × Hugging Face), Nov 1–24 2026. Track 3, Agentic Fine-tuning.
- **Round 1** closed Nov 15; **finalists** notified Nov 18; **final deadline** Nov 24 23:59 AoE ≈ **Nov 25 05:29 IST**. From Nov 21 that is ~3.5 days.
- **Model rule:** the core submission must use a dense, open-weights **Cohere Labs** model. `CohereLabs/tiny-aya-global` (3.35B) is the working assumption; confirm against the permitted-model list on the rules FAQ before any GPU spend.
- **Minimum bar:** at least 2 repos from the track used together (Track 3 has no higher minimum). **Preferred:** 3+.
- **Target hardware:** one 24 GB NVIDIA card (Colab L4 / Kaggle / RunPod). 4-bit + LoRA.
