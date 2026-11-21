# Working on VeriLoop

Five people, one repo, one lead who merges. This is how a PR gets merged in under a minute
instead of becoming a negotiation.

## The one rule that keeps five branches conflict-free

**A branch touches only the directories it owns.** Ownership is in `WORK_DIVISION.md` (canonical),
`CODEOWNERS` (mirrors it) and `STRUCTURE.md` §2. If your change genuinely needs a file outside your directory —
`Makefile`, `contracts/*.yaml`, `STRUCTURE.md`, `pyproject.toml` — say so in the PR body
and expect the lead to either make the edit himself or tell you to. Do not quietly edit
someone else's module; that is how two people end up rewriting the same function in two
branches and only one survives.

## Branches

```
main                        always green. Nobody commits to it directly.
feature/<owner>/<phase>-<slug>
```

Examples:

```
feature/world/1-audit-log-canonicalisation
feature/reward/2-state-verified-terms
feature/data/3-kyc-template-family
feature/gpu-eval/4-base-model-rollout
```

One branch, one phase, one concern. A branch open longer than three days is a branch that
is going to conflict — split it or land it.

## Before you push

```bash
git fetch origin
git rebase origin/main          # rebase, never merge. A merge commit on a feature branch
                               # makes the history unreadable and the diff unreviewable.
make ci                        # the growing allowlist gate — must be green
```

Then open the PR and **fill in the evidence section with real terminal output.** The lead
merges on evidence, not on description. A PR that says "tests pass" without pasting them
will be sent back, and that is not a formality: the whole project rests on numbers someone
can rerun.

## What a phase's PR must contain to be merged

1. The implementation.
2. Its tests — which already exist in `tests/` as the spec, and which you made pass
   **without weakening an assertion.** If a test is wrong, say so explicitly and explain
   why; that is a legitimate PR, silently editing the assertion is not.
3. Its tests added to the `make ci` target in the same commit. CI starts as a small
   allowlist and grows one phase at a time; a permanently-red CI gets ignored, which is
   worse than no CI.
4. The gate command for that phase, and its output, in the PR body.
5. A "what I could not verify" line. Required.

## Interface freeze

The stubs in `veriloop/` ARE the interface. **Changing a signature is a breaking change for
every other person in the team**, even if your tests pass. If you need a different
signature, raise it before you write it, not after.

Two signatures in particular are load-bearing and must not drift:

- `state_verified(contract, log, snapshot, *, terminated, steps)` — it must never grow a
  parameter that can carry the model's text. `tests/test_imports.py` enforces this, and it
  is the single assertion that keeps the project's thesis measurable.
- `call_tool(world, step, tool, args)` — the World owns its AuditLog, so there is one log
  per episode and no way to log to the wrong one.

## Things that have broken projects like this before

- **`git add -A` while someone else is working in the tree.** It sweeps their half-finished
  files into your commit. Commit specific paths.
- **Committing `.venv/`, `data/splits/*.jsonl`, `outputs/`, `audit/*.json`.** All gitignored.
  If you find one already tracked, `git rm -r --cached <path>` in its own commit.
- **Training on `E3`.** It is a probe. The moment it enters a rollout batch, every
  robustness claim in the write-up becomes circular.
- **Changing the reward after seeing results.** Attribution is gone and the entire
  experiment rests on it. The reward, the contracts and the splits are frozen at the end
  of the reward phase; a later change is reported as a stated deviation with its reason.
- **A branch that rebases onto a `main` where someone changed your module.** Re-read your
  file after the rebase, do not assume your version won.
