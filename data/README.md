# Data

`data/splits/*.jsonl` is **generated** and gitignored (large). `manifest.json` and
`checksums.txt` are **committed** — they are the leakage record, and the only thing that
makes a claim about a split auditable after the fact.

```
make splits     # -> data/splits/*.jsonl + manifest.json + checksums.txt
```

## Splits (MEASURED — 500 scenarios, generator fingerprint `ebe3a4c20242bdd468715a82c8adb2fc`)

| Split | What it is | Count | Negative | Trained on? |
|---|---|---|---|---|
| `train` | in-distribution, known templates | 100 | 17% | yes |
| `train_rl` | `train` filtered to scenarios the SFT model solves 10–90% of the time | 100 | 17% | yes |
| `E1` | held-out data, templates already seen | 100 | 17% | **no** |
| `E2` | novel templates, never seen | 100 | 60% | **no** |
| `E3` | adversarial traps: injected instruction, tempting shortcut, unsolvable case | 100 | 60% | **never — not even by the healing loop's training set** |
| `E4` | paraphrase variants of known templates | 100 | 0% | **no** |

**`train_rl` IS CURRENTLY AN UNFILTERED COPY OF `train`.** Every row in it carries
`"rl_filtered": false`, and so does the manifest. The 10–90% filter needs a pass-rate probe,
which needs the SFT model, which does not exist yet. Phase 5 narrows it. If you train on it
before that, you are training on the wide set and the row is mislabelled, not merely
unoptimised.

New numbers in a known template proves nothing about generalisation, which is why E2 and E4
exist alongside E1. `E3` is a probe: the moment it enters a rollout batch it stops being one,
and every robustness claim in the write-up becomes circular.

The negative-fraction band (10–25%) is checked on `train` and `E1` only. `E2` is the novel
domain and `E3` is traps by construction, so their fractions are deliberately outside it.
Below ~10% the false-block rate is unmeasurable; above ~25% "decide nothing" starts scoring
well.

## Every split was gated twice before it was written

1. **Each contract validates** — an unsatisfiable contract in a batch makes every sample earn
   `Γ = 0`, the group advantage collapses to zero, and the run reads as "RL did nothing".
2. **Each gold route is replayed and scored, and must be a PERFECT episode (exactly 1.0).**
   Validation only says the contract is satisfiable; only a replay says the gold route
   satisfies it. All 500 routes score 1.0.

## Manifest records

scenario counts per split · negative fraction per split · template lists per split ·
generator seed plan · per-file sha256 · dedupe rule applied · the n-gram threshold · the
fingerprint of the code that produced the splits.

It does NOT record a git SHA or a wall clock. The manifest is committed, and committing it
moves HEAD, so a SHA would be stale the moment it landed. `generator_fingerprint` is a sha256
over `veriloop/{env,contracts,gen}` plus `rewards/terms.py` — the code a split actually
depends on — so it is stable unless something that genuinely affects the splits changes,
which is exactly when you would want a diff.
