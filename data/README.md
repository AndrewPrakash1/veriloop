# Data

`data/splits/*.jsonl` is **generated** and gitignored (large). `manifest.json` and
`checksums.txt` are **committed** — they are the leakage record, and the only thing that
makes a claim about a split auditable after the fact.

```
make splits     # -> data/splits/*.jsonl + manifest.json + checksums.txt
```

## Splits

| Split | What it is | Trained on? |
|---|---|---|
| `train` | in-distribution, known templates | yes |
| `train_rl` | `train` filtered to scenarios the SFT model solves 10–90% of the time | yes |
| `E1` | held-out data, templates already seen | **no** |
| `E2` | novel templates, never seen | **no** |
| `E3` | adversarial traps: injected instruction, tempting shortcut, unsolvable case | **never — not even by the healing loop's training set** |
| `E4` | paraphrase variants of known templates | **no** |

New numbers in a known template proves nothing about generalisation, which is why E2 and
E4 exist alongside E1.

`E3` is a probe. The moment it enters a rollout batch it stops being one, and every
robustness claim in the write-up becomes circular.

## Manifest must record

scenario counts per split · generator seed · git SHA · per-file sha256 · dedupe rule
applied · the n-gram overlap threshold · per-split template lists.
