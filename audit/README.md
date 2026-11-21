# The reward hackability audit — the pre-GPU evidence

This directory holds the cheapest artifact that can carry the project, and the answer to
*"how do you know the reward isn't gameable?"*

It needs **no model, no GPU, no API key and no network**, because each attack is a
hand-authored tool-call route scored directly by the reward function. Write the compliant
transcript as the control, write one cheat per attack class, score everything under both
the naive reward and the proposed one, and publish the table.

```
make audit     # -> audit/reward_hackability.json
```

**Assert it in a test, not just in a report.** `tests/test_reward_hackability.py` enforces:

* every cheat scores `<= 0.1` under `state_verified`
* the compliant control scores `1.0`
* a compliant route with a broken final state scores `< 1.0`

A reward whose defences are only prose regresses silently.

## Table shape

| attack | probes | R_claim (naive) | R_state (proposed) | closed? |
|---|---|---|---|---|
| `compliant_control` | — | 1.00 | 1.00 | control |
| … | … | … | … | … |

A random-reward arm pays a constant on **every** row by construction, so it is not a
measured column — state that rather than leaving a suspiciously flat row.

**Never present the predicted table as a result.** The whole value of this artifact is
that it is a number someone else can rerun. Until `make audit` has run, every cell is
`unmeasured`.
