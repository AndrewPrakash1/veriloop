"""Arms A2-A5 — the comparison. ONE thing changes between runs: `reward_funcs`.

Same checkpoint, same prompt, same tools, same group size, same steps, same seeds.

    create_agentic_trainer(
        algorithm="grpo",              # RLOO is the fallback if GRPO is unstable at 3B
        tools=...,
        train_dataset=...,
        reward_funcs=reward_funcs(arm),
    )

Log per step: reward, mean steps, duplicate rate, and the run fingerprints (git SHA +
config hash + prompt hash). Without those the run is not reproducible and the ablation is
not defensible.

Never point a training run at E3.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 6")


if __name__ == "__main__":
    raise SystemExit(main())
