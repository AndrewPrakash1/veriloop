"""Arm A1 — SFT cold start on gold trajectories.

Purpose is FORMAT ONLY: teach the tool-call syntax and the episode shape. Behaviour comes
from the RL arms. Keeping this stage narrow is what makes the RL comparison meaningful.

After training, probe the per-scenario pass rate and keep scenarios the SFT model solves
10-90% of the time — RL needs headroom, not free wins and not impossible tasks.

Fallback: AgentTune's SFT rails, or plain `transformers` Trainer + PEFT if the full
AgentTune install is fighting the environment. Record which one ran in the run config.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 5")


if __name__ == "__main__":
    raise SystemExit(main())
