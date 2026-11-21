"""Build the two Hugging Face collections.

  datasets: scenario splits, gold trajectories, AgentTune `trajectories.jsonl` per arm,
            eval logs, AuditKIT reports.
  models:   LoRA adapter(s) + train_metrics.json + the run config (with its fingerprints).

Add a GGUF / 4-bit quant if there is time — it is a bonus at T3 and a requirement at T2b.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 8")


if __name__ == "__main__":
    raise SystemExit(main())
