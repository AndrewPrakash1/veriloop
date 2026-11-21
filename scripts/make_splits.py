"""Phase 3 gate: generate, verify and FREEZE the scenario splits.

    make splits          # -> data/splits/*.jsonl + data/manifest.json + data/checksums.txt

Three things happen here that are easy to skip and expensive to omit:

  1. **Every scenario's gold route is replayed and scored.** A scenario whose own gold route
     does not score exactly 1.0 is rejected by `generate` before it exists. An unsatisfiable
     scenario in a rollout batch makes every sample earn `Γ = 0`, the group advantage
     collapses to zero, and the run reads as "RL did nothing" instead of "data bug".
  2. **Leakage is checked on CONTENT, not on ids.** Two scenarios drawn from the same
     `(template, seed)` under different split labels are the same scenario.
  3. **The manifest and checksums are written**, and they are COMMITTED. The splits themselves
     are gitignored because they are large; the manifest is the leakage record, and the only
     thing that makes a claim about a split auditable after the fact.

No GPU, no model, no API key, no network.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from veriloop.gen.splits import build_scenarios, write_splits  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate and freeze the scenario splits.")
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)

    started = time.time()
    print("generating (every gold route is replayed and scored; this is the gate)...")
    scenarios = build_scenarios()
    print(f"  {len(scenarios)} scenarios accepted in {time.time() - started:.1f}s")

    manifest = write_splits(scenarios, args.out)

    print(f"\n{'split':<8} {'count':>6}  {'negative':>9}  templates")
    print("-" * 78)
    for split in sorted(manifest["counts"]):
        rows = [s for s in scenarios if s.split == split]
        neg = sum(1 for s in rows if s.is_negative) / max(1, len(rows))
        templates = ",".join(t.replace("kyc_", "").replace("analytics_", "an_").replace("negative_", "neg_")
                             for t in manifest["templates_by_split"][split])
        print(f"{split:<8} {manifest['counts'][split]:>6}  {neg:>8.1%}  {templates[:40]}")

    print(f"\nnegative faction band {manifest['negative_band']}: "
          f"{json.dumps(manifest['negative_fraction'])}")
    print("train_rl.jsonl: UNFILTERED copy (rl_filtered=false on every row) — Phase 5 narrows it.")
    print(f"\nwrote {args.out}/manifest.json, {args.out}/checksums.txt, and "
          f"{len(manifest['files'])} split files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
