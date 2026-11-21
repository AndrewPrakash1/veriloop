"""Generate and FREEZE the scenario splits.

Freeze = write `data/manifest.json` + `data/checksums.txt`, committed to git. The splits
themselves are gitignored because they are large; the manifest is the leakage record.

Independently re-derive a sample of gold labels with a second implementation and hand-audit
50 scenarios. A generator and a checker that share a bug produce consistent-but-wrong gold
that passes every automated test — this step is the only thing that catches it.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    raise NotImplementedError("Phase 3")


if __name__ == "__main__":
    raise SystemExit(main())
