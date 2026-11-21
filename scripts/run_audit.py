"""THE pre-GPU artifact: write `audit/reward_hackability.json`.

Scores every attack transcript under both rewards on the same scenario and contract, then
asserts the three invariants (`make test-reward` asserts them too — a report that nothing
tests regresses silently).

Needs no GPU, no key, no model.
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    """`python scripts/run_audit.py --out audit/reward_hackability.json`"""
    raise NotImplementedError("Phase 2")


if __name__ == "__main__":
    raise SystemExit(main())
