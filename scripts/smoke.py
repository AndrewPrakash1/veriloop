"""Phase 0 gate: prove both kits import and run on a fixture.

Must be runnable with no GPU, no model, no network and no API key, and must exit non-zero
if either kit is missing or broken. Print both versions so the run config can record them.

  * AgentTune: `Project(strategy=..., harness=...)` then `proj.infer("...")` on a stub tool.
  * AuditKIT: `auditkit eval --model echo <<< "What is 2+2?"` — the zero-cost test double.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Phase 0")


if __name__ == "__main__":
    raise SystemExit(main())
