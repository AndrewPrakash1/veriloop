"""AuditKIT wiring, recorded mode.

Reads AgentTune's own `trajectories.jsonl` (written by the rollout engine) with
`auditkit.agent_eval.episodes_from_agenttune` — by RECORD SHAPE, never by import — and
scores each episode with a verified outcome as the headline.

Notes:
  * The oracle is `FinalStateAssertion` / `CustomPredicate` over the world snapshot we
    re-execute, not an LLM judge. No judge-agreement check is needed because the oracle is
    deterministic; AuditKIT should reproduce our verdict on 100% of episodes, and any
    disagreement is a bug in ONE of the two implementations worth reporting.
  * `write_sidecar()` puts the verdicts back keyed by trajectory id. AuditKIT never writes
    into AgentTune's data.
  * Unavailable evidence must be scored `unscored`, never as a failure — that is the
    library's own rule and it matters when a log is truncated.
"""

from __future__ import annotations

from typing import Any


def load_episodes(trajectories_path: str) -> list[Any]:
    """Read AgentTune trajectories into AuditKIT agent episodes."""
    raise NotImplementedError("Phase 4")


def run_eval(
    trajectories_path: str,
    contract_path: str,
    out_path: str,
    *,
    model_label: str,
) -> dict:
    """Score one arm's recordings and write `eval_results.json`."""
    raise NotImplementedError("Phase 4")
