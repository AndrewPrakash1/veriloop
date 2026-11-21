"""AgentTune AgentStrategy wiring.

`ReActStrategy` is the default. `PlanAndSolve` is a stretch comparison — a second strategy
is an extra ablation arm, and the ablation budget is already spoken for, so do not add it
to the headline table.
"""

from __future__ import annotations

from typing import Any


def build_strategy(model: Any, tool_schemas: list[dict], *, mode: str = "react") -> Any:
    """Return an AgentTune strategy bound to `model` and the tool schemas.

    `mode`: "react" | "plan_and_solve". Anything else raises.
    """
    raise NotImplementedError("Phase 4")


def next_call_from_completion(text: str, tool_names: set[str]) -> dict:
    """Parse one assistant turn into {"tool", "args"}.

    Must return a schema-INVALID shape rather than raising on unparseable output: an
    unparseable turn is a real, scorable failure mode (`schema_fuzz`), and raising would
    turn a data point into a crash.
    """
    raise NotImplementedError("Phase 4")
