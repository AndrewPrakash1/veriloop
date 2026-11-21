"""AgentTune Harness + the scripted policy the audit and the tests run on."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from veriloop.env.episode import Policy


def build_harness(world: Any, *, step_budget: int) -> Any:
    """Return an AgentTune DictToolHarness over `veriloop.env.tools.TOOL_SPECS`.

    Takes the World, which owns its log — a separate log argument is how a caller ends up
    logging to the wrong episode.
    """
    raise NotImplementedError("Phase 4")


class ScriptedPolicy(Policy):
    """Replays a fixed route. This is why the hackability audit needs no model.

    A cheat is just a route, so it can be scored before any policy exists. The same class
    serves the tests, `scripts/run_audit.py` and the `compliant_control` row.
    """

    def __init__(self, route: Sequence[Mapping[str, Any]], final_message: str = ""):
        self._route = list(route)
        self._i = 0
        self.final_message = final_message

    @property
    def exhausted(self) -> bool:
        return self._i >= len(self._route)

    def next_call(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the next scripted call. Raise `StopIteration` when the route is done."""
        raise NotImplementedError("Phase 2")


def scripted_policy(route, final_message: str = "") -> ScriptedPolicy:
    """Convenience constructor."""
    return ScriptedPolicy(route, final_message)
