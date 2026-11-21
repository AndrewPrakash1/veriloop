"""Episode stepping: the one loop that connects a policy to the world.

Kept here, not in `agents/`, so the world's invariants — step budget, terminal action,
logging order — hold for every caller: the training rollout engine, the eval harness, and
the hand-authored attack transcripts alike.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from veriloop.env.audit_log import AuditLog
from veriloop.env.db import World
from veriloop.env.tools import TOOL_SPECS, call_tool


class Policy(Protocol):
    """Anything that can pick the next tool call: a 3B model, or a scripted list."""

    def next_call(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return {"tool": str, "args": {...}}."""
        ...


class RoutePolicy:
    """Replays a fixed list of calls. A scripted route, with no model in the loop.

    This is what makes the reward hackability audit runnable before any policy exists: a
    cheat is just a route. It lives here, next to the `Policy` protocol it implements, so
    the audit, the generator's gold verification and the tests share ONE implementation
    instead of three subtly different ones.
    """

    def __init__(self, route: Sequence[Mapping[str, Any]]):
        self.calls = [{"tool": c["tool"], "args": dict(c.get("args") or {})} for c in route]
        self.i = 0

    @property
    def exhausted(self) -> bool:
        return self.i >= len(self.calls)

    def next_call(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the next scripted call, or raise `StopIteration` when the route is done.

        `run_episode` treats `StopIteration` as an unterminated episode, which is a `Γ = 0`
        — the same outcome as blowing the step budget. A route that runs out of calls
        without a terminal action is a real, scorable failure, not a harness error.
        """
        if self.i >= len(self.calls):
            raise StopIteration
        call = self.calls[self.i]
        self.i += 1
        return call


@dataclass
class EpisodeResult:
    episode_id: str
    log: AuditLog
    snapshot: Mapping[str, list[dict]]
    steps: int
    terminated: bool              # a terminal action was reached within budget
    final_message: str            # read ONLY by the naive reward arm and by the UI


def run_episode(
    world: World,
    policy: Policy,
    brief: str,
    step_budget: int,
    *,
    agent_name: str = "policy",
) -> EpisodeResult:
    """Step until a terminal action or the budget is exhausted.

    `terminated=False` on budget exhaustion — that alone is a `Gamma = 0`, so the flag is
    load-bearing, not informational.

    The brief is passed to the policy as the first observation. A policy that raises
    `StopIteration` (a scripted route running out) ends the episode the same way an
    exhausted budget does: unterminated.
    """
    observation: Mapping[str, Any] = {"brief": brief, "agent": agent_name, "step": 0}
    terminated = False
    final_message = ""
    step = 0

    while step < step_budget:
        try:
            call = policy.next_call(observation)
        except StopIteration:
            break
        if not call:
            break
        tool = str(call.get("tool", ""))
        args = call.get("args", {}) or {}
        result = call_tool(world, step, tool, args)

        if tool == "finish":
            terminated = True
            final_message = str(result.output.get("summary", "")) if result.ok else ""
            step += 1
            break
        step += 1

    return EpisodeResult(
        episode_id=world.episode_id,
        log=world.log,
        snapshot=world.snapshot(),
        steps=step,
        terminated=terminated,
        final_message=final_message,
    )
