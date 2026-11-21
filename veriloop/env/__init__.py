"""The deterministic world.

Owns: the SQLite schema, the tool surface, the append-only audit log, and episode
stepping. Knows nothing about rewards, training, or the model's prose.

Contract with the rest of the library: after an episode, `world.snapshot()` and
`world.log` are everything the reward engine needs.
"""

from veriloop.env.audit_log import AuditLog, CallRecord, canonical
from veriloop.env.db import World, open_world
from veriloop.env.episode import EpisodeResult, Policy, run_episode
from veriloop.env.tools import TOOL_SPECS, ToolResult, call_tool, run_sql_readonly

__all__ = [
    "World", "open_world", "AuditLog", "CallRecord", "canonical",
    "TOOL_SPECS", "ToolResult", "call_tool", "run_sql_readonly",
    "EpisodeResult", "Policy", "run_episode",
]
