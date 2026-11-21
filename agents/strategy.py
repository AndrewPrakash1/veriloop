"""AgentTune AgentStrategy wiring, and the one turn-parser the whole GPU path depends on.

`ReActStrategy` is the default. `PlanAndSolve` is a stretch comparison — a second strategy
is an extra ablation arm, and the ablation budget is already spoken for, so do not add it
to the headline table.
"""

from __future__ import annotations

import json
import re
from typing import Any

#: What the parser returns when a turn cannot be parsed at all. An unknown tool name is a
#: real, scorable failure (`schema_fuzz`): `call_tool` logs it with `schema_valid=False`,
#: which is a data point. Raising here would turn that data point into a crashed episode.
UNPARSABLE: dict[str, Any] = {"tool": "", "args": {}}

#: Keys a model might use for the tool name, in priority order.
_NAME_KEYS = ("tool", "name", "action", "tool_name", "function")
_ARGS_KEYS = ("args", "arguments", "action_input", "parameters", "input")

#: ReAct-style text protocol, for models with no native function calling.
_REACT_NAME = re.compile(r"^\s*(?:Action|Tool)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_REACT_ARGS = re.compile(
    r"^\s*(?:Action\s*Input|Action\s*Args|Arguments)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_FENCE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def _balanced_objects(text: str):
    """Yield every balanced `{...}` substring, in document order, string-aware."""
    depth = 0
    start: int | None = None
    in_string = False
    escaped = False
    for i, ch in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"' and depth:
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start : i + 1]
                start = None


def _pick(payload: dict, keys) -> Any:
    """First present key, case-insensitively."""
    lowered = {str(k).lower(): v for k, v in payload.items()}
    for key in keys:
        if key in lowered:
            return lowered[key]
    return None


def _from_payload(payload: Any) -> tuple[str | None, Any]:
    """Pull (name, args) out of a decoded JSON object, flat or `{"function": {...}}`."""
    if not isinstance(payload, dict):
        return None, None
    inner = payload.get("function")
    if isinstance(inner, dict):                       # OpenAI function-calling shape
        payload = {**inner, **{k: v for k, v in payload.items() if k != "function"}}
    name = _pick(payload, _NAME_KEYS)
    args = _pick(payload, _ARGS_KEYS)
    if isinstance(name, dict):                        # {"name": {"name": ...}} — fuzz
        name = _pick(name, _NAME_KEYS)
    return (name if isinstance(name, str) and name.strip() else None), args


def _parse_args(raw: Any) -> dict:
    """Normalise arguments to a dict. Anything else becomes `{}` — never a crash.

    `call_tool` does `dict(args or {})`, so a bare string here raises `ValueError` inside
    the environment and takes the episode with it. A dict that fails `validate_args`
    instead produces a logged, scorable schema-invalid call.
    """
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


def next_call_from_completion(text: str, tool_names: set[str]) -> dict:
    """Parse one assistant turn into `{"tool", "args"}`.

    Must return a schema-INVALID shape rather than raising on unparseable output: an
    unparseable turn is a real, scorable failure mode (`schema_fuzz`), and raising would
    turn a data point into a crash.

    Accepts, in order: fenced JSON, any balanced JSON object, then a ReAct-style
    `Action:` / `Action Input:` pair. The returned tool name is NOT checked against
    `tool_names` — a hallucinated name is exactly the failure the reward is meant to see,
    and `call_tool` already logs it as `schema_valid=False`.
    """
    text = text or ""

    # 1. Fenced blocks first: the model's prose often contains braces that are not JSON.
    for block in _FENCE.findall(text):
        for candidate in _balanced_objects(block):
            try:
                name, args = _from_payload(json.loads(candidate))
            except json.JSONDecodeError:
                continue
            if name:
                return {"tool": name, "args": _parse_args(args)}

    # 2. Any balanced object in the raw text.
    for candidate in _balanced_objects(text):
        try:
            name, args = _from_payload(json.loads(candidate))
        except json.JSONDecodeError:
            continue
        if name:
            return {"tool": name, "args": _parse_args(args)}

    # 3. ReAct-style text protocol.
    name_match = _REACT_NAME.search(text)
    if name_match:
        name = name_match.group(1).strip().strip('"\'`')
        if name:
            args_match = _REACT_ARGS.search(text)
            return {"tool": name, "args": _parse_args(args_match.group(1) if args_match else None)}

    return dict(UNPARSABLE)


def to_agenttune_action(call: dict) -> dict:
    """veriloop's `{"tool", "args"}` -> AgentTune's `{"name", "arguments"}`.

    One function, so the two shapes are translated in exactly one place. `run_episode`
    reads `action["name"]` and `action["arguments"]`; veriloop reads `call["tool"]` and
    `call["args"]`.
    """
    return {"name": call.get("tool", ""), "arguments": dict(call.get("args") or {})}


def build_strategy(model: Any, tool_schemas: list[dict], *, mode: str = "react") -> Any:
    """Return an AgentTune strategy bound to `model` and the tool schemas.

    `mode`: "react" | "plan_and_solve". Anything else raises.
    """
    if mode not in ("react", "plan_and_solve"):
        raise ValueError(f"unknown strategy mode {mode!r}; expected 'react' or 'plan_and_solve'")
    if mode != "react":
        raise NotImplementedError(
            "plan_and_solve: PlanExecuteStrategy's constructor has not been verified against "
            "the installed AgentTune build, and the module docstring keeps it out of the "
            "headline table. ReAct is the only strategy the ablation runs."
        )
    raise NotImplementedError("Phase 4")
