"""Derives the model-facing tool schemas from `veriloop.env.tools.TOOL_SPECS`.

One source of truth. Hand-writing the schemas a second time is how the model ends up
offered a tool the environment does not have — which is itself one of the failure modes
under study, so do not introduce it by accident.
"""

from __future__ import annotations

from typing import Any

from veriloop.env.tools import TOOL_SPECS


def openai_tools() -> list[dict]:
    """The tool schemas in OpenAI function-calling form, DERIVED from `TOOL_SPECS`.

    Derived, never authored: `TOOL_SPECS[name].parameters` IS the JSON Schema the model is
    shown, and it is the same object `validate_args` checks a call against. Copying it into
    a hand-written literal is how the two drift apart.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": dict(spec.parameters),
            },
        }
        for spec in TOOL_SPECS.values()
    ]


def prompt_block() -> str:
    """Human-readable tool list for the prompt (an alternative to function-calling).

    For a 3B model on a text protocol this is the load-bearing description: if a tool's
    arguments are not legible here, the model cannot guess them, and the resulting
    schema-invalid call reads as a model failure when it is a prompt failure.
    """
    lines: list[str] = []
    for spec in TOOL_SPECS.values():
        props: dict[str, Any] = spec.parameters.get("properties", {})
        required = set(spec.parameters.get("required", []))
        rendered = ", ".join(
            f"{name}: {schema['type']}" + ("" if name in required else " (optional)")
            for name, schema in props.items()
        ) or "no arguments"
        terminal = "  [TERMINAL — ends the episode]" if spec.is_terminal else ""
        lines.append(f"- {spec.name}({rendered}){terminal}\n    {spec.description}")
    return "\n".join(lines)


def tool_names() -> set[str]:
    """The tool names the model may call. Derived from the same single source."""
    return set(TOOL_SPECS)


def tool_schema(name: str) -> dict:
    """One tool's JSON Schema, or raise. Used by the rollout engine's action space."""
    spec = TOOL_SPECS.get(name)
    if spec is None:
        raise KeyError(f"unknown tool {name!r}; known: {sorted(TOOL_SPECS)}")
    return dict(spec.parameters)


def action_space() -> list[dict]:
    """The harness action space: what `Harness.action_space()` must return.

    AgentTune's `run_episode` reads `{s["name"] for s in harness.action_space()}`, so the
    key is `name` here even though veriloop's own call shape is `{"tool", "args"}`. The
    translation between the two shapes lives in `agents/harness.py`, in one place.
    """
    return [
        {"name": spec.name, "description": spec.description,
         "parameters": dict(spec.parameters), "is_terminal": spec.is_terminal}
        for spec in TOOL_SPECS.values()
    ]
