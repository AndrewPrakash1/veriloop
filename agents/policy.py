"""The model-backed policy — the ONE thing that turns a language model into a VeriLoop agent.

`veriloop/env/episode.py::run_episode` drives a `Policy`, and a `Policy` is anything with
`next_call(observation) -> {"tool", "args"}`. Every other implementation in this repo is a
scripted route. This is the only one that calls a model, so it is the only place where the
prompt, the chat template and the generation settings exist — and the place to look when the
model "fails" in a way that is really a prompt failure.

WHAT THE POLICY NEEDS FROM `run_episode`

The observation must carry the HISTORY of tool results, or the policy is calling blind. That
is what `history` / `last_*` in the observation are for; `run_episode` populates them.

THE THREE THINGS THAT ARE EASY TO GET WRONG

1. **Tiny Aya's chat template DROPS `tool` messages.** So tool results are rendered as text,
   inside the user turn, prefixed `Observation:` — not as a second role. A prior run of this
   exact model + stack hit this; getting it wrong means the model never sees an observation
   and the failure reads as the model being stupid.
2. **Generation must stop when the tool-call JSON closes.** A base model that does not emit
   EOS will happily ramble to the token cap on every single turn. At ~8 turns per episode and
   thousands of episodes that is the difference between a 2-hour arm and a 5-hour arm. The
   stopping criterion below checks only when the newest token is a `}`, so it costs almost
   nothing per step.
3. **Observation text is truncated.** A `run_sql` result or a customer row can be long, and an
   8K-context model with a 12-step budget will blow its window mid-episode. The prior art used
   ~1,500 characters per observation and that is the default here.

This module does NOT import torch at import time. `veriloop/` and `agents/` must stay
importable on a machine with no GPU, no torch and no model — that property is what lets the
reward engine, the audit and the whole Phase 4 test suite run before any GPU hour is spent.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from agents.prompts import STEP_BUDGET_NOTE, SYSTEM
from agents.strategy import _balanced_objects, next_call_from_completion
from agents.toolspec import prompt_block, tool_names

#: Long observations are truncated before they reach the prompt. Chars, not tokens, on
#: purpose: no tokenizer on the path of a pure rendering helper.
DEFAULT_OBSERVATION_CHARS = 1500

#: Per-turn cap. A tool call is a short JSON object; 128 leaves generous room for a
#: `run_sql` query while bounding the damage from a model that will not stop.
DEFAULT_MAX_NEW_TOKENS = 128


def _fmt(value: Any, limit: int) -> str:
    """Render a tool result for the prompt, truncated to `limit` characters."""
    if value is None:
        return "null"
    text = value if isinstance(value, str) else json.dumps(value, default=str, sort_keys=True)
    if len(text) > limit:
        return text[:limit] + f"... [{len(text) - limit} chars truncated]"
    return text


def render_turn(
    observation: Mapping[str, Any],
    *,
    max_steps: int = 12,
    observation_chars: int = DEFAULT_OBSERVATION_CHARS,
) -> str:
    """Build the single user turn for one step.

    Module-level and TORCH-FREE on purpose: `train/sft.py` renders its training prompts with
    THIS function, so the format a model is trained on is byte-identical to the format it is
    scored on. A second prompt builder living in `train/` is how a model learns a task that
    nobody evaluates.

    See the module docstring: tool results are TEXT, prefixed `Observation:` (Tiny Aya's chat
    template drops `tool` messages), and long results are truncated so a 12-step episode cannot
    walk out of the model's 8K window.
    """
    parts = [
        SYSTEM.format(brief=observation.get("brief", "")),
        STEP_BUDGET_NOTE.format(budget=max_steps),
        "",
        "Tools you may call:",
        prompt_block(),
    ]
    for record in observation.get("history") or []:
        parts.append("")
        parts.append(f"Action: {record.get('tool', '')}({_fmt(record.get('args'), 300)})")
        if record.get("ok"):
            parts.append(f"Observation: {_fmt(record.get('output'), observation_chars)}")
        else:
            parts.append(f"Observation: ERROR — {_fmt(record.get('error'), observation_chars)}")
    if not observation.get("history"):
        parts.append("")
        parts.append("No tool calls have been made yet.")
    parts.append("")
    parts.append(
        'Reply with ONE JSON object and nothing else, e.g. '
        '{"tool": "lookup_customer", "args": {"customer_id": "c60000"}}'
    )
    return "\n".join(parts)


def _make_stopping_criteria(torch: Any) -> Any:
    """Build the `StoppingCriteria` class against an already-imported torch.

    A factory rather than a module-level class purely so this module stays importable with no
    torch installed.
    """

    class _StopWhenToolCallCloses(torch.StoppingCriteria):
        """Stop the moment a complete tool-call JSON object has been emitted."""

        def __init__(self, tokenizer: Any, prompt_len: int) -> None:
            super().__init__()
            self.tokenizer = tokenizer
            self.prompt_len = prompt_len

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
            if input_ids.shape[-1] <= self.prompt_len:
                return False
            # Cheap gate first: only inspect when the newest token could close an object.
            newest = self.tokenizer.decode(input_ids[0, -1:], skip_special_tokens=True)
            if "}" not in newest and "`" not in newest:
                return False
            text = self.tokenizer.decode(input_ids[0, self.prompt_len:], skip_special_tokens=True)
            for candidate in _balanced_objects(text):
                try:
                    payload = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                probe = payload if "function" not in payload else payload["function"]
                if isinstance(probe, dict) and (
                    "tool" in probe or "name" in probe or "action" in probe
                ):
                    return True
            return False

    return _StopWhenToolCallCloses


class ModelPolicy:
    """A `Policy` backed by a causal LM. One generation per tool call.

    Deliberately holds the model rather than reaching for a global: two policies (a base and a
    tuned one) must be able to coexist for an A/B comparison without sharing state.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        *,
        max_steps: int = 12,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        temperature: float = 0.8,
        top_p: float = 0.95,
        observation_chars: int = DEFAULT_OBSERVATION_CHARS,
        seed: int | None = None,
    ) -> None:
        import torch  # deferred: this module must import without torch present

        self.model = model
        self.tokenizer = tokenizer
        self.torch = torch
        self.max_steps = max_steps
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.observation_chars = observation_chars
        self.seed = seed
        #: Last raw completion, kept for the trajectory record and for debugging a parse miss.
        self.last_completion = ""
        #: Every completion this policy produced, in order.
        self.completions: list[str] = []

    # ------------------------------------------------------------------ rendering --------
    def render(self, observation: Mapping[str, Any]) -> str:
        """The exact prompt for this turn: `render_turn` with this policy's settings."""
        return render_turn(
            observation,
            max_steps=self.max_steps,
            observation_chars=self.observation_chars,
        )

    # ----------------------------------------------------------------- generation --------
    def _generate(self, prompt: str) -> str:
        tok, model, torch = self.tokenizer, self.model, self.torch

        messages = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[-1]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
            "stopping_criteria": self._stopping(prompt_len),
        }
        if self.temperature and self.temperature > 0:
            generate_kwargs |= {"do_sample": True, "temperature": self.temperature,
                                "top_p": self.top_p}
        else:
            generate_kwargs |= {"do_sample": False}

        with torch.no_grad():
            out = model.generate(**inputs, **generate_kwargs)
        return tok.decode(out[0][prompt_len:], skip_special_tokens=True)

    def _stopping(self, prompt_len: int) -> list[Any]:
        criteria = []
        try:
            criteria.append(_make_stopping_criteria(self.torch)(self.tokenizer, prompt_len))
        except Exception:
            # The criterion is an optimisation, never a correctness requirement: if a
            # tokenizer or torch build does not support it, the episode must still run.
            criteria = []
        return criteria

    # --------------------------------------------------------------------- policy --------
    def next_call(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        """Generate one turn and parse it. Never raises on output the parser cannot read."""
        prompt = self.render(observation)
        try:
            completion = self._generate(prompt)
        except Exception as exc:                                  # noqa: BLE001
            # A generation failure is a DATA point, not a crash: the episode continues with a
            # schema-invalid call, exactly like an unparseable turn. Losing a 3-hour run to
            # one bad step is the alternative.
            self.last_completion = ""
            self.completions.append("")
            return {"tool": "", "args": {}, "_generation_error": f"{type(exc).__name__}: {exc}"}

        self.last_completion = completion
        self.completions.append(completion)
        call = next_call_from_completion(completion, tool_names())
        return call

    # ------------------------------------------------------------------ debugging --------
    def turn_prompt(self, observation: Mapping[str, Any]) -> str:
        """The exact prompt for this turn. For a trajectory record and for diagnosis."""
        return self.render(observation)
