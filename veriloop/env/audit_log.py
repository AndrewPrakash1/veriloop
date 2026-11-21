"""The append-only call log — the evidence the reward is computed from.

Append-only is a correctness property, not a convention: if the log can be edited, the
reward inherits the mutability. Enforcement is in code (`_digest` is recomputed and
compared) and asserted in tests.

Two ways to add records:
  * `append(record)` — what `tools.call_tool` uses; requires the next sequence number.
  * `record(tool, args, ...)` / `replay(calls)` — scripting helpers for the hand-authored
    attack transcripts and the tests. They exist so a cheat is DATA, not a fixture written
    in three different styles.

A NOTE ON `result_hash`. It is the digest of the CALL, not of its output:
`sha256(tool | step | canonical(args))`. That is deliberate. The hash is needed *before*
execution (so the record can be written ahead of any state change), and a digest of the
call is exactly what an independent replay needs to confirm the route was the one claimed.
It does not hash the tool's return value.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


def canonical(args: Mapping[str, Any] | None) -> str:
    """Canonical JSON for one args mapping.

    Dedupe and argument-matching hash on THIS, never on the raw text: `{"a":1,"b":2}` and
    `{"b":2,"a":1}` are the same call, and the `reformatted_repeat` attack exists to prove
    it. Keep this function the only place that decides what "same call" means.
    """
    return json.dumps(dict(args or {}), sort_keys=True, separators=(",", ":"), default=str)


def call_digest(tool: str, args: Mapping[str, Any] | None, step: int) -> str:
    """Deterministic identity of a call. See the module docstring on `result_hash`."""
    h = hashlib.sha256()
    h.update(f"{tool}|{step}|".encode())
    h.update(canonical(args).encode())
    return h.hexdigest()[:16]


def _digest_of(records: Sequence["CallRecord"]) -> str:
    """Rolling digest used to detect tampering."""
    h = hashlib.sha256()
    for r in records:
        if r is None:                       # tampered
            h.update(b"<tampered>")
            continue
        h.update(f"{r.seq}|{r.step}|{r.tool}|{r.args_canonical}|{r.result_hash}|"
                 f"{int(r.ok)}|{int(r.schema_valid)}\n".encode())
    return h.hexdigest()


@dataclass(frozen=True)
class CallRecord:
    """One tool call, as it will be scored."""

    seq: int
    step: int
    tool: str
    args: Mapping[str, Any]        # the RAW args as the model supplied them
    args_canonical: str            # canonical JSON — dedupe hashes on THIS
    result_hash: str               # digest of the call (see the module docstring)
    ok: bool
    schema_valid: bool


@dataclass
class AuditLog:
    """Ordered records for one episode, plus the rules that make them trustworthy."""

    records: list[CallRecord] = field(default_factory=list)
    _digest: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        if not self._digest:
            object.__setattr__(self, "_digest", _digest_of(self.records))

    # --- writing -----------------------------------------------------------
    def append(self, record: CallRecord) -> None:
        """Append. Raise if `record.seq` is not the next sequence number."""
        if record.seq != len(self.records):
            raise ValueError(
                f"non-contiguous sequence: expected seq={len(self.records)}, got {record.seq}"
            )
        self.records.append(record)
        self._digest = _digest_of(self.records)

    def record(
        self,
        tool: str,
        args: Mapping[str, Any] | None = None,
        *,
        step: int | None = None,
        schema_valid: bool = True,
        ok: bool = True,
        result_hash: str | None = None,
    ) -> CallRecord:
        """Scripting helper: log a call without executing it.

        Used by the attack transcripts and the tests. It computes `args_canonical` via
        `canonical()`, not inline, or the two paths would disagree about what "same call"
        means.
        """
        args = dict(args or {})
        st = len(self.records) if step is None else step
        rec = CallRecord(
            seq=len(self.records),
            step=st,
            tool=tool,
            args=args,
            args_canonical=canonical(args),
            result_hash=result_hash or call_digest(tool, args, st),
            ok=ok,
            schema_valid=schema_valid,
        )
        self.append(rec)
        return rec

    def replay(self, calls: Sequence[Any]) -> None:
        """Log a whole route. Each entry may be:
          * "tool"
          * "tool:check_type"                  -> the record_check convenience form
          * (tool, {args})
        """
        for c in calls:
            if isinstance(c, tuple):
                tool, args = c[0], c[1]
                self.record(tool, args)
                continue
            tool, sep, check = str(c).partition(":")
            if sep:
                self.record(tool, {"check_type": check, "evidence_ref": "D-900"})
            else:
                self.record(tool, {})

    def assert_append_only(self) -> None:
        """Raise if the log has been mutated since it was created."""
        if _digest_of(self.records) != self._digest:
            raise RuntimeError("audit log was mutated — it must be append-only")

    # --- reading: the questions the reward engine asks ---------------------
    def as_dicts(self) -> list[dict]:
        return [
            {
                "seq": r.seq, "step": r.step, "tool": r.tool, "args": dict(r.args),
                "args_canonical": r.args_canonical, "result_hash": r.result_hash,
                "ok": r.ok, "schema_valid": r.schema_valid,
            }
            for r in self.records
        ]

    def tools_called(self) -> list[str]:
        return [r.tool for r in self.records]

    def has_call(self, tool: str, args_subset: Mapping[str, Any] | None = None) -> bool:
        """True if `tool` was called at least once, optionally with these args.

        Matching uses `canonical()`, so key order and whitespace do not matter. With
        `args_subset`, only the named keys must match.
        """
        for r in self.records:
            if r.tool != tool:
                continue
            if args_subset is None:
                return True
            if all(dict(r.args).get(k) == v for k, v in args_subset.items()):
                return True
        return False

    def step_of(self, tool: str) -> int | None:
        """The step index of the FIRST call to `tool`, or None.

        Needed because `must_not_call.when` is evaluated at the call's step, not at the end.
        """
        for r in self.records:
            if r.tool == tool:
                return r.step
        return None

    def identical_repeats(self) -> int:
        """Count of calls that duplicate an EARLIER call on canonicalised args."""
        seen: set[tuple[str, str]] = set()
        repeats = 0
        for r in self.records:
            key = (r.tool, r.args_canonical)
            if key in seen:
                repeats += 1
            else:
                seen.add(key)
        return repeats

    def has_cycle(self, window: int = 3) -> bool:
        """True if the (tool, canonical-args) sequence repeats with period <= window."""
        seq = [(r.tool, r.args_canonical) for r in self.records]
        for period in range(1, window + 1):
            if len(seq) >= 2 * period and seq[-period:] == seq[-2 * period:-period]:
                return True
        return False
