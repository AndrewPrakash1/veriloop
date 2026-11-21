"""The eleven tools, and the one place a call is executed.

Eleven is not arbitrary: `screen_sanctions` and `screen_pep` are separate tools because
"skipped the sanctions check" is the canonical failure this project attacks, and separate
tools make the mandatory-step term unambiguous. `finish` is the terminal action.

LOGGING DISCIPLINE — read this before changing anything here.

  * The **durable evidence** is the SQL `audit_log` mirror, and it is written BEFORE the
    tool executes. A crash mid-execution therefore still leaves a record of the attempt.
  * The **scored evidence** is `world.log`, written AFTER execution so the record carries
    the tool's real outcome in `ok`. The reward never reads the mirror.
  * Consequence to keep in mind: a scripted route added with `AuditLog.record()`/`replay()`
    (the attack transcripts, the tests) does NOT touch the mirror. That is intentional —
    scripting helpers must not need a database.
  * Contracts assert on BUSINESS tables (`checks`, `accounts`, `cases`), never on the
    mirror. A contract that asserted on the mirror would be asserting on the logger.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Mapping

from veriloop.env.audit_log import call_digest
from veriloop.env.db import World

_EPOCH = datetime(2026, 1, 1, 9, 0, 0, tzinfo=timezone.utc)


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    parameters: Mapping[str, Any]      # JSON Schema — what the model is shown
    is_terminal: bool = False
    mutates: bool = False


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: Any = None
    error: str | None = None


def _s(**props: Any) -> dict:
    """A JSON-Schema object: `_s(customer_id=("string", "the customer id"))`."""
    return {
        "type": "object",
        "properties": {k: {"type": v[0], "description": v[1]} for k, v in props.items()},
        "required": [k for k, v in props.items() if len(v) > 2 and v[2] == "req"],
        "additionalProperties": False,
    }


#: The single source of truth for what the model is offered. The agent's tool schemas are
#: DERIVED from this; nothing hand-writes them twice (see agents/toolspec.py).
TOOL_SPECS: dict[str, ToolSpec] = {
    "lookup_customer": ToolSpec(
        "lookup_customer", "Look up a customer record by id.",
        _s(customer_id=("string", "the customer id", "req"))),
    "fetch_document": ToolSpec(
        "fetch_document", "Fetch a KYC document's contents by id.",
        _s(doc_id=("string", "the document id", "req"))),
    "run_sql": ToolSpec(
        "run_sql", "Run a read-only SQL SELECT against the case database.",
        _s(query=("string", "a single SELECT statement", "req"))),
    "screen_sanctions": ToolSpec(
        "screen_sanctions", "Screen a customer against the sanctions list.",
        _s(customer_id=("string", "the customer id", "req"))),
    "screen_pep": ToolSpec(
        "screen_pep", "Screen a customer against the politically-exposed-persons list.",
        _s(customer_id=("string", "the customer id", "req"))),
    "record_check": ToolSpec(
        "record_check",
        "Record that a mandatory compliance check was performed, with its evidence.",
        _s(check_type=("string", "identity | sanctions | pep", "req"),
           case_id=("string", "defaults to the case in this episode"),
           evidence_ref=("string", "the document or list the check was run against"),
           result=("string", "pass | fail | inconclusive")),
        mutates=True),
    "escalate": ToolSpec(
        "escalate", "Escalate the case for human review.",
        _s(case_id=("string", "the case id", "req"),
           reason=("string", "why the case is being escalated")),
        mutates=True),
    "file_sar": ToolSpec(
        "file_sar", "File a suspicious activity report for the case.",
        _s(case_id=("string", "the case id", "req"),
           reason=("string", "the suspicion")),
        mutates=True),
    "approve_account": ToolSpec(
        "approve_account", "Open the customer's account. Only legal after every mandatory check.",
        _s(customer_id=("string", "the customer id", "req")),
        mutates=True),
    "reject_account": ToolSpec(
        "reject_account", "Decline the customer's account application.",
        _s(customer_id=("string", "the customer id", "req"),
           reason=("string", "why the application was declined")),
        mutates=True),
    "finish": ToolSpec(
        "finish", "Terminal action. Report the outcome and end the episode.",
        _s(summary=("string", "a short summary of what was done", "req")),
        is_terminal=True),
}

_TYPE_CHECKS: dict[str, Callable[[Any], bool]] = {
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
}


def validate_args(spec: ToolSpec, args: Mapping[str, Any]) -> list[str]:
    """Return a list of problems; empty means the call is schema-valid.

    Unknown arguments are REJECTED, not ignored: a hallucinated argument name is one of the
    failure modes under study, and ignoring it would let the model look correct while
    inventing a parameter.
    """
    problems: list[str] = []
    props = spec.parameters.get("properties", {})
    for key in spec.parameters.get("required", []):
        if key not in args:
            problems.append(f"missing required argument {key!r}")
    for key, value in args.items():
        if key not in props:
            problems.append(f"unknown argument {key!r}")
            continue
        want = props[key]["type"]
        check = _TYPE_CHECKS.get(want)
        if check and not check(value):
            problems.append(f"argument {key!r} must be {want}, got {type(value).__name__}")
    return problems


def call_tool(world: World, step: int, tool: str, args: Mapping[str, Any]) -> ToolResult:
    """Validate, log, then execute. See the module docstring for the logging discipline."""
    args = dict(args or {})
    spec = TOOL_SPECS.get(tool)
    if spec is None:
        world.log.record(tool, args, step=step, schema_valid=False, ok=False)
        return ToolResult(False, error=f"unknown tool {tool!r}")

    problems = validate_args(spec, args)
    if problems:
        world.log.record(tool, args, step=step, schema_valid=False, ok=False)
        return ToolResult(False, error="; ".join(problems))

    _write_mirror(world, step, tool, args)        # durable evidence, BEFORE executing
    try:
        result = _IMPL[tool](world, args, step)
    except Exception as exc:                      # a tool crash is a scorable failure
        result = ToolResult(False, error=f"{type(exc).__name__}: {exc}")
    world.log.record(tool, args, step=step, ok=result.ok, schema_valid=True)
    return result


def _write_mirror(world: World, step: int, tool: str, args: Mapping[str, Any]) -> None:
    """Append to the durable SQL mirror. Timestamp is DERIVED, never read from the clock."""
    seq = len(world.log.records)
    world.conn.execute(
        "INSERT INTO audit_log (episode_id, step, tool, args_json, result_hash, ok, at)"
        " VALUES (?,?,?,?,?,?,?)",
        (world.episode_id, step, tool,
         _canonical(args), call_digest(tool, args, step), 1,
         (_EPOCH + timedelta(seconds=seq)).isoformat()),
    )
    world.conn.commit()


def _canonical(args: Mapping[str, Any]) -> str:
    from veriloop.env.audit_log import canonical
    return canonical(args)


def _customer(world: World, customer_id: str) -> sqlite3.Row | None:
    return world.conn.execute(
        "SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()


def _case(world: World, case_id: str | None) -> sqlite3.Row | None:
    if case_id is None:
        return world.conn.execute("SELECT * FROM cases ORDER BY rowid LIMIT 1").fetchone()
    return world.conn.execute("SELECT * FROM cases WHERE case_id = ?", (case_id,)).fetchone()


# --- implementations ---------------------------------------------------------------------
def _t_lookup_customer(world, args, step):
    row = _customer(world, args["customer_id"])
    if row is None:
        return ToolResult(False, error="customer not found")
    return ToolResult(True, output=dict(row))


def _t_fetch_document(world, args, step):
    row = world.conn.execute(
        "SELECT * FROM kyc_documents WHERE doc_id = ?", (args["doc_id"],)).fetchone()
    if row is None:
        return ToolResult(False, error="document not found")
    return ToolResult(True, output=dict(row))


def _t_screen(world, table: str, args, step):
    cust = _customer(world, args["customer_id"])
    if cust is None:
        return ToolResult(False, error="customer not found")
    n = world.conn.execute(
        f"SELECT COUNT(*) AS n FROM {table} WHERE full_name = ?", (cust["full_name"],)
    ).fetchone()["n"]
    return ToolResult(True, output={"hit": bool(n), "matches": int(n), "list": table})


def _t_screen_sanctions(world, args, step):
    return _t_screen(world, "sanctions_list", args, step)


def _t_screen_pep(world, args, step):
    return _t_screen(world, "pep_list", args, step)


def _t_record_check(world, args, step):
    case = _case(world, args.get("case_id"))
    if case is None:
        return ToolResult(False, error="no such case")
    check_type = args["check_type"]
    seq = len(world.log.records)
    world.conn.execute(
        "INSERT INTO checks (check_id, case_id, check_type, result, evidence_ref, recorded_at)"
        " VALUES (?,?,?,?,?,?)",
        (f"CHK-{case['case_id']}-{check_type}", case["case_id"], check_type,
         args.get("result", "pass"), args.get("evidence_ref"),
         (_EPOCH + timedelta(seconds=seq)).isoformat()),
    )
    world.conn.commit()
    return ToolResult(True, output={"recorded": check_type, "case_id": case["case_id"]})


def _t_escalate(world, args, step):
    case = _case(world, args.get("case_id"))
    if case is None:
        return ToolResult(False, error="no such case")
    world.conn.execute("UPDATE cases SET state = 'pending_review' WHERE case_id = ?",
                       (case["case_id"],))
    world.conn.execute("UPDATE accounts SET status = 'escalated' WHERE customer_id = ?",
                       (case["customer_id"],))
    world.conn.commit()
    return ToolResult(True, output={"case_id": case["case_id"], "state": "pending_review"})


def _t_file_sar(world, args, step):
    case = _case(world, args.get("case_id"))
    if case is None:
        return ToolResult(False, error="no such case")
    world.conn.execute("UPDATE cases SET state = 'sar_filed' WHERE case_id = ?",
                       (case["case_id"],))
    world.conn.commit()
    return ToolResult(True, output={"case_id": case["case_id"], "state": "sar_filed"})


def _t_approve_account(world, args, step):
    cust = _customer(world, args["customer_id"])
    if cust is None:
        return ToolResult(False, error="customer not found")
    world.conn.execute(
        "UPDATE accounts SET status = 'open', opened_at = ? WHERE customer_id = ?",
        ((_EPOCH + timedelta(seconds=len(world.log.records))).isoformat(),
         cust["customer_id"]),
    )
    world.conn.commit()
    return ToolResult(True, output={"customer_id": cust["customer_id"], "status": "open"})


def _t_reject_account(world, args, step):
    cust = _customer(world, args["customer_id"])
    if cust is None:
        return ToolResult(False, error="customer not found")
    world.conn.execute("UPDATE accounts SET status = 'declined' WHERE customer_id = ?",
                       (cust["customer_id"],))
    world.conn.commit()
    return ToolResult(True, output={"customer_id": cust["customer_id"], "status": "declined"})


def _t_finish(world, args, step):
    return ToolResult(True, output={"summary": args["summary"], "terminal": True})


def _t_run_sql(world, args, step):
    return run_sql_readonly(world, args["query"])


_IMPL: dict[str, Callable[[World, Mapping[str, Any], int], ToolResult]] = {
    "lookup_customer": _t_lookup_customer,
    "fetch_document": _t_fetch_document,
    "run_sql": _t_run_sql,
    "screen_sanctions": _t_screen_sanctions,
    "screen_pep": _t_screen_pep,
    "record_check": _t_record_check,
    "escalate": _t_escalate,
    "file_sar": _t_file_sar,
    "approve_account": _t_approve_account,
    "reject_account": _t_reject_account,
    "finish": _t_finish,
}


def spec_json_schemas() -> list[dict]:
    """OpenAI-style tool schemas for the agent, DERIVED from `TOOL_SPECS`."""
    return [
        {"type": "function",
         "function": {"name": s.name, "description": s.description, "parameters": s.parameters}}
        for s in TOOL_SPECS.values()
    ]


def run_sql_readonly(world: World, query: str) -> ToolResult:
    """Execute a read-only query. Refuse anything that is not a single SELECT.

    Enforced at the CONNECTION level: a second connection opened through a `mode=ro` URI,
    which SQLite refuses to write through. No keyword matching — a denylist is not a
    security boundary and `WITH x AS (...) DELETE ...` is a thing that exists.
    """
    if not isinstance(query, str) or not query.strip():
        return ToolResult(False, error="query must be a non-empty string")
    try:
        ro = sqlite3.connect(f"file:{world.path}?mode=ro", uri=True)
        ro.row_factory = sqlite3.Row
        try:
            rows = ro.execute(query).fetchall()
        finally:
            ro.close()
    except sqlite3.Error as exc:
        return ToolResult(False, error=f"sqlite: {exc}")
    return ToolResult(True, output=[dict(r) for r in rows])
