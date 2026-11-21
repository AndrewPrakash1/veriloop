"""The demo backend. Stdlib only, no GPU, no model, no network, no API key.

WHAT THIS SERVES

  GET /                the single demo page
  GET /api/meta        the scenario, its contract, and the route catalogue
  GET /api/replay?id=  replay ONE route against a FRESH world and return what happened

WHY IT EXISTS

The submission's evidence is tables. Judges do not watch a table. So the demo answers the
one question the tables cannot: *what actually happens to the case, and did the agent do
the work?* It replays a route live — real tool calls, real SQLite, real state change — and
shows the resulting world next to what the contract demanded.

WHAT IT DELIBERATELY DOES NOT DO

It does not compute the reward. The verdict columns come from `audit/reward_hackability.json`
(Person 2, Phase 2) and read "unmeasured" until that file exists. The ✓/✗ checklist beside
the world state is a DISPLAY AID comparing two real things — the world and the contract —
and must never be presented as the score. Two implementations of the score that disagree
would put a number on screen contradicting the write-up, in front of judges.
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import yaml  # noqa: E402

from veriloop.env.db import open_world        # noqa: E402
from veriloop.env.episode import run_episode  # noqa: E402
from veriloop.env.tools import TOOL_SPECS     # noqa: E402

CONTRACTS = os.path.join(ROOT, "contracts", "EXAMPLE_kyc_sanctions_hit.yaml")
ATTACKS = os.path.join(ROOT, "audit", "attacks.yaml")
AUDIT_JSON = os.path.join(ROOT, "audit", "reward_hackability.json")


class Route:
    """Adapts a scripted route to the Policy protocol `run_episode` expects."""

    def __init__(self, calls):
        self.calls = [{"tool": c["tool"], "args": c.get("args", {})} for c in calls]
        self.i = 0

    def next_call(self, observation):
        if self.i >= len(self.calls):
            raise StopIteration
        call = self.calls[self.i]
        self.i += 1
        return call


def _load():
    with open(CONTRACTS) as f:
        contract = yaml.safe_load(f)
    with open(ATTACKS) as f:
        attacks = yaml.safe_load(f)["attacks"]
    audit = {}
    if os.path.exists(AUDIT_JSON):
        with open(AUDIT_JSON) as f:
            for row in json.load(f).get("rows", []):
                audit[row["attack_id"]] = row
    return contract, attacks, audit


def meta():
    contract, attacks, audit = _load()
    return {
        "scenario": contract["id"],
        "brief": contract["brief"].strip(),
        "domain": contract["domain"],
        "step_budget": contract["step_budget"],
        "min_steps": contract["min_steps"],
        "must_call": contract["must_call"],
        "must_not_call": contract["must_not_call"],
        "final_state": contract["final_state"],
        "tools": [{"name": s.name, "description": s.description,
                   "mutates": s.mutates, "terminal": s.is_terminal}
                  for s in TOOL_SPECS.values()],
        "routes": [{"id": a["id"], "probes": a["probes"],
                    "non_trivial": a.get("non_trivial", True),
                    "final_message": a.get("final_message", ""),
                    "n_calls": len(a.get("transcript") or []),
                    "reward": audit.get(a["id"])} for a in attacks],
        "audit_present": bool(audit),
    }


def replay(route_id: str):
    contract, attacks, audit = _load()
    attack = next((a for a in attacks if a["id"] == route_id), None)
    if attack is None:
        return {"error": f"no such route: {route_id}"}

    pre = contract["pre_state"]
    world = open_world(pre["seed"], pre, episode_id=f"demo-{route_id}")
    before = world.snapshot()
    result = run_episode(world, Route(attack.get("transcript") or []),
                         contract["brief"], contract["step_budget"])
    after = result.snapshot

    def changed(table, key, col):
        b = {r[key]: r[col] for r in before[table]}
        a = {r[key]: r[col] for r in after[table]}
        return [{"key": k, "before": b.get(k), "after": a.get(k)}
                for k in sorted(set(b) | set(a)) if b.get(k) != a.get(k)]

    import sqlite3
    world.close()

    return {
        "route": {"id": attack["id"], "probes": attack["probes"],
                  "non_trivial": attack.get("non_trivial", True)},
        "claimed": attack.get("final_message", ""),
        "steps": result.steps,
        "terminated": result.terminated,
        "calls": [
            {"step": r.step, "tool": r.tool, "args": dict(r.args),
             "ok": r.ok, "schema_valid": r.schema_valid}
            for r in result.log.records
        ],
        "duplicates": result.log.identical_repeats(),
        "cycled": result.log.has_cycle(),
        "checks": after["checks"],
        "accounts": after["accounts"],
        "cases": after["cases"],
        "changed": {
            "accounts": changed("accounts", "account_id", "status"),
            "cases": changed("cases", "case_id", "state"),
        },
        "reward": audit.get(attack["id"]),
        "audit_present": bool(audit),
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code, body, ctype="application/json"):
        raw = body if isinstance(body, bytes) else json.dumps(body, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(raw)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        url = urlparse(self.path)
        try:
            if url.path in ("/", "/index.html"):
                with open(os.path.join(ROOT, "demo", "index.html"), "rb") as f:
                    return self._send(200, f.read(), "text/html; charset=utf-8")
            if url.path == "/api/meta":
                return self._send(200, meta())
            if url.path == "/api/replay":
                rid = (parse_qs(url.query).get("id") or [""])[0]
                return self._send(200, replay(rid))
            return self._send(404, {"error": "not found"})
        except Exception as exc:                      # a failure must be VISIBLE, not silent
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main():
    port = int(os.environ.get("VERILOOP_DEMO_PORT", "8765"))
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"VeriLoop demo → http://127.0.0.1:{port}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
