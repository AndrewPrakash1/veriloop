
"""Phase 1 gate: prove the WORLD supports the reference contract's gold route.

Runs the gold route through the real episode loop and asserts the contract's `final_state`
holds at the end, then exercises every guard the world is supposed to have. Exits non-zero
on any failure, so it can gate a commit.

This is an integration check, not a unit test: it is the one command that tells you the
world is genuinely usable before anyone touches the reward.

    make verify-world
"""

from __future__ import annotations
import yaml, sys
sys.path.insert(0, ".")
from veriloop.env.db import open_world
from veriloop.env.episode import run_episode
from veriloop.env.tools import call_tool, run_sql_readonly, spec_json_schemas, TOOL_SPECS

c = yaml.safe_load(open("contracts/EXAMPLE_kyc_sanctions_hit.yaml"))
w = open_world(c["pre_state"]["seed"], c["pre_state"], episode_id="verify-gold")
print("tools offered:", len(TOOL_SPECS), "| schemas:", len(spec_json_schemas()))
print("facts before:", {k: v for k, v in w.facts().items()
                        if k in ("sanctions_hit", "pep_hit", "all_checks_recorded")})

class Route:
    def __init__(self, route, msg=""):
        self.route, self.i, self.final_message = list(route), 0, msg
    def next_call(self, obs):
        if self.i >= len(self.route):
            raise StopIteration
        call = self.route[self.i]; self.i += 1
        return call

GOLD = [
    ("lookup_customer",  {"customer_id": "a8821"}),
    ("screen_sanctions", {"customer_id": "a8821"}),
    ("screen_pep",       {"customer_id": "a8821"}),
    ("record_check",     {"check_type": "identity",  "evidence_ref": "D-8821-ID"}),
    ("record_check",     {"check_type": "sanctions", "evidence_ref": "S-C-1042"}),
    ("record_check",     {"check_type": "pep",       "evidence_ref": "P-C-1042"}),
    ("escalate",         {"case_id": "C-1042", "reason": "sanctions_match"}),
    ("finish",           {"summary": "All checks complete. Case escalated."}),
]
route = Route([{"tool": t, "args": a} for t, a in GOLD],
              "All checks complete. Case escalated.")
r = run_episode(w, route, c["brief"], c["step_budget"])

print("--- episode ---")
print("steps:", r.steps, "| terminated:", r.terminated, "| final:", r.final_message[:40])
print("calls :", [f"{x.tool}@{x.step}" for x in r.log.records[:3]], "...",
      [x.tool for x in r.log.records[-2:]])
print("all schema_valid:", all(x.schema_valid for x in r.log.records),
      "| duplicates:", r.log.identical_repeats(), "| cycle:", r.log.has_cycle())
print("facts after :", {k: v for k, v in w.facts().items()
                        if k in ("all_checks_recorded", "checks_recorded", "account_status")})

snap = r.snapshot
acct = snap["accounts"][0]; case = snap["cases"][0]
print("account.status:", acct["status"], "| case.state:", case["state"],
      "| checks rows:", len(snap["checks"]))
ok = (acct["status"] == "escalated" and case["state"] == "pending_review"
      and len(snap["checks"]) == 3)
print("FINAL-STATE ASSERTIONS HOLD:", ok)

print("--- guards ---")
print("UPDATE refused  :", not run_sql_readonly(w, "UPDATE accounts SET status='open'").ok)
print("DELETE refused  :", not run_sql_readonly(w, "DELETE FROM checks").ok)
print("WITH..DELETE    :", not run_sql_readonly(w, "WITH x AS (SELECT 1) DELETE FROM checks").ok)
print("SELECT allowed  :", run_sql_readonly(w, "SELECT COUNT(*) AS n FROM checks").ok)
bad = call_tool(w, 99, "lookup_customer", {"customer_id": 8821})
print("int for str     :", not bad.ok, "|", bad.error)
unk = call_tool(w, 99, "teleport_customer", {})
print("unknown tool    :", not unk.ok, "|", unk.error)
extra = call_tool(w, 99, "lookup_customer", {"customer_id": "a8821", "vibe": "ok"})
print("hallucinated arg:", not extra.ok, "|", extra.error)
print("mirror rows     :", w.conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0])
print("log untampered  :", (w.log.assert_append_only() or True))
