"""THE LIVE DEMO — runs the real model on the GPU and streams its decisions to a browser.

ON THE POD

    cd /workspace/veriloop
    python3 demo/live_server.py --port 8765

Then open the JupyterLab proxy URL, which needs no extra port exposure because it rides on the
host you already have open:

    <your-jupyter-url>/proxy/8765/

(e.g. https://gpu-7513a.45.195.90.155.nip.io/proxy/8765/ — take the scheme+host from your
JupyterLab tab and append /proxy/8765/.)

WHY THIS IS NOT demo/server.py

`demo/server.py` replays scripted ATTACK ROUTES and deliberately runs no model: it answers
"does the cheat get paid?". This answers the other question — "what does the trained model
actually do?" — by loading A0 (base) and A1 (the SFT adapter) and running real episodes against
a real SQLite world.

WHAT A JUDGE SEES

Pick a scenario, pick an arm, press run. Every tool call appears as the model makes it,
with the model's own words beside it, and then the verdict.

THE VERDICT IS COMPUTED TWICE, ON PURPOSE

  * `rewards.state_verified` — the reward, from the terms.
  * `eval/oracle.py`        — an INDEPENDENT re-derivation that never imports the reward.

`eval/oracle.py` exists because a shared bug between reward and evaluator is exactly the failure
it is there to catch. So the page shows both, and marks disagreement loudly rather than picking
a winner. Two numbers that agree are evidence; one number is a claim.

NOTHING HERE IS SIMULATED. No cached trajectories, no fixtures — the model generates every
token on the GPU, and the database change is real.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_SPLIT = os.path.join(ROOT, "data", "splits", "E1.jsonl")
ADAPTER = os.path.join(ROOT, "outputs", "A1")
ARMS = {"A0": None, "A1": ADAPTER}

#: One episode at a time. There is one GPU, and two concurrent episodes would each crawl and
#: interleave their streams into one incomprehensible log.
GPU = threading.Lock()

POLICIES: dict[str, object] = {}
POLICY_LOCK = threading.Lock()
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()

MAX_STEPS = 12


# --------------------------------------------------------------------------- helpers -----
def normalize_path(path: str) -> str:
    """Strip a jupyter-server-proxy prefix so one server works directly AND behind the proxy.

    Behind the proxy the browser asks for `/proxy/8765/api/meta`; directly it asks for
    `/api/meta`. Normalising here means the page can use plain relative URLs and work in both
    places without knowing where it is mounted.
    """
    return re.sub(r"^/proxy/\d+", "", path) or "/"


def load_scenarios(path: str) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def one_per_template(rows: list[dict]) -> list[int]:
    """Indices of one scenario per template, in order.

    A judge has about ninety seconds. Eight scenarios covering every template beats sixty
    near-identical ones, and it is the same sample the RL probe used.
    """
    seen: set[str] = set()
    picked: list[int] = []
    for index, row in enumerate(rows):
        template = row.get("template", "?")
        if template not in seen:
            seen.add(template)
            picked.append(index)
    return picked


def policy_for(arm: str, temperature: float):
    """Load an arm's policy once and reuse it. Two arms must coexist for a side-by-side."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; known: {sorted(ARMS)}")
    with POLICY_LOCK:
        if arm not in POLICIES:
            from agents.loader import build_policy

            adapter = ARMS[arm]
            # seed=None on purpose: ModelPolicy re-seeds before EVERY generation when a seed is
            # set, which would make the policy deterministic and every run identical. A demo
            # that shows the same episode twice is not a demo.
            POLICIES[arm] = build_policy(
                MAX_STEPS, adapter=adapter, temperature=temperature, seed=None,
            )
    return POLICIES[arm]


class StreamingPolicy:
    """Wraps a policy and appends each call to a shared list AS IT IS MADE.

    `run_episode` drives the policy and only returns when the episode is over, so the only place
    a per-step record can be captured is this seam. The page polls the list, which is why a
    sixty-second episode still feels live.
    """

    def __init__(self, inner, record: list) -> None:
        self.inner = inner
        self.record = record

    def next_call(self, observation):
        prompt = self.inner.render(observation)
        before = len(getattr(self.inner, "completions", []))
        started = time.time()
        call = self.inner.next_call(observation)
        completions = getattr(self.inner, "completions", [])
        completion = completions[-1] if len(completions) > before else ""
        self.record.append({
            "step": len(self.record),
            "tool": str(call.get("tool", "")),
            "args": dict(call.get("args") or {}),
            "completion": completion[:1200],
            "prompt_chars": len(prompt),
            "seconds": round(time.time() - started, 2),
            "generation_error": call.get("_generation_error"),
        })
        return call


# -------------------------------------------------------------------------- the episode ---
def run_job(job_id: str, split: str, index: int, arm: str, temperature: float) -> None:
    job = JOBS[job_id]
    try:
        from veriloop.contracts.schema import Contract
        from veriloop.env.db import open_world
        from veriloop.env.episode import run_episode
        from veriloop.eval.oracle import replay_and_assert
        from veriloop.rewards.state_verified import reward_from_terms
        from veriloop.rewards.terms import compute_terms

        job["stage"] = "loading the model (first run only, ~1-2 min)"
        policy = policy_for(arm, temperature)

        row = load_scenarios(split)[index]
        contract = Contract(**row["contract"])
        pre = dict(contract.pre_state)
        world = open_world(pre["seed"], pre, episode_id=f"demo-{contract.id}")

        job["scenario"] = {
            "scenario_id": row["scenario_id"],
            "template": row.get("template"),
            "is_negative": bool(row.get("is_negative")),
            "brief": contract.brief,
            "must_call": contract.must_call,
            "step_budget": contract.step_budget,
        }

        with GPU:
            job["stage"] = f"running {arm} · {contract.id}"
            started = time.time()
            stream: list[dict] = []
            try:
                result = run_episode(world, StreamingPolicy(policy, stream), contract.brief,
                                     contract.step_budget or MAX_STEPS, agent_name=arm)
            finally:
                # The world must be closed whatever happens, but the snapshot has to be read
                # first: `result.snapshot` is the state the reward is computed from.
                elapsed = time.time() - started
                snapshot = getattr(result, "snapshot", None) if "result" in dir() else None
                world.close()

            # Join the streamed calls with the audit log, which is what carries ok/schema_valid.
            records = list(result.log.records)
            for step, entry in enumerate(stream):
                if step < len(records):
                    entry["ok"] = bool(records[step].ok)
                    entry["schema_valid"] = bool(records[step].schema_valid)
            job["calls"] = stream

            terms = compute_terms(contract, result.log, result.snapshot,
                                  terminated=result.terminated, steps=result.steps)
            reward = reward_from_terms(terms)
            oracle = replay_and_assert(
                contract,
                [{"tool": r.tool, "args": dict(r.args)} for r in result.log.records],
                pre["seed"],
            )
            job["verdict"] = {
                "arm": arm,
                "reward": reward,
                "terms": terms.as_row(),
                "steps": result.steps,
                "terminated": bool(result.terminated),
                "seconds": round(elapsed, 1),
                "oracle_verified": bool(oracle["verified"]),
                "oracle_failures": oracle.get("failures") or [],
                "agree": bool(oracle["verified"]) and reward == 1.0,
                "final_message": (result.final_message or "")[:400],
                "accounts": snapshot.get("accounts") if snapshot else None,
                "checks": snapshot.get("checks") if snapshot else None,
                "cases": snapshot.get("cases") if snapshot else None,
            }
        job["stage"] = "done"
        job["state"] = "done"
    except Exception as exc:                                       # noqa: BLE001
        # A failure must be VISIBLE, with the traceback: this runs in front of judges and a
        # silent 500 would look like a hung page.
        job["state"] = "error"
        job["stage"] = "error"
        job["error"] = f"{type(exc).__name__}: {exc}"
        job["traceback"] = traceback.format_exc()[-2000:]


# ----------------------------------------------------------------------------- routes ----
def meta(split: str) -> dict:
    rows = load_scenarios(split)
    return {
        "split": os.path.basename(split),
        "n": len(rows),
        "arms": {arm: ("base model" if a is None else os.path.relpath(a, ROOT))
                 for arm, a in ARMS.items()},
        "adapter_present": os.path.isdir(ADAPTER),
        "adapter_files": sorted(os.listdir(ADAPTER)) if os.path.isdir(ADAPTER) else [],
        "scenarios": [
            {"index": i, "scenario_id": r["scenario_id"], "template": r.get("template"),
             "is_negative": bool(r.get("is_negative")),
             "brief": r["contract"]["brief"].strip()[:200]}
            for i, r in enumerate(rows)
        ],
        "sample": one_per_template(rows),
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
        path = normalize_path(url.path)
        query = parse_qs(url.query)
        split = (query.get("split") or [DEFAULT_SPLIT])[0]
        try:
            if path in ("/", "/index.html", "/live.html"):
                with open(os.path.join(ROOT, "demo", "live.html"), "rb") as handle:
                    return self._send(200, handle.read(), "text/html; charset=utf-8")
            if path == "/api/meta":
                return self._send(200, meta(split))
            if path == "/api/start":
                index = int((query.get("index") or ["0"])[0])
                arm = (query.get("arm") or ["A1"])[0].upper()
                temperature = float((query.get("temperature") or ["0.8"])[0])
                job_id = f"job-{int(time.time() * 1000)}"
                JOBS[job_id] = {"state": "running", "stage": "queued", "calls": [], "started": time.time()}
                threading.Thread(target=run_job, args=(job_id, split, index, arm, temperature),
                                 daemon=True).start()
                return self._send(200, {"job": job_id})
            if path == "/api/job":
                job_id = (query.get("job") or [""])[0]
                job = JOBS.get(job_id)
                if job is None:
                    return self._send(404, {"error": f"no such job {job_id}"})
                payload = dict(job)
                payload.setdefault("elapsed", round(time.time() - job.get("started", time.time()), 1))
                return self._send(200, payload)
            return self._send(404, {"error": f"not found: {path}"})
        except Exception as exc:                                   # noqa: BLE001
            return self._send(500, {"error": f"{type(exc).__name__}: {exc}",
                                    "traceback": traceback.format_exc()[-1500:]})


def main() -> int:
    parser = argparse.ArgumentParser(description="Live VeriLoop demo — real model, real world.")
    parser.add_argument("--port", type=int, default=int(os.environ.get("VERILOOP_LIVE_PORT", "8765")))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    if not os.path.isdir(ADAPTER):
        print(f"WARNING: {ADAPTER} not found — only A0 (base) will be selectable.", flush=True)
    print(f"loading nothing yet; the first run loads that arm (~1-2 min).")
    print(f"VeriLoop live demo → http://{args.host}:{args.port}/", flush=True)
    print(f"behind JupyterLab's proxy:  <your-jupyter-url>/proxy/{args.port}/", flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
