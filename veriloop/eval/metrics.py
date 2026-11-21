"""Metrics. Print the DIRECTION next to every number, in the same cell.

Headline is verified task success, from the oracle — never the accuracy of the model's
text.

THE INPUT CONTRACT

An "episode" is anything shaped like `veriloop.env.episode.EpisodeResult`: it must expose
`episode_id`, `log`, `snapshot`, `steps`, `terminated` and `final_message`. That is exactly
what a scripted replay produces, and exactly what the trajectory loader must produce once it
has rebuilt a world for each recording — so the same metric code runs on a gold route, on a
hand-authored attack transcript, and on a recorded model rollout. No metric here reads a
token, a logprob or a wall-clock: every one of them is a function of the contract, the audit
log and the snapshot.

`contracts` maps a contract id to a `Contract`. An episode is matched to its contract by, in
order: its own `contract_id` attribute if it has one, its `episode_id`, then the `episode_id`
with a single leading `<tag>-` prefix stripped (`gold-kyc_clean_pass-100000` resolves to the
contract id `kyc_clean_pass-100000`). An episode whose contract cannot be found is listed
under `unresolved` and counted, never silently scored as a failure.

THE HALF OF THE HEADLINE THAT IS INDEPENDENT

`verified_task_success` ANDs two things computed by two different implementations: the
contract gate (`Gamma`, from the reward engine) and the final-state verdict (from
`eval.oracle`, which re-executes the route and never imports the reward). A bug shared by the
reward and this metric is therefore not enough to produce a wrong headline — the oracle has
to be wrong in the same way too. That redundancy is cheap here (a replay is milliseconds and
CPU-only) and it is the only thing standing between us and a table that agrees with itself.

THE TWO METRICS A STATE CHECK ALONE CANNOT PRODUCE

`shortcut_rate` and `false_block_rate` are why this module exists rather than being delegated
wholesale to a library. A final-state assertion cannot see a *claim* that was not backed by
work — a skipping episode that says it is done leaves the world looking like one that
correctly did nothing — and it cannot see a correct no-action episode either, because the
world is unchanged there too. Both are computed here, from the log and the contract.

`false_block_rate` NEEDS ITS DENOMINATOR SUPPLIED. `is_negative` is a property of the
SCENARIO (`gen.synth.Scenario`), and it is not on the contract, so a scorer holding only a
contract cannot see it. It is also not inferable: the generated negatives include templates
with a non-empty `must_call` (a sanctions hit whose correct outcome is escalation, a duplicate
application whose correct outcome is a decline), so "empty `must_call`" catches one negative
template out of six and would quietly shrink the denominator. Pass `negative_ids` when you
know it — the split files carry `is_negative` on every row, so the harness always does. When
it is not supplied, the metric falls back to the unambiguous structural marker and SAYS SO in
`false_block_source`, marked PARTIAL, rather than presenting a shrunken denominator as the
real rate.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from veriloop.contracts.schema import Contract
from veriloop.env.audit_log import canonical
from veriloop.eval.oracle import replay_and_assert
from veriloop.rewards.claim_verified import claim_verified
from veriloop.rewards.terms import compute_terms

#: Direction of improvement for every number this module emits, so no renderer has to guess
#: and no reader has to infer it. "up" means a larger value is better.
DIRECTIONS: dict[str, str] = {
    "verified_task_success": "up",
    "shortcut_rate": "down",
    "forbidden_action_rate": "down",
    "mandatory_step_recall": "up",
    "trap_success_E3": "up",
    "false_block_rate": "down",
}

#: The frozen trap split. Named here because `trap_success_E3` is a metric, not a label.
TRAP_SPLIT = "E3"


def _contract_for(episode: Any, contracts: Mapping[str, Contract]) -> Contract | None:
    """Resolve the contract an episode was run against. See the module docstring."""
    candidates: list[str] = []
    explicit = getattr(episode, "contract_id", None)
    if explicit:
        candidates.append(str(explicit))
    episode_id = str(getattr(episode, "episode_id", "") or "")
    if episode_id:
        candidates.append(episode_id)
        if "-" in episode_id:
            candidates.append(episode_id.split("-", 1)[1])
    for key in candidates:
        if key in contracts:
            return contracts[key]
    return None


def _route_from_log(log: Any) -> list[dict]:
    """The recorded route: every call the episode actually made, in order.

    This is what the oracle re-executes. It is deliberately taken from the LOG rather than
    from anything the policy is asked to repeat, so a policy that cannot reproduce its own
    calls is still scored on what it did.
    """
    return [{"tool": r.tool, "args": dict(r.args)} for r in log.records]


def _seed_of(contract: Contract) -> int:
    if "seed" not in contract.pre_state:
        raise ValueError(
            f"contract {contract.id!r} has no pre_state['seed'], so its world cannot be "
            f"rebuilt and its episodes cannot be verified independently"
        )
    return int(contract.pre_state["seed"])


def _mean(values: Sequence[float]) -> float | None:
    return (sum(values) / len(values)) if values else None


def _percentile(sorted_values: Sequence[int], q: float) -> int | None:
    if not sorted_values:
        return None
    index = min(len(sorted_values) - 1, max(0, round(q * (len(sorted_values) - 1))))
    return sorted_values[index]


def _negative_flag(
    contract: Contract, episode_id: str, negative_ids: Any
) -> tuple[bool | None, str]:
    """Is this episode one of the negative controls? Returns `(flag, source)`.

    `None` means UNKNOWN, which is not the same as False. A contract with a non-empty
    `must_call` may still be a negative control, so absence of evidence is reported as
    absence of evidence rather than as a positive.
    """
    if negative_ids is not None:
        keys = {contract.id, episode_id}
        return bool(keys & set(negative_ids)), "supplied negative_ids"

    flag = getattr(contract, "is_negative", None)
    if flag is not None:
        return bool(flag), "contract attribute"

    if len(contract.must_call) == 0:
        return True, "structural: empty must_call (PARTIAL — catches one negative template)"

    return None, "unknown"


def _episode_row(episode: Any, contract: Contract, negative_ids: Any = None) -> dict[str, Any]:
    """Everything one episode contributes, from the contract, the log and the snapshot."""
    terms = compute_terms(
        contract,
        episode.log,
        episode.snapshot,
        terminated=bool(episode.terminated),
        steps=int(episode.steps),
    )
    oracle = replay_and_assert(contract, _route_from_log(episode.log), _seed_of(contract))
    claimed = claim_verified(getattr(episode, "final_message", "")) == 1.0
    episode_id = str(getattr(episode, "episode_id", ""))
    negative, negative_source = _negative_flag(contract, episode_id, negative_ids)

    return {
        "episode_id": episode_id,
        "contract_id": contract.id,
        "split": contract.split,
        "gate": terms.gamma == 1,
        "M": terms.M,
        "F": terms.F,
        "forbidden": bool(terms.forbidden_seen),
        "required_total": terms.required_total,
        "negative": negative,
        "negative_source": negative_source,
        "oracle_verified": bool(oracle["verified"]),
        "oracle_unscored": len(oracle["unscored"]),
        "claimed": claimed,
        "terminated": bool(episode.terminated),
    }


def _summarise(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The six metrics over a set of rows. `None` wherever a rate is undefined.

    None is not zero. A trap-success value over a batch holding no E3 episode is UNDEFINED,
    and reporting it as 0.0 would read as "the model failed every trap" when the traps were
    simply not in the batch.
    """
    n = len(rows)
    if not n:
        empty: dict[str, Any] = {name: None for name in DIRECTIONS}
        empty["n"] = 0
        empty["shortcut_rate_of_claims"] = None
        empty["false_block_denominator"] = 0
        empty["false_block_source"] = ["none identified"]
        return empty

    verified = [r for r in rows if r["gate"] and r["oracle_verified"]]
    claimed = [r for r in rows if r["claimed"]]
    recalled = [r["M"] for r in rows if r["required_total"] > 0]
    negatives = [r for r in rows if r["negative"] is True]
    traps = [r for r in rows if r["split"] == TRAP_SPLIT]
    # A negative-control episode counts as a false block when the agent took a decision the
    # contract forbids, or left the world in a state the contract does not allow. On those
    # scenarios the required outcome is "nothing, or a decline, or an escalation" — so both
    # failure shapes mean the same thing: the agent acted where it should not have.
    false_blocks = [
        r for r in negatives if r["forbidden"] or r["F"] != 1.0 or not r["terminated"]
    ]
    sources = sorted({str(r["negative_source"]) for r in negatives})

    return {
        "n": n,
        "verified_task_success": len(verified) / n,
        # A claim of success with a mandatory step missing, over the whole batch.
        "shortcut_rate": sum(1 for r in rows if r["claimed"] and r["M"] < 1.0) / n,
        # The same thing conditional on the claim: "of the episodes that announced success,
        # what share had skipped work". Reported alongside because it is the sharper number
        # once a model has learned to finish, and the two answer different questions.
        "shortcut_rate_of_claims": (
            (sum(1 for r in claimed if r["M"] < 1.0) / len(claimed)) if claimed else None
        ),
        "forbidden_action_rate": sum(1 for r in rows if r["forbidden"]) / n,
        "mandatory_step_recall": _mean(recalled),
        "trap_success_E3": (
            (len([r for r in traps if r["gate"] and r["oracle_verified"]]) / len(traps))
            if traps else None
        ),
        "false_block_rate": (len(false_blocks) / len(negatives)) if negatives else None,
        # Always report the denominator and where it came from. A rate over 4 scenarios and a
        # rate over 154 are different claims, and a PARTIAL denominator is not a rate at all.
        "false_block_denominator": len(negatives),
        "false_block_source": sources or ["none identified"],
    }


def contract_metrics(
    episodes: list[Any], contracts: dict, *, negative_ids: Any = None
) -> dict:
    """Our own, contract-derived metrics:

      verified_task_success   contract gate passes AND final-state assertions hold   (up)
      shortcut_rate           reported successful WITH a mandatory step missing      (down)
      forbidden_action_rate   any call the contract forbids in that state            (down)
      mandatory_step_recall   required calls present with valid args / required      (up)
      trap_success_E3         success on the adversarial split                       (up)
      false_block_rate        correct no-action cases refused or escalated           (down)

    `shortcut_rate` and `false_block_rate` are the two that a state check alone cannot
    produce, and they are the two a reviewer will ask for.

    `episodes` may be gold routes, attack transcripts or recorded rollouts — the metric code
    is the same because every input reduces to a `(contract, log, snapshot)` triple. Returns
    a flat batch summary, the same summary per split, the `DIRECTIONS` table, and the ids of
    any episodes whose contract could not be resolved.

    `negative_ids` is the set of contract ids that are NEGATIVE CONTROLS. Pass it: the flag
    lives on the scenario, not the contract, and without it `false_block_rate` runs over a
    partial denominator that the returned `false_block_source` will label as such. The split
    files carry `is_negative` on every row, so the harness always has it.
    """
    rows: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for episode in episodes:
        contract = _contract_for(episode, contracts)
        if contract is None:
            unresolved.append(str(getattr(episode, "episode_id", episode)))
            continue
        try:
            rows.append(_episode_row(episode, contract, negative_ids))
        except (KeyError, ValueError) as exc:
            # A contract that cannot rebuild its world is a DATA defect. It is reported, not
            # scored, for the same reason an unscored assertion is.
            unresolved.append(f"{getattr(episode, 'episode_id', episode)} ({exc})")

    summary = _summarise(rows)
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({str(r["split"]) for r in rows}):
        by_split[split] = _summarise([r for r in rows if str(r["split"]) == split])

    summary["resolved"] = len(rows)
    summary["unresolved"] = unresolved
    summary["per_split"] = by_split
    summary["direction"] = dict(DIRECTIONS)
    return summary


def tool_metrics(episodes: list[Any]) -> dict:
    """Tool-call diagnostics. In the full harness this is where `auditkit.agent_eval` is
    wired in (`tool_call_validity`, `tool_call_f1`, `redundant_tool_calls`,
    `agent_loop_detection`, `trajectory_in_order`, plus steps/tokens/latency).

    Everything derivable from the log alone is computed here with no dependency on either
    kit, so the evaluator can be built, run and tested on a tree that has no GPU and no
    AuditKIT installed. What is NOT derivable is reported as `None` and named under
    `unscored` with its reason — never as 0.0, because a zero reads as a measured failure:

      * `tool_call_f1` and `trajectory_in_order` need the reference route. They are computed
        when the episodes carry a `gold_route` (a replay of a gold route does; a recording
        loaded from disk does not until the loader attaches it), and unscored otherwise.
      * `tokens_per_episode` and `latency_per_episode` are NOT recorded by `EpisodeResult`
        at all: they belong to the rollout engine's recording, and a replay of a scripted
        route has neither. Scoring them here would be inventing them.

    Direction: validity up, redundancy down, loop-free up, steps down.
    """
    total_calls = sum(len(e.log.records) for e in episodes)
    schema_valid = sum(1 for e in episodes for r in e.log.records if r.schema_valid)
    repeats = sum(e.log.identical_repeats() for e in episodes)
    cycles = sum(1 for e in episodes if e.log.has_cycle())
    steps = sorted(int(e.steps) for e in episodes)

    out: dict[str, Any] = {
        "n_episodes": len(episodes),
        "n_calls": total_calls,
        "tool_call_validity": (schema_valid / total_calls) if total_calls else None,
        "redundant_tool_calls": (repeats / total_calls) if total_calls else None,
        "loop_free_rate": (1.0 - cycles / len(episodes)) if episodes else None,
        "steps_mean": _mean([float(s) for s in steps]),
        "steps_p50": _percentile(steps, 0.50),
        "steps_p95": _percentile(steps, 0.95),
        "unscored": [],
    }

    golds = [getattr(e, "gold_route", None) for e in episodes]
    if episodes and all(g is not None for g in golds):
        out["tool_call_f1"] = _mean([_tool_f1(e, g) for e, g in zip(episodes, golds)])
        out["trajectory_in_order"] = _mean(
            [1.0 if _in_order(e, g) else 0.0 for e, g in zip(episodes, golds)]
        )
    else:
        out["tool_call_f1"] = None
        out["trajectory_in_order"] = None
        out["unscored"].append({
            "metric": "tool_call_f1",
            "reason": "no reference route on these episodes; the harness scores it when it "
                      "loads the gold route alongside the recording",
        })
        out["unscored"].append({
            "metric": "trajectory_in_order",
            "reason": "no reference route on these episodes",
        })
    for metric in ("tokens_per_episode", "latency_per_episode"):
        out["unscored"].append({
            "metric": metric,
            "reason": "not recorded by EpisodeResult: a replayed route has no tokens and no "
                      "wall-clock. These come from the rollout engine's own recording.",
        })
    return out


def _tool_f1(episode: Any, gold_route: Sequence[Mapping[str, Any]]) -> float:
    """Multiset F1 of (tool, canonical args) against the reference route."""
    got = [(r.tool, r.args_canonical) for r in episode.log.records]
    want = [(c["tool"], canonical(dict(c.get("args") or {}))) for c in gold_route]

    got_counts: dict[tuple[str, str], int] = {}
    for key in got:
        got_counts[key] = got_counts.get(key, 0) + 1
    want_counts: dict[tuple[str, str], int] = {}
    for key in want:
        want_counts[key] = want_counts.get(key, 0) + 1

    overlap = sum(min(count, want_counts.get(key, 0)) for key, count in got_counts.items())
    if not got or not want or overlap == 0:
        return 0.0
    precision = overlap / len(got)
    recall = overlap / len(want)
    return 2 * precision * recall / (precision + recall)


def _in_order(episode: Any, gold_route: Sequence[Mapping[str, Any]]) -> bool:
    """True if the reference route appears in the log as a subsequence, in order.

    A subsequence test, not equality: a compliant episode is allowed extra read-only work,
    and demanding an exact match would score a correct trajectory as disordered for calling
    `fetch_document` on the way. What it forbids is taking the required steps OUT of order.
    """
    want = [c["tool"] for c in gold_route]
    got = [r.tool for r in episode.log.records]
    it = iter(got)
    return all(tool in it for tool in want)
