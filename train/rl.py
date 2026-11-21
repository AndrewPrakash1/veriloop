"""Arms A2-A5 — the comparison. ONE thing changes between runs: `reward_funcs(arm)`.

Same checkpoint, same prompt, same tools, same group size, same steps, same seeds. The reward
is the only switch point, so a difference between two arms cannot be anything else.

WHY RLOO AND NOT GRPO

`configs/base.yaml` names RLOO as the fallback and this file is that branch. RLOO needs no
critic and no reward model: sample G rollouts of the same prompt, score each with the arm's
reward, and use the MEAN OF THE OTHER G-1 as that rollout's baseline,

    A_i = R_i - mean(R_j, j != i)

which is unbiased for the policy gradient and costs one extra reward evaluation per rollout
instead of a whole value network. At 3.3B on one rented card that is the difference between a
run that finishes tonight and one that does not.

WHAT THE GRADIENT IS TAKEN ON

Only the tokens the MODEL produced. Each generation is re-encoded as
`chat_template(prompt) + completion` and log-probs are gathered on the completion span alone —
the prompt is context, not an action. `agents.policy.ModelPolicy.render` builds the prompt and
this file rebuilds the same string through the same chat template, because a one-token
disagreement between what is sampled and what is scored is a silent bug: the loss still falls,
the run still "succeeds", and the gradient is simply wrong.

Log per step: reward, mean steps, duplicate rate, and the run fingerprints. Without those the
run is not reproducible and the ablation is not defensible.

Never point a training run at E3. Enforced below, not just asked for.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Any, Callable

# `python3 train/rl.py` puts TRAIN/ on sys.path[0], not the repo root. Same shim as
# `scripts/verify_world.py`, derived from this file's location rather than the cwd.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Training on E3 would burn the held-out adversarial split the write-up needs. The rule is
#: stated in `train/rl.py`'s own docstring and in WORK_DIVISION.md; this makes it a hard stop
#: rather than a convention someone remembers at 04:00.
FORBIDDEN_SPLITS = ("E3",)


def assert_trainable_split(path: str) -> None:
    """Refuse to train on a held-out eval split."""
    name = os.path.basename(path)
    for forbidden in FORBIDDEN_SPLITS:
        if forbidden in name:
            raise SystemExit(
                f"refusing to train on {name}: {forbidden} is a held-out eval split. "
                f"Training on it destroys the only adversarial measurement in the project. "
                f"Use data/splits/train_rl.jsonl."
            )


# ------------------------------------------------------------------------- config -------
def load_config(path: str) -> dict:
    """Read a YAML arm config, resolving a single level of `extends:`.

    `extends: base.yaml` is how every arm inherits the model, dtype and LoRA shape, so an
    ablation cannot differ in anything but its reward. Resolving it here rather than at the
    call site keeps the arm the one switch point.
    """
    import yaml

    with open(path) as handle:
        config = yaml.safe_load(handle) or {}
    parent = config.pop("extends", None)
    if parent:
        base_path = os.path.join(os.path.dirname(os.path.abspath(path)), parent)
        merged = load_config(base_path)
        # A child's mapping merges over the parent's, so an arm can leave base.yaml alone and
        # still add `data.file`.
        for key, value in config.items():
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = merged[key] | value
            else:
                merged[key] = value
        return merged
    return config


# ------------------------------------------------------------------------ returns -------
def advantages(rewards: list[float]) -> list[float]:
    """RLOO leave-one-out advantages. Unbiased, critic-free, and they sum to zero.

    With fewer than two rollouts there is no other rollout to compare against, so every
    advantage is zero — the honest answer, and it means a group of one costs but teaches
    nothing.
    """
    n = len(rewards)
    if n < 2:
        return [0.0] * n
    total = float(sum(rewards))
    return [float(r) - (total - float(r)) / (n - 1) for r in rewards]


def total_reward(funcs: list[Callable], episode: Any, contract: Any) -> float:
    """Sum the arm's reward functions.

    An arm with none cannot be trained — that is A0 and A1, and the registry returns an empty
    list deliberately. Raising here rather than training on a zero reward and calling it an
    ablation is the same rule as the registry's.
    """
    if not funcs:
        raise ValueError(
            "this arm has no reward functions (A0/A1 are not trained against a reward). "
            "Pass --arm A2, A3, A4 or A5."
        )
    return float(sum(float(f(episode, contract)) for f in funcs))


def completion_slice_start(n_prompt_tokens: int) -> int:
    """Index into the shifted log-prob vector where the COMPLETION begins.

    `logits[:, i]` predicts `input_ids[i+1]`, so the completion's first token sits at shifted
    index `n_prompt_tokens - 1`. An off-by-one here trains happily on the wrong span, so it is
    a named function with its own test.
    """
    return max(int(n_prompt_tokens) - 1, 0)


def duplicate_rate(result: Any) -> float | None:
    """Fraction of calls in an episode that repeat an earlier (tool, args) pair.

    Computed from the log rather than from `compute_terms`, so it costs no extra database
    read. It is the per-step trace of the reward's duplicate term, and the reason to log it
    is that a policy can raise its reward by shortening an episode while quietly looping.
    """
    records = list(getattr(getattr(result, "log", None), "records", []) or [])
    if not records:
        return None
    seen: set[tuple] = set()
    repeats = 0
    for record in records:
        key = (record.tool, json.dumps(dict(record.args), sort_keys=True, default=str))
        if key in seen:
            repeats += 1
        seen.add(key)
    return round(repeats / len(records), 4)


# ------------------------------------------------------------------------ counting ------
def planned_episodes(steps: int, batch_size: int, group_size: int) -> int:
    """How many episodes a run will roll out. Read this before starting one."""
    return int(steps) * int(batch_size) * int(group_size)


def plan_line(steps: int, batch_size: int, group_size: int, *,
              seconds_per_episode: float | None, inr_per_hour: float = 92.0) -> str:
    """The plan, in the units that decide whether to press enter."""
    episodes = planned_episodes(steps, batch_size, group_size)
    line = (f"{steps} steps x {batch_size} prompts x {group_size} rollouts = "
            f"{episodes} episodes")
    if seconds_per_episode:
        minutes = episodes * seconds_per_episode / 60.0
        line += (f"  ~{minutes:.0f} min at {seconds_per_episode:.1f}s/episode"
                 f"  ~Rs {minutes / 60.0 * inr_per_hour:.0f}")
    return line


# ------------------------------------------------------------------------- rollouts -----
class TurnRecorder:
    """Wraps a policy and records (prompt, completion) for every turn it produces.

    `run_episode` drives the policy and only ever sees the parsed call, so the text the
    gradient must be taken on has to be captured at that seam. One wrapper, one place — the
    same rule the rest of the repo uses for translation layers.
    """

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.turns: list[tuple[str, str]] = []

    def next_call(self, observation: Any) -> Any:
        prompt = self.policy.render(observation)
        before = len(getattr(self.policy, "completions", []))
        call = self.policy.next_call(observation)
        completions = getattr(self.policy, "completions", [])
        # Read the completion by index rather than trusting `last_completion`: this survives
        # ModelPolicy's generation-failure path, which appends "" and still returns a call.
        completion = completions[-1] if len(completions) > before else ""
        self.turns.append((prompt, completion))
        return call


def rollout_group(policy: Any, scenario: dict, group_size: int, *, max_steps: int) -> list[dict]:
    """Run `group_size` independent episodes of ONE scenario. One dict per rollout."""
    from veriloop.contracts.schema import Contract
    from veriloop.env.db import open_world
    from veriloop.env.episode import run_episode

    contract = Contract(**scenario["contract"])
    pre = dict(contract.pre_state)
    budget = max_steps or contract.step_budget
    out: list[dict] = []
    for index in range(group_size):
        recorder = TurnRecorder(policy)
        world = open_world(pre["seed"], pre, episode_id=f"rl-{contract.id}-{index}")
        try:
            started = time.time()
            result = run_episode(world, recorder, contract.brief, budget, agent_name="rl")
            elapsed = time.time() - started
        finally:
            world.close()
        out.append({
            "rollout": index,
            "contract": contract,
            "result": result,
            "turns": list(recorder.turns),
            "seconds": elapsed,
        })
    return out


# -------------------------------------------------------------------------- scoring -----
def completion_logprob_sum(model: Any, tokenizer: Any, prompt: str, completion: str, *,
                           max_length: int) -> tuple[Any, int]:
    """Sum of log P(completion tokens | chat-templated prompt). Differentiable.

    The prompt is rebuilt through the SAME chat template `ModelPolicy._generate` uses; if
    those two ever disagree, the gradient is taken on a sequence the model never saw.
    """
    import torch

    device = next(model.parameters()).device
    zero = torch.zeros((), device=device)
    if not completion.strip():
        return zero, 0

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
    if not completion_ids:
        return zero, 0

    if len(completion_ids) >= max_length:
        # The action is what must survive truncation, so trim the completion's tail — never
        # its head — and drop the prompt first.
        completion_ids = completion_ids[:max_length]
        prompt_ids = []
    else:
        room = max_length - len(completion_ids)
        if len(prompt_ids) > room:
            prompt_ids = prompt_ids[len(prompt_ids) - room:]

    ids = torch.tensor([list(prompt_ids) + list(completion_ids)], dtype=torch.long, device=device)
    logits = model(input_ids=ids).logits[:, :-1]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    picked = logprobs.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0]
    span = picked[completion_slice_start(len(prompt_ids)):]
    return span.sum(), int(span.numel())


def reward_of(funcs: list[Callable], rollout: dict) -> float:
    return total_reward(funcs, rollout["result"], rollout["contract"])


# ---------------------------------------------------------------------------- train -----
def train(args: argparse.Namespace) -> dict:
    import torch

    from agents.loader import build_policy, load_model, load_tokenizer
    from veriloop.rewards.registry import reset_random, reward_funcs

    config = load_config(args.config) if args.config else {}
    arm = str(args.arm).upper()
    funcs = reward_funcs(arm)
    if not funcs:
        raise SystemExit(
            f"arm {arm} is not trained against a reward (A0/A1 are the base model and the SFT "
            f"cold start). Use --arm A2/A3/A4/A5."
        )

    data_file = args.data or (config.get("data") or {}).get("file") or "data/splits/train_rl.jsonl"
    assert_trainable_split(data_file)

    steps = args.steps if args.steps is not None else int(config.get("steps", 300))
    group_size = (args.group_size if args.group_size is not None
                  else int(config.get("rollout_group_size", 8)))
    batch_size = (args.batch_size if args.batch_size is not None
                  else int(config.get("batch_size", 2)))
    rollout = config.get("rollout") or {}
    max_steps = int(args.max_episode_steps or rollout.get("max_steps", 12))
    temperature = float(rollout.get("temperature", 0.8))
    max_length = int(args.max_length)
    lr = args.lr if args.lr is not None else float(config.get("lr", 1e-6))
    model_id = args.model_id or str(config.get("model", "CohereLabs/tiny-aya-global"))
    dtype = str(config.get("dtype", "bf16"))

    init_adapter = args.init_adapter or config.get("init_adapter")
    if init_adapter and not os.path.isdir(init_adapter):
        raise SystemExit(
            f"--init-adapter {init_adapter!r} does not exist.\n"
            f"The arm configs name `outputs/A1_sft`; if you trained to `outputs/A1`, pass\n"
            f"    --init-adapter outputs/A1"
        )

    scenarios = [json.loads(line) for line in open(data_file) if line.strip()]
    if not scenarios:
        raise SystemExit(f"{data_file} has no scenarios")

    print("=" * 74)
    print(f"arm {arm}   reward: {[getattr(f, '__name__', repr(f)) for f in funcs]}")
    print(f"data {data_file} ({len(scenarios)} scenarios)   adapter: {init_adapter or '(BASE)'}")
    if args.probe:
        print(f"PROBE: {args.probe} prompt(s) x {group_size} rollouts, no optimizer, no save")
    else:
        print(f"  {plan_line(steps, batch_size, group_size, seconds_per_episode=None)}")
    print("=" * 74, flush=True)

    random.seed(args.seed)
    reset_random(args.seed)                 # A5's coin flips, re-seeded per run
    torch.manual_seed(args.seed)

    tokenizer = load_tokenizer(model_id)
    model = load_model(model_id, adapter=init_adapter, dtype=dtype)

    if args.probe:
        # TRAP 1: seed=None. `ModelPolicy._generate` re-seeds before EVERY call, so a fixed
        # seed makes all G rollouts of a group the identical episode, every advantage exactly
        # 0.0, and training does nothing while printing a falling loss.
        policy = build_policy(max_steps, model=model, tokenizer=tokenizer,
                              temperature=temperature, seed=None)
        return probe(policy, scenarios, group_size, max_steps, args, funcs)

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = config.get("lora") or {}
    model = get_peft_model(model, LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])),
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr)

    policy = build_policy(max_steps, model=model, tokenizer=tokenizer,
                          temperature=temperature, seed=None)

    order = list(range(len(scenarios)))
    random.Random(args.seed).shuffle(order)
    history: list[dict] = []
    started = time.time()

    for step in range(steps):
        step_started = time.time()
        chosen = [scenarios[order[(step * batch_size + b) % len(order)]]
                  for b in range(batch_size)]

        groups: list[dict] = []
        for scenario in chosen:
            # TRAP 2: sampling needs the KV cache. `prepare_model_for_kbit_training` sets
            # use_cache=False for gradient checkpointing, and leaving it off makes every
            # generation quadratic in the prompt length — ~10 of them per episode on ~1.5k
            # token prompts.
            model.config.use_cache = True
            rollouts = rollout_group(policy, scenario, group_size, max_steps=max_steps)
            groups.append({
                "scenario": scenario,
                "rollouts": rollouts,
                "rewards": [reward_of(funcs, r) for r in rollouts],
            })

        all_rewards = [r for g in groups for r in g["rewards"]]
        losses = 0.0
        turns_scored = 0
        zero_signal = 0

        model.config.use_cache = False
        optimizer.zero_grad(set_to_none=True)
        for group in groups:
            adv = advantages(group["rewards"])
            if all(abs(a) < 1e-9 for a in adv):
                # Every rollout scored the same, so the advantage is exactly zero by
                # construction: a backward pass would buy compute and nothing else.
                zero_signal += 1
                continue
            for rollout_result, advantage in zip(group["rollouts"], adv):
                turns = rollout_result["turns"]
                if not turns:
                    continue
                # A longer episode divides the same group advantage over more turns, so the
                # per-turn weight is normalised by the episode's length and by the number of
                # groups, which keeps the step size independent of the rollout shape.
                weight = advantage / (len(turns) * len(groups))
                for prompt, completion in turns:
                    logprob, n_tokens = completion_logprob_sum(
                        model=model, tokenizer=tokenizer, prompt=prompt,
                        completion=completion, max_length=max_length,
                    )
                    if n_tokens == 0:
                        continue
                    loss = -(logprob / n_tokens) * weight
                    loss.backward()
                    losses += float(loss.detach())
                    turns_scored += 1
        optimizer.step()

        steps_taken = [r["result"].steps for g in groups for r in g["rollouts"]]
        dupes = [d for g in groups for r in g["rollouts"]
                 if (d := duplicate_rate(r["result"])) is not None]
        record = {
            "step": step + 1,
            "episodes": sum(len(g["rollouts"]) for g in groups),
            "mean_reward": (sum(all_rewards) / len(all_rewards)) if all_rewards else None,
            "reward_min": min(all_rewards) if all_rewards else None,
            "reward_max": max(all_rewards) if all_rewards else None,
            "reward_std": _std(all_rewards),
            "mean_steps": (sum(steps_taken) / len(steps_taken)) if steps_taken else None,
            "mean_duplicate_rate": (sum(dupes) / len(dupes)) if dupes else None,
            "turns_scored": turns_scored,
            "groups_with_zero_signal": zero_signal,
            "loss_sum": round(losses, 6),
            "seconds": round(time.time() - step_started, 2),
            "elapsed": round(time.time() - started, 1),
        }
        history.append(record)
        print(f"step {record['step']:3d}/{steps}  reward {_fmt(record['mean_reward'])} "
              f"[{_fmt(record['reward_min'])},{_fmt(record['reward_max'])}]  "
              f"steps {_fmt(record['mean_steps'])}  dup {_fmt(record['mean_duplicate_rate'])}  "
              f"turns {turns_scored}  zerosig {zero_signal}  "
              f"{record['seconds']}s  total {record['elapsed']}s", flush=True)

        if args.out and record["step"] % max(1, args.save_every) == 0:
            _save(model, tokenizer, args.out)

    if args.out:
        _save(model, tokenizer, args.out)

    metrics = {
        "arm": arm,
        "algorithm": "rloo",
        "init_adapter": init_adapter,
        "data": data_file,
        "steps": steps,
        "batch_size": batch_size,
        "group_size": group_size,
        "planned_episodes": planned_episodes(steps, batch_size, group_size),
        "lr": lr,
        "seed": args.seed,
        "temperature": temperature,
        "history": history,
        "first_mean_reward": history[0]["mean_reward"] if history else None,
        "final_mean_reward": history[-1]["mean_reward"] if history else None,
        "runtime_seconds": round(time.time() - started, 1),
        "fingerprints": _fingerprints(args.config),
    }
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "rl_metrics.json"), "w") as handle:
            json.dump(metrics, handle, indent=2, default=str)
        print(f"\nwrote {args.out}/rl_metrics.json", flush=True)
    print(json.dumps({k: v for k, v in metrics.items() if k != "history"}, indent=2, default=str))
    return metrics


def probe(policy: Any, scenarios: list[dict], group_size: int, max_steps: int,
          args: argparse.Namespace, funcs: list[Callable]) -> dict:
    """Rollouts and rewards, no optimizer. The Rs-4 way to find a bug.

    The question it answers is not "is the code right" but "does the reward separate these
    rollouts at all". A group where every rollout scores the same contributes exactly nothing
    to the gradient, so a probe that shows all-zero advantages is telling you to look at the
    reward before paying for a run.
    """
    print("\nPROBE — no gradients taken, nothing saved\n", flush=True)
    report: list[dict] = []
    for scenario in scenarios[: args.probe]:
        started = time.time()
        rollouts = rollout_group(policy, scenario, group_size, max_steps=max_steps)
        rewards = [reward_of(funcs, r) for r in rollouts]
        adv = advantages(rewards)
        per = time.time() - started
        report.append({
            "scenario_id": scenario["scenario_id"],
            "template": scenario.get("template"),
            "rewards": [round(r, 4) for r in rewards],
            "advantages": [round(a, 4) for a in adv],
            "turns": [len(r["turns"]) for r in rollouts],
            "terminated": [bool(r["result"].terminated) for r in rollouts],
            "duplicate_rate": [duplicate_rate(r["result"]) for r in rollouts],
            "seconds": round(per, 2),
        })
        print(f"{scenario['scenario_id']}  ({scenario.get('template')})")
        print(f"  rewards      {report[-1]['rewards']}")
        print(f"  advantages   {report[-1]['advantages']}")
        print(f"  turns        {report[-1]['turns']}   terminated {report[-1]['terminated']}")
        print(f"  dup rate     {report[-1]['duplicate_rate']}")
        print(f"  {report[-1]['seconds']}s for {group_size} rollouts "
              f"= {per / group_size:.2f}s/episode\n", flush=True)

    if report:
        live = [r for r in report if any(a != 0.0 for a in r["advantages"])]
        total_rollouts = group_size * len(report)
        spe = sum(r["seconds"] for r in report) / total_rollouts
        print("=" * 74)
        print(f"{len(live)}/{len(report)} prompt(s) produced a NON-ZERO advantage.")
        print(plan_line(args.steps or 30, args.batch_size or 2, group_size,
                        seconds_per_episode=spe))
        print("=" * 74)
        if not live:
            print("WARNING: every group scored identically. The reward is not separating these")
            print("rollouts, so the gradient is exactly zero — check the reward, then train.")
        print(f"\nMeasured {spe:.2f}s/episode. Use that number, not an estimate, to size the run.")
    return {"probe": report}


# --------------------------------------------------------------------------- utils ------
def _std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return round((sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5, 4)


def _fmt(value: Any) -> str:
    return " n/a " if value is None else f"{float(value):+.3f}"


def _save(model: Any, tokenizer: Any, out: str) -> None:
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    print(f"  saved adapter -> {out}", flush=True)


def _fingerprints(config_path: str | None) -> dict:
    import hashlib
    import subprocess

    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                      stderr=subprocess.DEVNULL).strip()
    except Exception:                                             # noqa: BLE001
        sha = "no-git-tree"
    out: dict[str, Any] = {"git_sha": sha}
    for label, path in (("config_sha256", config_path),):
        if path and os.path.exists(path):
            out[label] = hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]
    try:
        from agents.prompts import SYSTEM

        out["prompt_sha256"] = hashlib.sha256(SYSTEM.encode()).hexdigest()[:16]
    except Exception:                                             # noqa: BLE001
        pass
    return out


def _parser() -> argparse.ArgumentParser:
    """Exposed so a test can assert defaults without a GPU."""
    parser = argparse.ArgumentParser(
        description="RLOO over an arm's reward, from the SFT checkpoint.",
        epilog=(
            "Validate for ~Rs 4 before paying for a run:\n"
            "  python3 train/rl.py --arm A3 --init-adapter outputs/A1 --probe 2\n"
            "Then train:\n"
            "  python3 train/rl.py --arm A3 --init-adapter outputs/A1 --steps 30 "
            "--out outputs/A3_state"
        ),
    )
    parser.add_argument("--config", default="configs/grpo_state.yaml")
    parser.add_argument("--arm", default="A3", help="A2, A3, A4 or A5 (A0/A1 are not trained)")
    parser.add_argument("--init-adapter", default=None, help="the SFT checkpoint to start from")
    parser.add_argument("--data", default=None, help="override the config's data file")
    parser.add_argument("--out", default=None, help="where to save the adapter and metrics")
    parser.add_argument("--steps", type=int, default=None, help="optimizer steps")
    parser.add_argument("--group-size", type=int, default=None, help="rollouts per prompt (G)")
    parser.add_argument("--batch-size", type=int, default=None, help="prompts per step (B)")
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-episode-steps", type=int, default=None,
                        help="tool-call budget per episode (default: the contract's)")
    parser.add_argument("--model-id", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--probe", type=int, default=0,
                        help="roll out N prompts x G, print rewards, take no gradients")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.probe and args.out:
        raise SystemExit("--probe takes no optimizer and saves nothing; drop --out.")
    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
