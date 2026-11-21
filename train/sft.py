"""Arm A1 — SFT cold start on gold trajectories.

Purpose is FORMAT ONLY: teach the tool-call syntax and the episode shape. Behaviour comes
from the RL arms. Keeping this stage narrow is what makes the RL comparison meaningful.

After training, probe the per-scenario pass rate and keep scenarios the SFT model solves
10-90% of the time — RL needs headroom, not free wins and not impossible tasks.

Fallback: AgentTune's SFT rails, or plain `transformers` Trainer + PEFT if the full
AgentTune install is fighting the environment. Record which one ran in the run config.

---

WHAT THIS ACTUALLY TRAINS ON

One example per STEP, not per episode. For a gold route of 8 calls there are 8 examples: at
step *k* the prompt is the brief plus the k tool results already produced, and the target is
the k-th call of the gold route. That is the same next-action decision the model faces at
inference, so the loss lands on exactly the thing being evaluated.

Two invariants hold the whole thing together, and both are asserted in
`tests/test_train_sft.py`:

  1. The training prompt is produced by `agents.policy.render_turn` — THE SAME function the
     policy uses at inference. A second prompt builder here would train the model on a format
     nobody scores, and the resulting "the model is bad" would be a bug in this file.
  2. Every target is the gold call serialised as the JSON the parser already reads, so a round
     trip proves the model is being taught something the harness can consume.

The observations come from the REAL loop: `RecordingPolicy` wraps the gold route and captures
the observation `run_episode` hands it at each step, including the tool OUTPUTS that the audit
log does not store. So the history in a training prompt is built by the same code that builds
it at inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from typing import Any, Sequence

# Running this file directly — `python3 train/sft.py`, which is exactly what `make sft` does —
# puts TRAIN/ on sys.path[0] and NOT the repo root, so `import veriloop` fails no matter what
# the working directory is. Same shim as `scripts/verify_world.py`, except derived from THIS
# file's location rather than from the cwd, so it also works if you run it from elsewhere.
if __package__ in (None, ""):
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from veriloop.env.episode import RoutePolicy, run_episode

#: `configs/base.yaml`. A default only, so this script runs standalone; whichever value
#: actually ran is written into `run_config.json`.
DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"


class RecordingPolicy:
    """Replays a route AND records every observation the episode loop handed it.

    This is how the SFT prompts get their history without a second implementation of the
    observation-building rules: the loop builds them, this keeps them.
    """

    def __init__(self, route: Sequence[dict]) -> None:
        self.route = list(route)
        self._inner = RoutePolicy(self.route)
        self.observations: list[dict] = []

    def next_call(self, observation: Any) -> Any:
        # A deep copy, not a reference: run_episode builds a NEW observation each step, but
        # copying makes that guarantee ours rather than something we merely rely on.
        self.observations.append(json.loads(json.dumps(dict(observation), default=str)))
        return self._inner.next_call(observation)


def load_scenarios(path: str, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[:limit] if limit else rows


def target_text(call: dict) -> str:
    """Serialise one gold call as the JSON the parser reads when it scores a model turn."""
    return json.dumps({"tool": call["tool"], "args": dict(call.get("args") or {})},
                      sort_keys=True, separators=(", ", ": "))


def field(scenario: Any, name: str) -> Any:
    """Read a scenario attribute from EITHER shape.

    A scenario exists in two forms in this repo: the `Scenario` dataclass the generator
    produces (`gen.synth`), and the JSONL row a split file carries. They hold the same content,
    so anything that consumes a scenario should take both rather than forcing a caller to
    convert — the conversion is where a field quietly goes missing.
    """
    if isinstance(scenario, dict):
        return scenario[name]
    return getattr(scenario, name)


def contract_of(scenario: Any) -> Any:
    """The `Contract` for a scenario, whether it is already one or still a mapping."""
    from veriloop.contracts.schema import Contract

    raw = field(scenario, "contract")
    return raw if isinstance(raw, Contract) else Contract(**raw)


def build_examples(scenarios: list[dict], *, max_steps: int | None = None) -> list[dict]:
    """One training example per gold step. Torch-free: importable and testable on CPU.

    Returns `[{"scenario_id", "step", "prompt", "target"}]`.
    """
    from agents.policy import render_turn
    from veriloop.env.db import open_world

    examples: list[dict] = []
    for scenario in scenarios:
        contract = contract_of(scenario)
        budget = max_steps or contract.step_budget
        pre = dict(contract.pre_state)
        gold_route = list(field(scenario, "gold_route"))
        scenario_id = field(scenario, "scenario_id")
        policy = RecordingPolicy(gold_route)
        world = open_world(pre["seed"], pre, episode_id=f"sft-{contract.id}")
        try:
            result = run_episode(world, policy, contract.brief, budget, agent_name="gold")
        finally:
            world.close()

        if not result.terminated:
            # Gold routes are verified at generation time, so this cannot normally happen.
            # Skipped LOUDLY rather than silently teaching a truncated episode.
            print(f"  SKIP {scenario_id}: gold route did not terminate")
            continue
        route = gold_route[: len(policy.observations)]
        for step, (observation, call) in enumerate(zip(policy.observations, route)):
            examples.append({
                "scenario_id": scenario_id,
                "step": step,
                "prompt": render_turn(observation, max_steps=budget),
                "target": target_text(call),
            })
    return examples


def fingerprints() -> dict:
    """Recorded into every run config. Without these the run is not reproducible."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                             # noqa: BLE001
        sha = "unknown"
    from agents.prompts import SYSTEM

    return {
        "git_sha": sha,
        "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest()[:16],
    }


# ---------------------------------------------------------------------------- dataset -----
def make_dataset(examples: list[dict], tokenizer: Any, max_length: int = 2048) -> Any:
    """Tokenise with ASSISTANT-ONLY loss masking.

    The mask is the token length of the prompt rendered on its own: everything the model is
    given is masked, everything it must produce is supervised. Without it the model is trained
    to reproduce the brief and the tool descriptions — most of the sequence, none of the task.
    """
    import torch

    class _ToolCallDataset(torch.utils.data.Dataset):
        def __init__(self) -> None:
            self.rows: list[dict] = []
            for example in examples:
                messages = [
                    {"role": "user", "content": example["prompt"]},
                    {"role": "assistant", "content": example["target"]},
                ]
                full = tokenizer.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=False)
                prompt_only = tokenizer.apply_chat_template(
                    messages[:1], tokenize=False, add_generation_prompt=True)
                full_ids = tokenizer(full, truncation=True, max_length=max_length,
                                     add_special_tokens=False)["input_ids"]
                prompt_len = len(tokenizer(prompt_only, truncation=True, max_length=max_length,
                                           add_special_tokens=False)["input_ids"])
                labels = list(full_ids)
                prompt_len = min(prompt_len, len(labels))
                labels[:prompt_len] = [-100] * prompt_len
                self.rows.append({"input_ids": full_ids, "labels": labels})

        def __len__(self) -> int:
            return len(self.rows)

        def __getitem__(self, index: int) -> dict:
            return self.rows[index]

    return _ToolCallDataset()


def make_collator(tokenizer: Any) -> Any:
    """Right-pad to the longest row in the batch, padding labels with -100."""
    import torch

    def collate(rows: list[dict]) -> dict:
        width = max(len(r["input_ids"]) for r in rows)
        pad_id = (tokenizer.pad_token_id if tokenizer.pad_token_id is not None
                  else tokenizer.eos_token_id)
        input_ids, labels, masks = [], [], []
        for row in rows:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [pad_id] * pad)
            labels.append(row["labels"] + [-100] * pad)
            masks.append([1] * len(row["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def supported_arguments(cls: Any, kwargs: dict) -> tuple[dict, list[str]]:
    """Drop TrainingArguments fields this transformers build does not accept.

    transformers renames and REMOVES fields between minor versions (`warmup_ratio` is gone in
    5.x; `evaluation_strategy` became `eval_strategy`), and an unexpected keyword is a hard
    error at construction — on a rented pod that is a whole session lost to a one-word rename.
    Filtering against the signature keeps one script working on the pod's build and on a fresh
    one. Returns `(accepted, dropped)`; the dropped names are logged, not hidden.
    """
    import inspect

    accepted = set(inspect.signature(cls.__init__).parameters)
    return {k: v for k, v in kwargs.items() if k in accepted}, sorted(set(kwargs) - accepted)


def train(config: dict) -> dict:
    """One epoch of QLoRA over the gold steps. Returns the metrics actually observed."""
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Trainer, TrainingArguments

    from agents.loader import load_model, load_tokenizer

    scenarios = load_scenarios(config["data"], config.get("limit"))
    print(f"building SFT examples from {len(scenarios)} scenarios...")
    examples = build_examples(scenarios)
    print(f"  {len(examples)} step-level examples")

    tokenizer = load_tokenizer(config["model"])
    model = load_model(config["model"], load_in_4bit=config["load_in_4bit"],
                       dtype=config["dtype"])
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=config["lora_r"], lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=list(config["target_modules"]), task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    bf16 = config["dtype"] == "bf16" and torch.cuda.is_bf16_supported()
    wants = {
        "output_dir": config["out"],
        "num_train_epochs": config["epochs"],
        "per_device_train_batch_size": config["batch_size"],
        "gradient_accumulation_steps": config["grad_accum"],
        "gradient_checkpointing": True,
        "learning_rate": config["lr"],
        "lr_scheduler_type": "cosine",
        # `warmup_steps`, not `warmup_ratio`: the ratio field is gone in transformers 5.x.
        "warmup_steps": config["warmup_steps"],
        "logging_steps": 5,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "bf16": bf16,
        "fp16": not bf16,
        "seed": config["seed"],
        "report_to": [],
    }
    args_kwargs, dropped = supported_arguments(TrainingArguments, wants)
    if dropped:
        print(f"  note: this transformers build does not accept {dropped} — ignored")

    started = time.time()
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**args_kwargs),
        train_dataset=make_dataset(examples, tokenizer, max_length=config["max_length"]),
        data_collator=make_collator(tokenizer),
    )
    outcome = trainer.train()
    elapsed = time.time() - started

    trainer.save_model(config["out"])
    tokenizer.save_pretrained(config["out"])
    steps = int(getattr(outcome, "global_step", 0)) or 1
    metrics = {
        "arm": "A1",
        "engine": "transformers.Trainer + peft",   # recorded: the fallback, not AgentTune
        "examples": len(examples),
        "scenarios": len(scenarios),
        "steps": steps,
        "train_loss": float(getattr(outcome, "training_loss", float("nan"))),
        "seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / steps, 2),
        "dtype": config["dtype"],
        "bf16_used": bf16,
        "max_length": config["max_length"],
        "fingerprints": fingerprints(),
        "config": config,
    }
    with open(os.path.join(config["out"], "train_metrics.json"), "w") as handle:
        json.dump(metrics, handle, indent=2, default=str)
    with open(os.path.join(config["out"], "run_config.json"), "w") as handle:
        json.dump({"config": config, "fingerprints": fingerprints()}, handle, indent=2)
    return metrics


def probe_pass_rates(config: dict, samples: int = 4) -> dict:
    """Per-scenario pass rate, then the 10-90% band RL needs.

    `samples` is low on purpose — this is a filter, not the headline evaluation. Note the
    RESOLUTION that implies: with 4 samples the only possible rates are 0, .25, .50, .75 and
    1.0, so the band keeps the two interior values. Written down because "kept 10-90%" reads
    as finer than 4 samples can express.
    """
    from agents.loader import build_policy
    from veriloop.env.db import open_world
    from veriloop.rewards.terms import compute_terms

    scenarios = load_scenarios(config["data"], config.get("limit"))
    policy = build_policy(12, model_id=config["model"], adapter=config["out"],
                          load_in_4bit=config["load_in_4bit"], dtype=config["dtype"],
                          seed=config["seed"])

    kept, dropped = [], 0
    for scenario in scenarios:
        contract = contract_of(scenario)
        pre = dict(contract.pre_state)
        passes = 0
        for _ in range(samples):
            world = open_world(pre["seed"], pre, episode_id=f"probe-{contract.id}")
            try:
                result = run_episode(world, policy, contract.brief, contract.step_budget)
                terms = compute_terms(contract, result.log, result.snapshot,
                                      terminated=result.terminated, steps=result.steps)
                if terms.gamma == 1 and terms.M >= 1.0 and terms.F >= 1.0:
                    passes += 1
            finally:
                world.close()
        rate = passes / samples
        if 0.10 <= rate <= 0.90:
            kept.append({**scenario, "sft_pass_rate": rate, "rl_filtered": True})
        else:
            dropped += 1

    out_path = config.get("probe_out") or "data/splits/train_rl_filtered.jsonl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as handle:
        for row in kept:
            handle.write(json.dumps(row, default=str) + "\n")
    print(f"  pass-rate probe: kept {len(kept)}/{len(scenarios)} -> {out_path}")
    return {"samples": samples, "kept": len(kept), "dropped": dropped, "of": len(scenarios),
            "written": out_path, "resolution": f"rates can only be k/{samples}"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Arm A1 — QLoRA SFT cold start.")
    parser.add_argument("--data", default="data/splits/train.jsonl")
    parser.add_argument("--out", default="outputs/A1_sft")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="first N scenarios (smoke test)")
    parser.add_argument("--no-4bit", action="store_true", help="full precision (needs ~14 GB)")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--probe", action="store_true", help="also run the pass-rate filter")
    parser.add_argument("--probe-samples", type=int, default=4)
    args = parser.parse_args(argv)

    config = {
        "data": args.data, "out": args.out, "model": args.model, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size, "grad_accum": args.grad_accum,
        "max_length": args.max_length, "warmup_steps": args.warmup_steps, "seed": args.seed,
        "limit": args.limit, "load_in_4bit": not args.no_4bit, "dtype": args.dtype,
        # configs/base.yaml's LoRA block.
        "lora_r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }
    metrics = train(config)
    if args.probe:
        metrics["probe"] = probe_pass_rates(config, samples=args.probe_samples)
    print(json.dumps(metrics, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
