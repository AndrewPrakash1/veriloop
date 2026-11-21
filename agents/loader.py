"""Loading the model — the one place a 4-bit / dtype / adapter decision lives.

Split out of `agents/policy.py` on purpose. The policy stays a pure "render → generate →
parse" object that can be built in a test with a stub model; everything that touches the Hub,
the disk, CUDA or bitsandbytes is here. When a run dies during setup, this is the file to read.

Settings follow `configs/base.yaml`: `CohereLabs/tiny-aya-global`, 4-bit NF4, bf16, and the
rollout's temperature. ONE DELIBERATE DEVIATION is documented on `MAX_NEW_TOKENS`.

torch and transformers are imported INSIDE the functions, so importing this module on a
machine with neither installed still works — which is what keeps `make ci` running with no GPU.
"""

from __future__ import annotations

from typing import Any

DEFAULT_MODEL_ID = "CohereLabs/tiny-aya-global"

#: `configs/base.yaml` sets `max_completion_tokens: 512`. This is deliberately lower, and it is
#: the single biggest cost lever in the project. A tool call is a short JSON object; the
#: policy also stops generation the moment that object closes, so the real cap almost never
#: binds. At ~8 turns per episode across thousands of episodes, 512 vs 128 is the difference
#: between a 2-hour arm and a 5-hour arm. Raise it only for a workflow whose arguments are
#: genuinely long (a `run_sql` query), and say why in the run config.
MAX_NEW_TOKENS = 128

DEFAULT_TEMPERATURE = 0.8


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID) -> Any:
    """Tokenizer only — cheap, no GPU, usable for a prompt dry-run."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        # Tiny Aya ships no pad token. Without one, batched generation warns and can pad with
        # garbage; with a single-sequence rollout it only breaks the attention mask.
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    adapter: str | None = None,
    load_in_4bit: bool = True,
    dtype: str = "bf16",
) -> Any:
    """Load the base model, optionally with a LoRA adapter merged in for evaluation.

    `load_in_4bit` uses bitsandbytes NF4, which is a CUDA-only path: it raises on CPU rather
    than silently loading in fp32 and eating 13 GB of RAM.
    """
    import torch
    from transformers import AutoModelForCausalLM

    torch_dtype = getattr(torch, dtype, None)
    if torch_dtype is None:
        raise ValueError(f"unknown dtype {dtype!r}; expected bf16, fp16 or float32")

    kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "load_in_4bit=True needs CUDA: bitsandbytes' NF4 path has no CPU kernel. "
                "Pass load_in_4bit=False to run a slow bf16 model on CPU, or use the GPU pod."
            )
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
        )
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if adapter:
        from peft import PeftModel

        # NOTE: merging on top of a 4-bit base does NOT raise — PEFT dequantizes, adds the
        # delta in fp32 and re-quantizes, adding rounding error on top of what is already
        # there. That is acceptable for an EVAL adapter; for an exported artifact the merge
        # must happen on an unquantized base (see scripts/export_gguf.py).
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model


def build_policy(
    max_steps: int = 12,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    adapter: str | None = None,
    load_in_4bit: bool = True,
    dtype: str = "bf16",
    temperature: float = DEFAULT_TEMPERATURE,
    max_new_tokens: int = MAX_NEW_TOKENS,
    seed: int | None = None,
    model: Any = None,
    tokenizer: Any = None,
) -> Any:
    """Build the model-backed policy. Pass `model`/`tokenizer` to reuse a loaded pair.

    Returns a `ModelPolicy`, whose only contract is `next_call(observation) -> {"tool","args"}`.
    """
    from agents.policy import ModelPolicy

    if tokenizer is None:
        tokenizer = load_tokenizer(model_id)
    if model is None:
        model = load_model(model_id, adapter=adapter, load_in_4bit=load_in_4bit, dtype=dtype)

    return ModelPolicy(
        model,
        tokenizer,
        max_steps=max_steps,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        seed=seed,
    )
