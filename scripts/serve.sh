#!/usr/bin/env bash
# Serve the base model for rollout/eval. Only needed from Phase 4 onward.
# SGLang or vLLM, OpenAI-compatible. Keep the port here so every script agrees on it.
set -euo pipefail
MODEL="${VERILOOP_MODEL:-CohereLabs/tiny-aya-global}"
PORT="${VERILOOP_PORT:-30000}"

echo "serving ${MODEL} on :${PORT}"
# python -m sglang.launch_server --model-path "$MODEL" --port "$PORT"
#   --context-length 8192 --trust-remote-code
