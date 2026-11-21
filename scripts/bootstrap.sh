#!/usr/bin/env bash
#
# VeriLoop pod bootstrap — run this FIRST on any Race Engineering GPU pod.
#
# WHAT IT IS FOR
#
# It answers one question before you spend a rupee on anything else: *can this box actually
# run CUDA kernels?* On an RTX 5090 that is not a given. The 5090 is Blackwell (sm_120) and
# needs CUDA 12.8+ / PyTorch >= 2.7; the pre-built ML stacks advertise PyTorch 2.5.1 on
# CUDA 12.4, which contains no sm_120 kernels. The failure mode is nasty because
# `torch.cuda.is_available()` returns True and the device name prints correctly — only a real
# kernel launch fails, with:
#
#     RuntimeError: no kernel image is available for execution on the device
#
# So this script does not ask whether CUDA is "available". It launches a real matmul.
#
# WHAT IT DOES NOT DO
#
# It does not train anything, and it cannot: `train/sft.py`, `train/rl.py` and
# `scripts/run_baseline.py` are still Phase 4/5/6 stubs that raise NotImplementedError. This
# script gets the ENVIRONMENT right so that when those land, the pod is ready in one command.
#
# UNTESTED ON A 5090 POD. The checks and the install commands are standard, but this has been
# written off-box. Read the output at each step; every install below is guarded so a failure
# is visible rather than silent.
#
#   usage:  bash scripts/bootstrap.sh [--install-agenttune]
#
set -uo pipefail

ok()   { printf '  \033[32mOK\033[0m    %s\n' "$1"; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$1"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; }
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }

INSTALL_AGENTTUNE=0
[ "${1:-}" = "--install-agenttune" ] && INSTALL_AGENTTUNE=1

# ----------------------------------------------------------------- 1. is there a GPU? -----
step "1. GPU present"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    bad "no nvidia-smi on PATH — this is not a GPU pod, or the driver is missing."
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap \
           --format=csv,noheader || true

# ------------------------------------------------------- 2. does torch see the card? ------
step "2. PyTorch build vs this card"
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || { bad "no $PY on PATH"; exit 1; }

TORCH_INFO=$("$PY" - <<'PY' 2>&1
try:
    import torch
except Exception as exc:                       # noqa: BLE001 — report, never crash the script
    print(f"IMPORT_FAILED {type(exc).__name__}: {exc}")
    raise SystemExit(0)
caps = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
arch = torch.cuda.get_arch_list()
want = f"sm_{caps[0]}{caps[1]}"
print(f"VERSION {torch.__version__}")
print(f"CUDA {torch.version.cuda}")
print(f"DEVICE {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}")
print(f"CAPABILITY {want}")
print(f"HAS_KERNELS {'yes' if any(a == want for a in arch) else 'no'}")
PY
)
echo "$TORCH_INFO" | sed 's/^/  /'

NEEDS_TORCH_FIX=0
case "$TORCH_INFO" in
    *IMPORT_FAILED*)  warn "torch is not importable — installing a Blackwell build"; NEEDS_TORCH_FIX=1 ;;
    *"HAS_KERNELS no"*)  warn "this torch has NO kernels for this card (the sm_120 trap)"; NEEDS_TORCH_FIX=1 ;;
esac

if [ "$NEEDS_TORCH_FIX" = "1" ]; then
    # cu128 (CUDA 12.8) is the first wheel set that carries sm_120 kernels. Newer cu13x
    # wheels also work; cu126 and below never will, whatever the torch version says.
    warn "installing a cu128 build. This REPLACES torch — AgentTune pins torch 2.11, and a"
    warn "full AgentTune install would replace it again, so it goes on later with --no-deps."
    "$PY" -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128 \
        || { bad "torch install failed — paste this output rather than guessing further"; exit 1; }
fi

# --------------------------------------------------- 3. THE PROOF: launch a real kernel ----
step "3. Real CUDA kernel (the only test that counts)"
"$PY" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("FAIL: torch.cuda.is_available() is False — no usable CUDA device")
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
c = a @ b
torch.cuda.synchronize()
print(f"  OK    fp16 4096x4096 matmul on {torch.cuda.get_device_name(0)} -> {float(c.sum()):.1f}")
print(f"  OK    bf16 supported: {torch.cuda.is_bf16_supported()}")
PY
if [ $? -ne 0 ]; then
    bad "a real kernel FAILED. If you see 'no kernel image is available', the torch build is"
    bad "wrong for this card — re-run step 2 rather than debugging the model."
    exit 1
fi

# ------------------------------------------------------------------- 4. python version ----
step "4. Python version (AgentTune needs >= 3.12)"
"$PY" -V
"$PY" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 12) else 1)' \
    && ok "3.12+ — AgentTune can be installed here" \
    || warn "below 3.12 — phases 0-4 still work, but AgentTune and train/ cannot import"

# ------------------------------------------------------------------ 5. the two kits ------
step "5. Toolkits"
if [ "$INSTALL_AGENTTUNE" = "1" ]; then
    # --no-deps on purpose: a full install pins torch and replaces the build we just fixed.
    # AgentTune imports torch at package-import time, so torch must already be correct.
    "$PY" -m pip install --no-deps "agenttune @ git+https://github.com/Lexsi-Labs/AgentTune.git" \
        || warn "AgentTune install failed — it is only needed by train/ and agents/harness"
else
    warn "skipped AgentTune (pass --install-agenttune). It is not needed for phases 0-4."
fi
"$PY" -m pip install auditkit >/dev/null 2>&1 \
    && ok "auditkit installed" \
    || warn "auditkit not installed (zero-dep core; only eval/harness needs it)"

# ------------------------------------------------------------------------- 6. next ------
step "6. What to do next"
cat <<'NEXT'
  1. Start tmux BEFORE anything long:      tmux new -s veriloop
  2. Get the repo + the frozen splits onto this box (scp, or git clone + make splits).
  3. Verify the pipeline is the one we tested on CPU:
         python -m pytest tests/test_eval.py -m phase4 -q
  4. ONLY THEN measure generation throughput — the number the whole GPU budget rests on.
     Unbatched speed for this model on a T4 was measured at 8.6 tok/s by a prior run, so
     anything near that means batching is missing, not that the card is slow.
  5. PAUSE the pod when you stop (billing stops, /workspace survives). Do not terminate.

  Reminder: `nvidia-smi` naming the card is NOT proof it works. Step 3 is the proof.
NEXT
