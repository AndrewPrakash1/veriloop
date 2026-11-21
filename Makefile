PY ?= .venv/bin/python    # 3.12+ required; see STRUCTURE.md §5
ARM ?= state

.PHONY: help smoke test ci verify-world test-env test-contracts test-reward test-gen audit splits baseline sft rl heal report clean

help:
	@grep -E '^[a-z-]+:' Makefile | sed 's/:.*//' | column -c 80

smoke:            ## phase 0: prove both kits import and run
	@$(PY) scripts/smoke.py

ci:               ## THE GATE. Grows one phase at a time as each lands.
	@$(PY) -m pytest tests/test_imports.py -p no:cacheprovider
	@$(PY) scripts/verify_world.py
	@$(PY) -m pytest tests/test_env.py -p no:cacheprovider
	@$(PY) -m pytest tests/test_resolve.py tests/test_contracts.py -m phase2 -p no:cacheprovider
	@$(PY) -m pytest tests/test_reward_terms.py tests/test_reward_hackability.py -m phase2 -p no:cacheprovider
	@$(PY) scripts/run_audit.py --out /tmp/veriloop-audit.json
	@$(PY) -m pytest tests/test_gen_splits.py -m phase3 -p no:cacheprovider
	@$(PY) scripts/make_splits.py --out /tmp/veriloop-splits-check
test-imports:     ## structural tests — always green, catch the design bugs
	@$(PY) -m pytest tests/test_imports.py -v -p no:cacheprovider

test:             ## the WHOLE spec, including phases not built yet (RED)
	@$(PY) -m pytest

verify-world:     ## phase 1 gate: the gold route satisfies its own contract
	@$(PY) scripts/verify_world.py

test-env:         ## phase 1
	@$(PY) -m pytest tests/test_env.py -m phase1

test-contracts:   ## phase 2
	@$(PY) -m pytest tests/test_contracts.py -m phase2

test-reward:      ## phase 2 — the one that matters
	@$(PY) -m pytest tests/test_reward_terms.py tests/test_reward_hackability.py -m phase2

test-gen:         ## phase 3
	@$(PY) -m pytest tests/test_gen_splits.py -m phase3

audit:            ## the pre-GPU evidence: score every attack under both rewards
	@$(PY) scripts/run_audit.py --out audit/reward_hackability.json

splits:           ## generate + freeze the scenario splits
	@$(PY) scripts/make_splits.py --out data

baseline:         ## arm A0 over E1-E4 on a served model
	$(PY) train/eval_arms.py --arm A0 --splits E1 E2 E3 E4

sft:              ## arm A1
	$(PY) train/sft.py --config configs/sft.yaml

rl:               ## one RL arm; ARM in {claim, state, state_shaped, random}
	$(PY) train/rl.py --config configs/grpo_$(ARM).yaml

heal:             ## one self-healing round, gated
	$(PY) -m veriloop.heal.loop --rounds 1

report:           ## rebuild the before/after table from outputs/
	$(PY) artifacts/build_report.py

clean:
	rm -rf .pytest_cache **/__pycache__ outputs/* data/splits/*.jsonl audit/*.json
