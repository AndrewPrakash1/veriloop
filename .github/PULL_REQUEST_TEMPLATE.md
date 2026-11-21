## What this PR does

<!-- one or two lines. If it takes a paragraph, it is two PRs. -->

**Phase:** <!-- 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 -->
**Owner:** <!-- world | reward | data | gpu-eval | writeup -->
**Branch:** `feature/<owner>/<phase>-<slug>`

## Directories touched

<!-- list them. A PR that touches a directory it does not own will be asked to split. -->

```
veriloop/env/
tests/test_env.py
```

## Evidence

Paste the actual command and its output. "It works" is not evidence and will be sent back.

```
$ make verify-world
...
PHASE 1 GATE: PASS
```

```
$ make test-env
8 passed, 1 skipped
```

## Checklist

- [ ] Every test I wrote (or had written for me) passes, and I did **not** weaken an
      existing assertion to make it pass
- [ ] I added my phase's tests to the `make ci` target in the same commit, so CI covers
      this from now on
- [ ] No contract referenced a tool that does not exist in `veriloop/env/tools.py`
- [ ] If I touched `veriloop/gen/` or `data/`: **no `E3` scenario can reach a training file**
- [ ] If I touched `veriloop/rewards/`: `state_verified` still takes no text-bearing
      argument (`tests/test_imports.py::test_reward_signature_never_sees_model_text`)
- [ ] I did not commit `.venv/`, `data/splits/*.jsonl`, `outputs/`, or anything from
      `audit/*.json`
- [ ] My branch is rebased on current `main`, not merged with it
- [ ] Any number I wrote down is measured, or explicitly labelled **prediction**

## What I could not verify

<!-- Required. Name the thing you are least sure about. "Nothing" is not an answer. -->
