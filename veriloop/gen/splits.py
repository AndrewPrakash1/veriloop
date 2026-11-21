"""Split definitions and the freeze step.

Freezing means: hashes, counts, dedupe and a manifest written at build time. Leakage
between train and eval turns every downstream number into fiction, so the manifest is
committed to git even though the splits themselves are gitignored.
"""

from __future__ import annotations

from typing import Mapping

from veriloop.gen.synth import Scenario

#: E1 in-distribution held-out · E2 novel templates · E3 adversarial traps · E4 paraphrase.
#: E3 IS NEVER TRAINED ON. The moment it enters a rollout batch it stops being a probe.
SPLITS: Mapping[str, str] = {
    "E1": "held-out data, templates already seen in training",
    "E2": "novel templates, never seen in training",
    "E3": "adversarial traps: injected instruction, tempting shortcut, unsolvable case",
    "E4": "paraphrase variants of known templates",
}


def write_splits(scenarios: list[Scenario], out_dir: str) -> dict:
    """Write splits, `manifest.json` and `checksums.txt`.

    Must enforce:
      * no `scenario_id` in two splits
      * no near-duplicate (n-gram overlap + hash) across splits
      * no `E3` scenario in any training file
      * counts, seeds, git SHA and per-file sha256 in the manifest
    """
    raise NotImplementedError("Phase 3")
