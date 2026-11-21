"""One module per workflow family. Each exposes TEMPLATES: dict[name, callable].

A template is a pure function of `(rng, seed, split)` returning a plain dict:

    {
      "domain":      "kyc" | "analytics",
      "is_negative": bool,            # correct outcome is no action / decline / escalate
      "difficulty":  int,             # optional; defaults to len(must_call) + 1
      "terminal":    {"tool": "finish", "args": {...}},
      "contract":    {brief, pre_state, must_call, must_not_call, final_state,
                      step_budget, min_steps},
    }

It must not read the clock, the network or global state — `rng` is the only source of
variation, so `(template, split, seed)` reproduces a scenario exactly.

**The gold route is NOT returned.** `synth.derive_gold` builds it from `must_call` plus
`terminal`. A hand-written route next to a hand-written contract drifts from it, and the
drift is invisible until an arm trains on a scenario whose gold does not satisfy its own
contract.

`is_negative` is what makes the false-block rate measurable. A state check on its own hands
a do-nothing agent credit on every episode where no action was needed, so the negative
fraction is tracked per split and asserted in `tests/test_gen_splits.py`.
"""

from veriloop.gen.templates import analytics, kyc, negatives

#: name -> template callable. Names are global: two modules may not define the same name, or
#: a split would silently generate the wrong family.
TEMPLATES: dict = {**kyc.TEMPLATES, **negatives.TEMPLATES, **analytics.TEMPLATES}

_MODULES = {"kyc": kyc, "negatives": negatives, "analytics": analytics}

_dupes = (len(kyc.TEMPLATES) + len(negatives.TEMPLATES) + len(analytics.TEMPLATES)) - len(TEMPLATES)
if _dupes:
    raise RuntimeError(
        f"{_dupes} duplicate template name(s) across template modules — a split would "
        f"silently generate the wrong family. Names must be globally unique."
    )

__all__ = ["TEMPLATES"]
