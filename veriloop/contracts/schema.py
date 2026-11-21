"""The contract model.

Shape (mirrors the worked examples in `contracts/*.yaml`):

    id, domain, brief, pre_state
    must_call      [{tool, args, when}]     mandatory actions, with arg constraints
    must_not_call  [{tool, when}]           forbidden in this state; violation => Gamma = 0
    final_state    {table: {assertions}}    what the world must look like at the end
    step_budget, min_steps, requires_terminal

Field notes that are load-bearing:

* `must_call.when` is a predicate over the WORLD, evaluated after the episode. It is how
  a contract says "escalate *if* sanctions hit" without making escalation unconditional.
  An unsatisfied `when` removes the constraint rather than failing it.
* `must_not_call.when` is evaluated at the call's step index, which is why the audit log
  stores a step number on every record.
* `final_state` assertions are declarative comparisons, not Python. Anything that cannot
  be expressed here is a sign the workflow needs a new column, not an `eval`.
* An EMPTY `must_call` is legal and meaningful: it is the negative-control case where the
  correct behaviour is to do nothing. `M` is then vacuously 1 and `F` checks that the
  world is unchanged. The generator must emit these deliberately.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field


class CallConstraint(BaseModel):
    tool: str
    args: Mapping[str, Any] = Field(default_factory=dict)   # may contain "$case.*" refs
    when: str | None = None                                 # predicate over the world


class FinalState(BaseModel):
    table: str
    #: e.g. {"customer_id": "$case.customer_id", "status": "escalated"}
    assertions: Mapping[str, Any]
    #: "<=", "==", "in", "absent" — kept declarative on purpose
    op: str = "=="


class Contract(BaseModel):
    id: str
    domain: str
    brief: str
    pre_state: Mapping[str, Any]
    must_call: Sequence[CallConstraint] = ()
    must_not_call: Sequence[CallConstraint] = ()
    final_state: Sequence[FinalState] = ()
    step_budget: int = 12
    min_steps: int = 2
    requires_terminal: bool = True

    #: Set by the generator; `E3` etc. Never used by the reward.
    split: str | None = None
