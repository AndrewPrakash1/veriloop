# Episode contracts

One YAML per contract shape, as data. The generator emits thousands from templates; these
two are hand-written so the schema has a reference that a human can read and argue with.

`EXAMPLE_kyc_sanctions_hit.yaml` is the reference scenario for the reward hackability
audit — every attack transcript in `../audit/attacks.yaml` is scored against it, because a
comparison only means something on a fixed scenario.

Two contracts are provided because they bracket the interesting case: one where the
compliant route ends in `escalate`, and one where escalating is itself the violation
(a false block on a clean applicant).
