# Episode contracts — DATA ONLY

> **This directory holds `.yaml`, never `.py`.** The lookalike path `veriloop/contracts/`
> is the Python package that validates these files. A `.py` file here is invisible to the
> test suite and to every import, because nothing puts this directory on `sys.path`.
> The stubs you are meant to fill in already exist at `veriloop/contracts/{schema,validate,resolve}.py`.

One YAML per contract shape, as data. The generator emits thousands from templates; these
two are hand-written so the schema has a reference that a human can read and argue with.

`EXAMPLE_kyc_sanctions_hit.yaml` is the reference scenario for the reward hackability
audit — every attack transcript in `../audit/attacks.yaml` is scored against it, because a
comparison only means something on a fixed scenario.

Two contracts are provided because they bracket the interesting case: one where the
compliant route ends in `escalate`, and one where escalating is itself the violation
(a false block on a clean applicant).
