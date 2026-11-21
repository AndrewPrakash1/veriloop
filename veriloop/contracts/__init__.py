"""Episode contracts: the objective, expressed as data.

An episode contract is the whole specification of what "done correctly" means for one
scenario. It is a separate artefact from the scenario's sampling parameters, so the
generator can emit thousands of contracts and the reward engine can score any of them
without knowing which template produced it.

This package owns the SHAPE of a contract and its validation. It never touches the
database or the model.
"""

from veriloop.contracts.schema import CallConstraint, Contract, FinalState
from veriloop.contracts.validate import ContractError, validate
from veriloop.contracts.resolve import resolve

__all__ = ["Contract", "CallConstraint", "FinalState", "validate", "ContractError", "resolve"]
