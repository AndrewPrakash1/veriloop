"""VeriLoop — state-verified rewards and self-healing post-training for agents.

The library is deliberately import-light: nothing in `veriloop/` needs a GPU, a model
download or a network call. Torch only enters through `agents/` and `train/`.

Import direction is enforced by tests/test_imports.py — the world must not know how it
is scored.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
