"""The only suite that can be GREEN before any phase lands: it asserts structure.

It parses the modules with `ast` rather than importing them, so it works on a tree of
stubs — and it catches the one design bug that is invisible at runtime and expensive
later: a cycle that lets the world know how it is scored.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

#: import direction: module prefix -> prefixes it is allowed to import from veriloop.*
ALLOWED = {
    "veriloop.env":       {"veriloop.env"},   # a package may always import itself
    "veriloop.contracts": {"veriloop.contracts"},
    "veriloop.gen":       {"veriloop.contracts", "veriloop.env", "veriloop.gen"},
    "veriloop.rewards":   {"veriloop.contracts", "veriloop.env", "veriloop.rewards"},
    "veriloop.eval":      {"veriloop.contracts", "veriloop.env", "veriloop.rewards", "veriloop.eval"},
    # heal/ reads the reward terms to classify a failure and reads eval to score a round —
    # both are legitimate: the loop consumes the reward, it does not define it.
    "veriloop.heal":      {"veriloop.contracts", "veriloop.env", "veriloop.gen",
                           "veriloop.eval", "veriloop.rewards", "veriloop.heal"},
    "agents":             {"veriloop.env", "veriloop.contracts", "agents"},
    "train":              {"veriloop", "agents", "train"},
}


def _veriloop_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.startswith("veriloop"):
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for a in node.names:
                if a.name.startswith("veriloop"):
                    found.add(a.name)
    return found


@pytest.mark.parametrize("prefix", sorted(ALLOWED))
def test_no_cycle(prefix: str) -> None:
    """A module in `prefix` may only import from its own allowed set."""
    pkg = ROOT / prefix.replace(".", "/")
    offenders = []
    for py in pkg.rglob("*.py"):
        for mod in _veriloop_imports(py):
            # find the longest allowed prefix that this module starts with
            if not any(mod == a or mod.startswith(a + ".") for a in ALLOWED[prefix]):
                offenders.append(f"{py.relative_to(ROOT)} imports {mod}")
    assert not offenders, "import direction violated:\n  " + "\n  ".join(offenders)


def test_world_does_not_import_rewards() -> None:
    """The world must not know how it is scored."""
    env = ROOT / "veriloop" / "env"
    bad = [str(p.relative_to(ROOT)) for p in env.rglob("*.py")
           if any("rewards" in m for m in _veriloop_imports(p))]
    assert not bad, f"env/ imports rewards/ in: {bad}"


def test_reward_signature_never_sees_model_text() -> None:
    """The proposed reward must not accept the model's prose.

    This is the single most important structural assertion in the project: the moment a
    `final_message` can reach `state_verified`, the reward can read text, and the whole
    failure mode under study becomes unmeasurable.
    """
    import inspect

    from veriloop.rewards.state_verified import state_verified

    params = set(inspect.signature(state_verified).parameters)
    forbidden = {"final_message", "message", "text", "completion", "answer", "response"}
    leaked = params & forbidden
    assert not leaked, f"state_verified accepts text-bearing params: {leaked}"

    from veriloop.rewards.claim_verified import claim_verified
    claim_params = set(inspect.signature(claim_verified).parameters)
    assert claim_params == {"final_message"}, (
        "claim_verified must take ONLY the final message — it is the naive arm, and it "
        f"must not be able to consult state or log (got {claim_params})"
    )
