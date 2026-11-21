"""`$case.*` substitution and the `when` predicate language.

One implementation of each, shared by the generator, the world and the reward. A second,
subtly different resolver is how a contract and its gold route silently disagree.

TWO DESIGN DECISIONS WORTH KNOWING BEFORE YOU EDIT THIS FILE

**1. `resolve` is loud, not tolerant.** An unresolvable reference raises, and a string that
*contains* `$case.` without being exactly one reference raises too. The tempting
alternative is to pass it through unchanged. Don't: a literal `"$case.missing"` reaching
the reward engine does not look broken — it looks like a *wrong argument*, so you would
spend an afternoon debugging the reward when the bug is in a contract.

**2. `eval_predicate` parses to an AST and walks a whitelist. It never calls `eval()`.**

This is the most important line in the file. Contracts are data, generated in thousands,
and in this project the data format decides whether a compliance step is mandatory. A data
format that executes is an injection hole — and the injected payload would be authored by
the very models under study.

The grammar is deliberately not Turing-complete:

    names           sanctions_hit, all_checks_recorded, pep_hit
    literals        True, False, numbers, 'strings'
    boolean         and  or  not
    comparison      ==  !=  <  <=  >  >=
    membership      in  not in
    grouping        ( )
    whole-collection  all(x, y, ...)   any(x, y, ...)   len(x)
    collection literal  [a, b]   (a, b)

Everything else — attribute access, indexing, comprehensions, lambdas, f-strings, imports,
arbitrary calls — is rejected at parse time. `validate()` calls `compile_predicate`, so an
unparseable predicate is caught at GENERATION time rather than when a rollout batch is
already running.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Mapping

#: A lone `$case.<key>` reference and nothing else. Any string *containing* `$case.` that
#: does not match this raises — see decision 1 above.
_REF = re.compile(r"^\$case\.([A-Za-z_][A-Za-z0-9_]*)$")

#: The only callables the language has. `all`/`any` take collection semantics literally:
#: `all(x, y)` is `bool(x) and bool(y)`.
_ALLOWED_CALLS = frozenset({"all", "any", "len"})

#: Node types permitted in a compiled predicate. Every node in the tree must be in this
#: set, which makes the language safe by construction rather than by denylist.
_ALLOWED_NODES = frozenset({
    ast.Expression, ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Name, ast.Load, ast.Constant,
    ast.Call,
    ast.List, ast.Tuple,
})


class PredicateError(ValueError):
    """Raised when a predicate uses syntax outside the language."""


# ---------------------------------------------------------------------------- resolve ----
def resolve(value: Any, pre_state: Mapping[str, Any]) -> Any:
    """Recursively substitute `"$case.<key>"` strings from `pre_state`.

    Walks dicts, lists and tuples. Anything that is not a string is returned unchanged, so
    numbers and booleans keep their types through the round trip.

    Raises:
        KeyError:   the reference is well-formed but `pre_state` has no such key.
        ValueError: the string contains `$case.` but is not a single reference.
                    Interpolation is not supported — one `$case.<key>` per argument.
    """
    if isinstance(value, str):
        match = _REF.match(value)
        if match:
            key = match.group(1)
            if key not in pre_state:
                raise KeyError(
                    f"contract references $case.{key}, which the scenario's pre_state does "
                    f"not define (it has: {sorted(pre_state)})"
                )
            return pre_state[key]
        if "$case." in value:
            raise ValueError(
                f"string {value!r} contains '$case.' but is not a single reference. "
                f"Interpolation is not supported; use one $case.<key> per argument."
            )
        return value

    if isinstance(value, Mapping):
        return {k: resolve(v, pre_state) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [resolve(v, pre_state) for v in value]
    return value


# ------------------------------------------------------------------ predicate compile ----
def compile_predicate(expr: str) -> ast.Expression:
    """Parse `expr` and reject anything outside the language. Returns the checked AST.

    Separate from `eval_predicate` so contract validation can check that a predicate is
    *well-formed* at generation time, with no world and no facts available.
    """
    if not isinstance(expr, str) or not expr.strip():
        raise PredicateError("predicate must be a non-empty string")

    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise PredicateError(f"predicate does not parse: {expr!r} ({exc.msg})") from exc

    for node in ast.walk(tree):
        if type(node) not in _ALLOWED_NODES:
            raise PredicateError(
                f"predicate uses {type(node).__name__}, which this language does not have: "
                f"{expr!r}. Allowed: and/or/not, == != < <= > >=, in, all/any/len, () []."
            )
        if isinstance(node, ast.Call):
            name = getattr(node.func, "id", None)
            if name not in _ALLOWED_CALLS:
                raise PredicateError(
                    f"predicate calls {name!r}, which is not one of {sorted(_ALLOWED_CALLS)}: "
                    f"{expr!r}"
                )
            if node.keywords:
                raise PredicateError(f"predicate passes keyword arguments: {expr!r}")
    return tree


def _compare(op: ast.cmpop, left: Any, right: Any) -> bool:
    if isinstance(op, ast.Eq):
        return left == right
    if isinstance(op, ast.NotEq):
        return left != right
    if isinstance(op, ast.Lt):
        return left < right
    if isinstance(op, ast.LtE):
        return left <= right
    if isinstance(op, ast.Gt):
        return left > right
    if isinstance(op, ast.GtE):
        return left >= right
    if isinstance(op, ast.In):
        return left in right
    if isinstance(op, ast.NotIn):
        return left not in right
    raise PredicateError(f"unsupported comparison {type(op).__name__}")


def _evaluate(node: ast.AST, facts: Mapping[str, Any]) -> Any:
    if isinstance(node, ast.Expression):
        return _evaluate(node.body, facts)

    if isinstance(node, ast.Constant):
        return node.value

    if isinstance(node, ast.Name):
        if node.id not in facts:
            raise KeyError(
                f"predicate references {node.id!r}, which the world does not expose "
                f"(it has: {sorted(facts)})"
            )
        return facts[node.id]

    if isinstance(node, ast.BoolOp):
        values = [_evaluate(v, facts) for v in node.values]
        return all(values) if isinstance(node.op, ast.And) else any(values)

    if isinstance(node, ast.UnaryOp):
        operand = _evaluate(node.operand, facts)
        return (not operand) if isinstance(node.op, ast.Not) else -operand

    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, facts)
        for op, comparator in zip(node.ops, node.comparators):
            right = _evaluate(comparator, facts)
            if not _compare(op, left, right):
                return False
            left = right
        return True

    if isinstance(node, ast.Call):
        args = [_evaluate(a, facts) for a in node.args]
        # compile_predicate guarantees a bare name from _ALLOWED_CALLS; the isinstance
        # check is for the type checker, not for safety.
        if not isinstance(node.func, ast.Name):
            raise PredicateError("call target must be a bare name")
        name = node.func.id
        if name == "all":
            return all(args)
        if name == "any":
            return any(args)
        if name == "len":
            if len(args) != 1:
                raise PredicateError("len() takes exactly one argument")
            return len(args[0])

    if isinstance(node, (ast.List, ast.Tuple)):
        return [_evaluate(e, facts) for e in node.elts]

    raise PredicateError(f"cannot evaluate {type(node).__name__}")   # unreachable by design


def eval_predicate(expr: str, world_facts: Mapping[str, Any]) -> bool:
    """Evaluate a `when` predicate over facts the world exposes.

    Returns a bool. Raises `PredicateError` for syntax outside the language and `KeyError`
    for a fact the world does not publish. It never sees the model's text — the facts come
    from the database and the audit log.
    """
    return bool(_evaluate(compile_predicate(expr), world_facts))
