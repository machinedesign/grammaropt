from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.random import RandomWalker
from grammaropt.random import optimize
from grammaropt.terminal_types import Int

arith = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
    po = "("
    pc = ")"
"""


def test_random_walker():
    types = {"int": Int(1, 10)}
    grammar = build_grammar(arith, types=types)
    for min_depth in range(1, 10):
        wl = RandomWalker(grammar, min_depth=min_depth, max_depth=10, random_state=42)
        wl.walk()
        expr = "".join([str(t) for t in wl.terminals])
        node = grammar.parse(expr)
        depth = _get_max_depth(node)
        assert depth >= min_depth


def test_optimize():
    types = {"int": Int(1, 10)}
    grammar = build_grammar(arith, types=types)
    wl = RandomWalker(grammar, min_depth=1, max_depth=10, random_state=42)
    codes, scores = optimize(_func, wl, nb_iter=10)
    assert len(codes) == 10
    assert len(scores) == 10
    for c, s in zip(codes, scores):
        assert _func(c) == s


def _func(x):
    return 1 if "cos" in x else 0


def _get_max_depth(node):
    if len(node.children) == 0:
        return 0
    return max(1 + _get_max_depth(c) for c in node.children)
