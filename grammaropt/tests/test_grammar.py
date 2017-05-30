import pytest

from parsimonious.exceptions import UndefinedLabel
from parsimonious.expressions import Compound

from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import Walker
from grammaropt.grammar import DeterministicWalker
from grammaropt.grammar import _Decision
from grammaropt.types import Int

arith = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
    po = "("
    pc = ")"
"""

class DummyWalker(Walker):

    def next_rule(self, rules, depth=0):
        for r in rules:
            if not isinstance(r, Compound):
                return r
        return r

    def next_value(self, rule):
        return "1"


def test_build_grammar():
    rules = r"""
        a = (b c) / b / d
        b = "(" b e ")"
        c = ")" c d "("
    """
    with pytest.raises(UndefinedLabel):
        grammar = build_grammar(rules)
    types = {"d": Int(1, 10), "e": Int(1, 10)}
    grammar = build_grammar(rules, types=types)
    assert "e" in grammar
    assert "d" in grammar
    assert type(grammar["e"]) == Int
    assert type(grammar["d"]) == Int



def test_extract_rules_from_grammar():
    types = {"int": Int(1, 10)}
    grammar = build_grammar(arith, types=types)
    rules = extract_rules_from_grammar(grammar)
    assert len(rules) == 19


def test_walker():
    types = {"int": Int(1, 10)}
    grammar = build_grammar(arith, types=types)
    rules = extract_rules_from_grammar(grammar)
    wl = DummyWalker(grammar)
    wl.walk()
    assert wl.terminals == ["x"]
    wl.walk()
    assert wl.terminals == ["x"]


def test_deterministic_walker():
    types = {"int": Int(1, 10)}
    grammar = build_grammar(arith, types=types)
    rules = extract_rules_from_grammar(grammar)
    dec = [
        _Decision(rule=grammar["S"], choice=grammar["T"]),
        _Decision(rule=grammar["T"], choice=grammar["int"]),
        _Decision(rule=grammar["int"], choice=5)
    ]
    expr = "5"
    wl = DeterministicWalker(grammar, expr)
    for _ in range(2):
        wl.walk()
        assert wl.decisions == dec
        assert wl.terminals == [5]
    expr = "x+cos(x+5)"
    wl = DeterministicWalker(grammar, expr)
    for _ in range(2):
        wl.walk()
        assert len(wl.decisions) == 9
        assert wl.terminals == ["x", "+", "cos", "(", "x", "+", 5, ")"]
