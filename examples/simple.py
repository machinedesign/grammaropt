import sys

from grammaropt.grammar import build_grammar
from grammaropt.grammar import as_str

from grammaropt.types import Int
from grammaropt.random import RandomWalker


rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
    po = "("
    pc = ")"
"""

types = {'int': Int(1, 10)}
grammar = build_grammar(rules, types=types)

wl = RandomWalker(grammar=grammar, min_depth=1, max_depth=10)

for _ in range(10):
    wl.walk()
    expr = as_str(wl.terminals)
    print(expr)
