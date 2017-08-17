from grammaropt.grammar import build_grammar
from grammaropt.grammar import Vectorizer

rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
    po = "("
    pc = ")"
    int = "1" / "2" / "3"
"""

grammar = build_grammar(rules)
v = Vectorizer(grammar)
corpus = [
    "1+2",
]
X = v.transform(corpus)
corpus_ = v.inverse_transform(X)
print(corpus_)
