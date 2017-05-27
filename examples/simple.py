from grammaropt.grammar import build_grammar
from grammaropt.grammar import StringWalker
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.types import Int
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnWalkerWithGroundtruth
from grammaropt.random import RandomWalker

rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = ("(" S ")") / ("sin(" S ")") / ("exp(" S ")") /  "x" / int
"""
types = {'int': Int(1, 5)}
grammar = build_grammar(rules, types=types)
rules = extract_rules_from_grammar(grammar)
tok_to_id = {r: i for i, r in enumerate(rules)}

wl = StringWalker('(x+1)*(x+2)')
wl.walk(grammar)
gt = wl.decisions

model = RnnModel(vocab_size=len(rules))
rnn = RnnAdapter(model, tok_to_id)

wl = RnnWalkerWithGroundtruth(rnn)
wl.groundtruth_(gt)
wl.walk(grammar)
print(wl.terminals)
