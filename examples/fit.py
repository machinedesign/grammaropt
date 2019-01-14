"""
======================================================
Fit
======================================================

Example of fitting a set of string expressions into
an RNN model.
"""


import sys

import torch

from grammaropt.grammar import build_grammar
from grammaropt.grammar import DeterministicWalker
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import as_str
from grammaropt.terminal_types import Int
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker
from grammaropt.random import RandomWalker


nb_iter = 1000
lr = 1e-4

# rules are a simple symbolic expression
rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = ("(" S ")") / ("sin(" S ")") / ("exp(" S ")") / "x" / int
"""
types = {'int': Int(1, 5)}

# build grammar
grammar = build_grammar(rules, types=types)
rules = extract_rules_from_grammar(grammar)
tok_to_id = {r: i for i, r in enumerate(rules)}

# build model
model = RnnModel(vocab_size=len(rules))
optim = torch.optim.Adam(model.parameters(), lr=lr) 
rnn = RnnAdapter(model, tok_to_id)

# generate uniformly at random an expression from the grammar
wl = RandomWalker(grammar=grammar, min_depth=1, max_depth=5, random_state=42)
wl.walk()
expr = as_str(wl.terminals)

# collect the sequence of decisions (production rules and values) needed to produce 
# the generated expression `expr`
wl = DeterministicWalker(grammar=grammar, expr=expr)
wl.walk()
gt = wl.decisions

for _ in range(1000):
    # fit the RNN model to make it more likely to generate `expr`
    model.zero_grad()
    wl = RnnDeterministicWalker(grammar=grammar, rnn=rnn, decisions=gt)
    wl.walk()
    loss = wl.compute_loss()
    loss.backward()
    optim.step()
    # check if the generation works by generating from the RNN model
    wl = RnnWalker(grammar=grammar, rnn=rnn)
    wl.walk()
    expr_rnn = as_str(wl.terminals)
    print('Loss : {:.5f}, Generated : {}, Groundtruth : {}'.format(loss.item(), expr_rnn, expr))
