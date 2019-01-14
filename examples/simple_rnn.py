"""
======================================================
Simple RNN
======================================================

Simple example of using RNNs.
"""

import pandas as pd
import numpy as np

import torch

from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker
from grammaropt.terminal_types import Int, Float
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker


X = np.random.uniform(-3, 3, size=(1000,))
y = X ** 2


def evaluate(code):
    from numpy import exp, cos, sin

    x = X
    y_pred = eval(code)
    score = (np.abs(y_pred - y) <= 0.1).mean()
    score = float(score)
    return score


rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / "x" / int
    po = "("
    pc = ")"
"""

types = {"int": Int(1, 10)}
grammar = build_grammar(rules, types=types)

rules = extract_rules_from_grammar(grammar)
tok_to_id = {r: i for i, r in enumerate(rules)}

# set hyper-parameters and build RNN model
nb_iter = 1000
vocab_size = len(rules)
emb_size = 128
hidden_size = 256
nb_features = 2
lr = 1e-3
gamma = 0.9

model = RnnModel(
    vocab_size=vocab_size,
    emb_size=emb_size,
    hidden_size=hidden_size,
    nb_features=nb_features,
)

optim = torch.optim.Adam(model.parameters(), lr=lr)
rnn = RnnAdapter(model, tok_to_id)

# optimization loop
acc_rnn = []
R_avg, R_max = 0.0, 0.0
wl = RnnWalker(
    grammar=grammar, rnn=rnn, min_depth=1, max_depth=10, strict_depth_limit=False
)
out = []
for it in range(nb_iter):
    wl.walk()
    code = as_str(wl.terminals)
    R = evaluate(code)
    R_avg = R_avg * gamma + R * (1 - gamma)
    model.zero_grad()
    loss = (R - R_avg) * wl.compute_loss()
    loss.backward()
    optim.step()
    R_max = max(R, R_max)
    acc_rnn.append(R_max)
    out.append({"code": code, "R": R})
    print(code, R, R_max)
