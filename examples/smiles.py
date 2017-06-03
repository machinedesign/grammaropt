import numpy as np
import pandas as pd

from molecules.molecule import is_valid

import torch

from grammaropt.grammar import build_grammar
from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker
from grammaropt.grammar import DeterministicWalker



rules = r"""
    smiles  =  chain
    atom  =  bracket_atom / aliphatic_organic / aromatic_organic
    aliphatic_organic  = "Cl" / "Br" / "B" / "N" / "O" / "S" / "P" / "F" / "I" / "C"
    aromatic_organic  =  "c" / "n" / "o" / "s"
    bracket_atom  =  "[" BAI "]"
    BAI  =  (isotope symbol BAC) / (isotope symbol) / (symbol BAC) / symbol
    BAC  =  (chiral BAH) / BAH / chiral
    BAH  =  (hcount BACH) / BACH / hcount
    BACH  =  (charge class) / charge / class
    symbol  =  aliphatic_organic / aromatic_organic
    isotope  =  (DIGIT DIGIT DIGIT) / (DIGIT DIGIT) / DIGIT
    DIGIT  =  "1" / "2" / "3" / "4" / "5" / "6" / "7" / "8"
    chiral  =  "@@" / "@"
    hcount  =  ("H" DIGIT) / "H"
    charge = ("-" DIGIT DIGIT) / ("+" DIGIT DIGIT) / ("-" DIGIT) / ("+" DIGIT) / "-" / "+"
    bond  =  "-" / "=" / "#" / "/" / "\\"
    ringbond  =  (bond DIGIT) / DIGIT
    branched_atom  = (atom RB BB) / (atom RB) / (atom BB) / atom
    RB  =  (ringbond RB) / ringbond
    BB  =  (branch BB) / branch
    branch  =  ( "(" bond chain ")" ) / ("(" "." chain ")" ) / ("(" chain ")" )
    chain = (bond branched_atom chain) / (branched_atom chain) / (branched_atom "." chain) / (bond branched_atom) / branched_atom
    class = ":" digits
    digits = (DIGIT digits) / DIGIT
"""

# build grammar
grammar = build_grammar(rules)
rules = extract_rules_from_grammar(grammar)
tok_to_id = {r: i for i, r in enumerate(rules)}
# build model
lr = 1e-3
gamma = 0.9
model = RnnModel(vocab_size=len(rules))
optim = torch.optim.Adam(model.parameters(), lr=lr) 
rnn = RnnAdapter(model, tok_to_id)

smiles = np.load('zinc_250k_subset.npz')['X']
avg_loss = 0.
smiles = smiles[0:100]
for epoch in range(10000):
    np.random.shuffle(smiles)
    for s in smiles:
        wl = DeterministicWalker(grammar=grammar, expr=s)
        wl.walk()
        gt = wl.decisions
        model.zero_grad()
        wl = RnnDeterministicWalker(grammar=grammar, rnn=rnn, decisions=gt)
        wl.walk()
        loss = wl.compute_loss()
        loss.backward()
        optim.step()
        avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
        print('avg loss : {:.4f}, loss : {:4f}'.format(avg_loss, loss.data[0]))
    if epoch % 10 == 0:
        # check if the generation works by generating from the RNN model
        wl = RnnWalker(grammar=grammar, rnn=rnn, min_depth=1, max_depth=10, strict_depth_limit=True)
        wl.walk()
        expr = as_str(wl.terminals)
        print(expr, is_valid(expr))
