"""
======================================================
SMILES
======================================================

Example of fitting SMILES, a representation of 
molecular graphs.
"""

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


def _get_max_depth(node):
    if len(node.children) == 0:
        return 0
    return max(1 + _get_max_depth(c) for c in node.children)


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
gamma = 0.99

model = RnnModel(vocab_size=len(rules), num_layers=2)
optim = torch.optim.Adam(model.parameters(), lr=lr)
rnn = RnnAdapter(model, tok_to_id)

smiles = np.load("zinc_250k_subset.npz")["X"]
# for s in smiles:
#    grammar.parse(s)
max_depth = 100
print("Size of training : {}".format(len(smiles)))
# max_depth = max(map(_get_max_depth, map(grammar.parse, smiles)))
nb_updates = 0
avg_loss = 0.0
print("Start training...")
for epoch in range(100):
    np.random.shuffle(smiles)
    for i, s in enumerate(smiles):
        wl = RnnDeterministicWalker.from_str(grammar, rnn, s)
        wl.walk()
        model.zero_grad()
        loss = wl.compute_loss() / len(wl.decisions)
        loss.backward()
        optim.step()
        avg_loss = avg_loss * gamma + loss.data[0] * (1 - gamma)
        print(
            "Example {:06d}/{:06d} avg loss : {:.4f}, loss : {:4f}".format(
                i, len(smiles), avg_loss, loss.data[0]
            )
        )
        if nb_updates % 100 == 0 and nb_updates > 0:
            print("Generating...")
            wl = RnnWalker(
                grammar=grammar,
                rnn=rnn,
                min_depth=1,
                max_depth=max_depth,
                strict_depth_limit=False,
            )
            nb_valid = 0
            nb = 100
            for _ in range(nb):
                # check if the generation works by generating from the RNN model
                wl.walk()
                expr = as_str(wl.terminals)
                print(expr)
                nb_valid += is_valid(expr)
            print("nb valid : {}/{}".format(nb_valid, nb))
            torch.save(model, "model.th")
        nb_updates += 1
