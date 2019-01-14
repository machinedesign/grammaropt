"""
======================================================
Torch model
======================================================

Example of using a Torch RNN model along with Vectorize to
fit a set of string expressions.
"""

import numpy as np
from itertools import product

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from grammaropt.grammar import build_grammar
from grammaropt.grammar import Vectorizer
from grammaropt.grammar import as_str
from grammaropt.grammar import NULL_SYMBOL
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnWalker

def acc(pred, true_classes):
    _, pred_classes = pred.max(1)
    acc = (pred_classes == true_classes).float().mean()
    return acc


# Grammar and corpus
rules = r"""
    S = (T "+" S) / (T "*" S) / (T "/" S) / T
    T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
    po = "("
    pc = ")"
    int = "0" / "1" / "2" / "3" / "4" / "5" / "6" / "7" / "8" / "9"
"""
grammar = build_grammar(rules)
corpus = [
    'x*{}+{}'.format(i, j)
    for i, j in product(range(10), range(10))
]
vect = Vectorizer(grammar, pad=True)
X = vect.transform(corpus)
X = [[0] + x for x in X]
X = np.array(X).astype('int32')

# Model
max_length = max(map(len, X))
vocab_size = len(vect.tok_to_id)
emb_size = 32
batch_size = 32
hidden_size = 32
epochs = 1000
model = RnnModel(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size)
optim = Adam(model.parameters(), lr=1e-3)
adp = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
wl = RnnWalker(grammar, adp, temperature=1.0, min_depth=1, max_depth=5)

# Training
I = X[:, 0:-1]
O = X[:, 1:]
crit = nn.CrossEntropyLoss()
avg_loss = 0.
avg_precision = 0.
for i in range(epochs):
    for j in range(0, len(I), batch_size):
        inp = I[j:j+batch_size]
        out = O[j:j+batch_size]
        out = out.flatten()
        inp = torch.from_numpy(inp).long()
        inp = Variable(inp)
        out = torch.from_numpy(out).long()
        out = Variable(out)
        
        model.zero_grad()
        y = model(inp)
        loss = crit(y, out)
        precision = acc(y, out)
        loss.backward()
        optim.step()

        avg_loss = avg_loss * 0.9 + loss.item() * 0.1
        avg_precision = avg_precision * 0.9 + precision.item() * 0.1
        if i % 10 == 0:
            print('Epoch : {:05d} Avg loss : {:.6f} Avg Precision : {:.6f}'.format(i, avg_loss, avg_precision))
            print('Generated :')
            wl.walk()
            expr = as_str(wl.terminals)
            print(expr)
