"""
======================================================
Autoencoder model with Torch
======================================================

Example of using a Torch RNN autoencoder model to
learn continuous representations of strings.
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from grammaropt.grammar import build_grammar
from grammaropt.grammar import Vectorizer
from grammaropt.grammar import NULL_SYMBOL
from grammaropt.grammar import as_str
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker

class Model(nn.Module):
    def __init__(self, vocab_size=10, emb_size=128, hidden_size=128, num_layers=1, use_cuda=False):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.use_cuda = use_cuda
        self.X = None
        # convolutional autoencoder
        self.features = nn.Sequential(
            nn.Conv1d(emb_size, 32, 3),
            nn.ReLU(True),
            nn.Conv1d(32, hidden_size, 3),
        )
        self.emb = nn.Embedding(vocab_size, emb_size)
        # LSTM decoder
        self.lstm = nn.LSTM(hidden_size, 128, batch_first=True, num_layers=num_layers)
        self.out_token  = nn.Linear(128, vocab_size)
    
    def forward(self, inp):
        T = inp.size(1)
        x = self.encode(inp)
        x = x.view(x.size(0), 1, -1)
        x = x.repeat(1, T, 1)
        o, _ = self.lstm(x)
        o = o.contiguous()
        o = o.view(o.size(0) * o.size(1), o.size(2))
        o = self.out_token(o)
        return o
    
    def encode(self, inp):
        x = self.emb(inp)
        x = x.transpose(1, 2)
        x = self.features(x)
        x = x.mean(2)
        return x
 
    def given(self, inp):
        x = self.encode(inp)
        self.X = x.view(x.size(0), 1, -1)

    def next_token(self, inp, state): 
        if self.use_cuda:
            inp = inp.cuda()
        _, state = self.lstm(self.X, state)
        h, c = state
        h = h[-1] # last layer
        o = self.out_token(h)
        return o, state
 

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
batch_size = 64
hidden_size = 2
epochs = 1000
model = Model(vocab_size=vocab_size, emb_size=emb_size, hidden_size=hidden_size)
optim = Adam(model.parameters(), lr=1e-3)
adp = RnnAdapter(model, tok_to_id=vect.tok_to_id, begin_tok=NULL_SYMBOL)
wl = RnnWalker(grammar, adp, temperature=1.0, min_depth=1, max_depth=5)

# Training
I = X
O = X
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
            model.given(inp)
            wl.walk()
            expr = as_str(wl.terminals)
            print(expr)
            inp = I
            inp = torch.from_numpy(inp).long()
            inp = Variable(inp)
            h = model.encode(inp)
            h = h.data.numpy()
            fig = plt.figure(figsize=(30, 10))
            plt.scatter(h[:, 0], h[:, 1])
            # from https://stackoverflow.com/questions/5147112/matplotlib-how-to-put-individual-tags-for-a-scatter-plot
            for label, x, y in zip(corpus, h[:, 0], h[:, 1]):
                plt.annotate(
                    label,
                    xy=(x, y), xytext=(-20, 20),
                    textcoords='offset points', ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))
            plt.savefig('latent_space.png')
            plt.close(fig)
