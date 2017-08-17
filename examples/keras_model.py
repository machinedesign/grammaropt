"""
======================================================
Keras model
======================================================

Example of using a keras RNN model along with Vectorize to
fit a set of string expressions.
"""

from itertools import product
import numpy as np
from keras.layers import LSTM, Input, TimeDistributed, Activation, Embedding, Dense
import keras.backend as K
from keras.models import Model
from keras.optimizers import Adam
from grammaropt.grammar import build_grammar
from grammaropt.grammar import Vectorizer


def precision(y_true, y_pred):
    return (y_true.argmax(axis=-1) == y_pred.argmax(axis=-1)).mean()


def categorical_crossentropy(y_true, y_pred):
    yt = y_true.flatten()
    ypr = y_pred.reshape((y_pred.shape[0]*y_pred.shape[1], y_pred.shape[2]))
    return K.categorical_crossentropy(yt, ypr)


def onehot(X, D=10):
    X = X.astype('int32')
    nb = np.prod(X.shape)
    x = X.flatten()
    m = np.zeros((nb, D))
    m[np.arange(nb), x] = 1.
    m = m.reshape(X.shape + (D,))
    return m.astype('float32')

def generate(model, length, nb=1):
    x = np.zeros((nb, length + 1,)).astype('int32')
    for i in range(length):
        y = model.predict(x[:, 0:length])
        for e in range(nb):
            symbol = np.random.choice(np.arange(y.shape[2]), p=y[e, i])
            x[e, i + 1] = symbol
    return x


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
epochs = 800
inp = Input(shape=(max_length - 1,))
x = Embedding(vocab_size, emb_size)(inp)
x = LSTM(32, return_sequences=True)(x)
x = TimeDistributed(Dense(vocab_size))(x)
out = Activation('softmax')(x)
model = Model(inputs=inp, outputs=out)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3))
I = X[:, 0:-1]
O = X[:, 1:]

# Training
avg_precision = 0.0
avg_loss = 0.0
for i in range(epochs):
    for j in range(0, len(I), batch_size):
        inp = I[j:j+batch_size]
        out = O[j:j+batch_size]
        out = onehot(out, D=vocab_size)
        loss = model.train_on_batch(inp, out)
        p = np.mean(precision(out, model.predict(inp)))
        avg_precision = avg_precision * 0.9 + p * 0.1
        avg_loss = avg_loss * 0.9 + loss * 0.1
        if i % 10 == 0:
            print('Epoch : {:05d} Avg loss : {:.6f} Avg Precision : {:.6f}'.format(i, avg_loss, avg_precision))
    y = generate(model, max_length - 1, nb=1)
    y = [expr[1:] for expr in y]
    try:
        y = vect.inverse_transform(y)
    except Exception:
        # happens because the `generate` function does not take into
        # account the forbidden rules by the grammar. So the exception
        # occurs because of a syntax error.
        continue
    print('Generated:')
    for expr in y:
        print(expr)
