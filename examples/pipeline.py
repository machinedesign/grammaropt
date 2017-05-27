import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show, output_file

import torch

from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.random import RandomWalker
from grammaropt.types import Int
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker

import warnings

warnings.filterwarnings("ignore")

iris = datasets.load_iris()
digits = datasets.load_digits()
dataset = digits

def evaluate(code):
    X = dataset.data
    y = dataset.target
    clf = _build_estimator(code)
    try:
        scores = cross_val_score(clf, X, y, cv=5)
    except Exception:
        return 0
    else:
        return np.mean(scores)


def _build_estimator(code):
    clf = eval(code)
    return clf


def main():
    rules = """
        pipeline = "make_pipeline" "(" elements "," estimator ")"
        elements = (element "," elements) / element
        element = pca / rf
        pca = "PCA" "(" "n_components" "=" small ")"
        estimator = rf / logistic
        logistic = "LogisticRegression" "(" ")"
        rf = "RandomForestClassifier" "(" "max_depth" "=" int ")"
    """
    types = {'int': Int(1, 10), 'small': Int(1, 10)}
    grammar = build_grammar(rules, types=types)
    rules = extract_rules_from_grammar(grammar)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    
    nb_iter = 1000
    vocab_size = len(rules)
    emb_size = 128
    hidden_size = 128
    nb_features = 1
    lr = 1e-5
    gamma = 0.9

    model = RnnModel(
        vocab_size=vocab_size, 
        emb_size=emb_size, 
        hidden_size=hidden_size, 
        nb_features=nb_features)
    optim = torch.optim.Adam(model.parameters(), lr=lr) 
    rnn = RnnAdapter(model, tok_to_id)

    acc_rnn = []
    R_avg = 0.
    R_max = 0.
    wl = RnnWalker(rnn)
    for it in range(nb_iter):
        wl.walk(grammar)
        out = wl.terminals
        code = ''.join(map(str, out))
        R = evaluate(code)
        R = float(R)
        R_avg = R_avg * gamma + R * (1 - gamma)
        model.zero_grad()
        loss = (R - R_avg) * wl.compute_loss()
        loss.backward()
        optim.step()
        R_max = max(R, R_max)
        acc_rnn.append(R_max)
        print(code, R)

    acc_random = []
    R_avg = 0.
    R_max = 0.
    wl = RandomWalker()
    for it in range(nb_iter):
        wl.walk(grammar)
        out = wl.terminals
        code = ''.join(map(str, out))
        R = evaluate(code)
        R = float(R)
        R_avg = R_avg * gamma + R * (1 - gamma)
        R_max = max(R, R_max)
        acc_random.append(R_max)
        print(code, R)

    TOOLS = 'pan,wheel_zoom,box_zoom,reset,save,box_select'
    p = figure(title='avg reward', tools=TOOLS)
    output_file("plot.html", title="legend.py example")
    p.line(np.arange(len(acc_rnn)), acc_rnn, legend='rnn', color='blue', line_width=2)
    p.line(np.arange(len(acc_random)), acc_random, legend='random', color='red', line_width=2)
    show(p)


if __name__ == '__main__':
    main()
