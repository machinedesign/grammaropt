"""
======================================================
Pipeline
======================================================

Simple example of optimizing a scikit-learn pipeline
"""

import warnings

import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import cross_val_score

import torch

from grammaropt.grammar import build_grammar
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import as_str
from grammaropt.random import RandomWalker
from grammaropt.terminal_types import Int, Float
from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import RnnWalker

warnings.filterwarnings("ignore")

digits = datasets.load_digits()
dataset = digits


def evaluate(code):
    X = dataset.data
    y = dataset.target
    clf = _build_estimator(code)
    try:
        scores = cross_val_score(clf, X, y, cv=5)
    except Exception:
        return 0.0
    else:
        return float(np.mean(scores))


def _build_estimator(code):
    clf = eval(code)
    return clf


def main():
    # grammar is simple scikit-learn pipeline
    rules = """
        pipeline = "make_pipeline" "(" elements "," estimator ")"
        elements = (element "," elements) / element
        element = pca / rf
        pca = "PCA" "(" "n_components" "=" int ")"
        estimator = rf / logistic
        logistic = "LogisticRegression" "(" ")"
        rf = "RandomForestClassifier" "(" "max_depth" "=" int "," "max_features" "=" float ")"
    """
    # build grammar
    types = {"int": Int(1, 10), "float": Float(0.0, 1.0)}
    grammar = build_grammar(rules, types=types)
    rules = extract_rules_from_grammar(grammar)
    tok_to_id = {r: i for i, r in enumerate(rules)}

    # set hyper-parameters and build RNN model
    nb_iter = 100
    vocab_size = len(rules)
    emb_size = 128
    hidden_size = 128
    nb_features = 2
    lr = 1e-4
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
    wl = RnnWalker(grammar=grammar, rnn=rnn)
    out = []
    for it in range(nb_iter):
        # walk to generate a parse tree
        wl.walk()
        # get the obtained terminals from the walker
        # and convert to a python code as str
        code = as_str(wl.terminals)
        # evaluate the code with python interpreter
        # and return the validation accuracy, which is the
        # reward
        R = evaluate(code)
        R_avg = R_avg * gamma + R * (1 - gamma)
        # update the model
        model.zero_grad()
        # loss is policy gradient : generate using the
        # model, observe reward R, then maximize the probability
        # of the generated decisions propoertionally to the reward.
        # `R_avg is the policy gradient baseline to make the gradient
        # have less variance.
        loss = (R - R_avg) * wl.compute_loss()
        loss.backward()
        optim.step()
        R_max = max(R, R_max)
        acc_rnn.append(R_max)
        out.append({"code": code, "R": R})
        print(code, R)
    df = pd.DataFrame(out)
    df = df.sort_values(by="R", ascending=False)
    df.to_csv("pipeline.csv")


if __name__ == "__main__":
    main()
