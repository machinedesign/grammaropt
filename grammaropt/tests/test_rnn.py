import math
import pytest

import numpy as np
from scipy.stats import norm
from scipy.stats import poisson
from scipy.stats import beta

from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import optimize
from grammaropt.rnn import _normalize
from grammaropt.rnn import _torch_logp_normal
from grammaropt.rnn import _torch_logp_poisson
from grammaropt.rnn import _torch_logp_beta

from grammaropt.grammar import DeterministicWalker
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import build_grammar
from grammaropt.types import Int
from grammaropt.types import Float

import torch
from torch.autograd import Variable

from grammaropt.tests.test_random import _get_max_depth


def test_model():
    model = RnnModel(vocab_size=10, emb_size=128, hidden_size=128, nb_features=1)
    inp = Variable(torch.zeros(1, 1)).long()
    state = Variable(torch.zeros(1, 1, 128)), Variable(torch.zeros(1, 1, 128))
    pred, _ = model.next_token(inp, state)
    assert pred.size() == (1, 10)
    pred, _ = model.next_value(inp, state)
    assert pred == 0.1504


def test_adapter():
    tok_to_id = {'z': 0, 'a': 1, 'b': 2, 'c': 3}
    
    # check assertion when vocab size should do not correpond to tok_to_id nb of elements
    with pytest.raises(AssertionError):
        model = RnnModel(vocab_size=3)
        rnn = RnnAdapter(model, tok_to_id)
    

    model = RnnModel(vocab_size=4, hidden_size=128)
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    
    # check assertion error when size of pr should correspond to vocab_size
    with pytest.raises(AssertionError):
        pr = Variable(torch.from_numpy(np.array([0.1, 0.1, 0.1])).view(1, -1))
        rnn.generate_next_token(pr)
    # check assertion error when probas sum should > 0
    with pytest.raises(AssertionError):
        pr = Variable(torch.from_numpy(np.array([0., 0., 0., 0.])).view(1, -1))
        rnn.generate_next_token(pr)
    
    # check if allowed has the correct behavior
    pr = Variable(torch.from_numpy(np.array([0.25, 0.25, 0.25, 0.25])).view(1, -1))
    tok = rnn.generate_next_token(pr)
    assert tok in ('a', 'b', 'c', 'z')
    tok = rnn.generate_next_token(pr, allowed=['a', 'b', 'c'])
    assert tok in ('a', 'b', 'c')
    tok = rnn.generate_next_token(pr, allowed=['a', 'b'])
    assert tok in ('a', 'b')
    tok = rnn.generate_next_token(pr, allowed=['a'])
    assert tok =='a'
    with pytest.raises(AssertionError):
        tok  = rnn.generate_next_token(pr, allowed=[])
    
    # check if generate value from Int is in the interval
    tok = Int(1, 10)
    stat = Variable(torch.from_numpy(np.array([5.])).view(1, 1))
    val = rnn.generate_next_value(stat, tok)
    assert 1 <= val <= 10
    
    # check unrecognizable token type because 'a' is not a Type
    with pytest.raises(TypeError):
        val = rnn.generate_next_value(stat, 'a')
    
    # check behavior of logp

    logp = rnn.token_logp('a', pr)
    assert logp.size() == (1,)
    assert logp.data[0] == np.log(pr[0, tok_to_id['a']])

    logp = rnn.value_logp(tok, 5, stat)
    assert logp.size() == (1,)
    
    with pytest.raises(TypeError):
        val = rnn.value_logp('a', 5, stat)
    
    # check if generate value from Float is in the interval
    tok = Float(0., 10.)
    stat = Variable(torch.from_numpy(np.array([5., 1.])).view(1, 2))
    val = rnn.generate_next_value(stat, tok)
    assert 0 <= val <= 10
    
    val = rnn.value_logp(tok, 5., stat)
    assert val.size() == (1,) 
 
    # check if probas returned by predict_next_token sum to 1
    state = Variable(torch.zeros(1, 1, 128)), Variable(torch.zeros(1, 1, 128))
    inp = Variable(torch.zeros(1, 1)).long()
    pr, _ = rnn.predict_next_token('a', state)
    assert pr.sum().data[0] == 1.
    
    # check next_value behavior
    stat, _ = rnn.predict_next_value('a', state)
    assert stat.size() == (1, 1)


def test_logp():
    mu = torch.FloatTensor([5.])
    val = 5
    logp = _torch_logp_poisson(val, mu)
    assert math.isclose(logp[0], poisson.logpmf(val, mu[0]), rel_tol=1e-5)

    mu = torch.FloatTensor([5.])
    std = torch.FloatTensor([1.])
    val = 2.
    logp = _torch_logp_normal(val, mu, std)
    assert math.isclose(logp[0], norm.logpdf(val, mu[0], std[0]), rel_tol=1e-5)

    a = torch.FloatTensor([5.])
    b = torch.FloatTensor([1.])
    val = 0.3
    logp = _torch_logp_beta(val, a, b)
    assert math.isclose(logp[0], beta.logpdf(val, a[0], b[0]), rel_tol=1e-5)
 

def test_normalize():
    assert _normalize([0, 0, 10]) == [0, 0, 1.]
    assert _normalize([10, 0, 0, 10]) == [0.5, 0, 0, 0.5]
    assert _normalize([0, 10, 0, 10]) == [0, 0.5, 0, 0.5]
    assert _normalize([0, 10, 0, 10, 0]) == [0, 0.5, 0, 0.5, 0]
    assert _normalize([0, 10, 0, 20, 0, 70]) == [0, 0.1, 0, 0.2, 0, 0.7]


def test_rnn_walker():
    rules = r"""
        S = (T "+" S) / (T "*" S) / (T "/" S) / T
        T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
        po = "("
        pc = ")"
    """
    types = {"int": Int(1, 10)}
    grammar = build_grammar(rules, types=types)
    rules = extract_rules_from_grammar(grammar)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    
    model = RnnModel(vocab_size=len(tok_to_id), hidden_size=128)
    
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    wl = RnnWalker(grammar, rnn, min_depth=1, max_depth=5)
    wl.walk()
    t0 = wl.terminals[:]
    d0 = wl._decisions[:]
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    wl = RnnWalker(grammar, rnn, min_depth=1, max_depth=5)
    wl.walk()
    t = wl.terminals
    d = wl._decisions
    assert t == t0
    assert d == d0

    loss = wl.compute_loss()
    assert loss.size() == (1,)

    for min_depth in range(1, 10):
        wl = RnnWalker(grammar, rnn, min_depth=min_depth, max_depth=10)
        wl.walk()
        expr =''.join([str(t) for t in wl.terminals])
        node = grammar.parse(expr)
        depth = _get_max_depth(node)
        assert depth >= min_depth

def test_optimize():
    rules = r"""
        S = (T "+" S) / (T "*" S) / (T "/" S) / T
        T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
        po = "("
        pc = ")"
    """
    types = {"int": Int(1, 10)}
    grammar = build_grammar(rules, types=types)
    rules = extract_rules_from_grammar(grammar)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    model = RnnModel(vocab_size=len(tok_to_id), hidden_size=128)
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    wl = RnnWalker(grammar, rnn, min_depth=1, max_depth=5)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    codes, scores = optimize(_func, wl, optim, nb_iter=10)
    assert len(codes) == 10
    assert len(scores) == 10
    for c, s in zip(codes, scores):
        assert _func(c) == s


def _func(x):
    return 1 if "cos" in x else 0


def test_rnn_deterministic_walker():
    rules = r"""
        S = (T "+" S) / (T "*" S) / (T "/" S) / T
        T = (po S pc) / ("sin" po S pc) / ("cos" po S pc) / ("exp" po S pc) / "x" / int
        po = "("
        pc = ")"
    """
    types = {"int": Int(1, 10)}
    grammar = build_grammar(rules, types=types)
    rules = extract_rules_from_grammar(grammar)
    tok_to_id = {r: i for i, r in enumerate(rules)}
    
    expr = '(cos(x)+sin(x))*3'
    dwl = DeterministicWalker(grammar, expr)
    dwl.walk()
    
    model = RnnModel(vocab_size=len(tok_to_id), hidden_size=128)
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    wl = RnnDeterministicWalker(grammar, rnn, dwl.decisions)
    wl.walk() 
    assert wl.decisions == dwl.decisions
    assert wl.terminals == dwl.terminals
