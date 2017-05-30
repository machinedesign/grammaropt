import pytest

import numpy as np

from grammaropt.rnn import RnnModel
from grammaropt.rnn import RnnWalker
from grammaropt.rnn import RnnDeterministicWalker
from grammaropt.rnn import RnnAdapter
from grammaropt.rnn import _normalize
from grammaropt.grammar import DeterministicWalker
from grammaropt.grammar import extract_rules_from_grammar
from grammaropt.grammar import build_grammar
from grammaropt.types import Int

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

    with pytest.raises(AssertionError):
        model = RnnModel(vocab_size=3)
        rnn = RnnAdapter(model, tok_to_id)
    
    model = RnnModel(vocab_size=4, hidden_size=128)
    rnn = RnnAdapter(model, tok_to_id, random_state=42)
    
    with pytest.raises(AssertionError):
        pr = Variable(torch.from_numpy(np.array([0.1, 0.1, 0.1])).view(1, -1))
        rnn.generate_next_token(pr)
     
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

    tok = Int(1, 10)
    stat = Variable(torch.from_numpy(np.array([5.])).view(1, 1))
    val = rnn.generate_next_value(stat, tok)
    assert 1 <= val <= 10
    
    with pytest.raises(TypeError):
        val = rnn.generate_next_value(stat, 'a')

    val = rnn.token_logp('a', pr)
    assert val.size() == (1,)
    assert val.data[0] == np.log(pr[0, tok_to_id['a']])

    val = rnn.value_logp(tok, 5, stat)
    assert val.size() == (1,)
    
    with pytest.raises(TypeError):
        val = rnn.value_logp('a', 5, stat)

    state = Variable(torch.zeros(1, 1, 128)), Variable(torch.zeros(1, 1, 128))
    inp = Variable(torch.zeros(1, 1)).long()
    pr, _ = rnn.predict_next_token('a', state)
    assert pr.sum().data[0] == 1.

    stat, _ = rnn.predict_next_value('a', state)
    assert stat.size() == (1, 1)

 

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
    wl = RnnDeterministicWalker(grammar, rnn, dwl.decisions, min_depth=1, max_depth=5)
    wl.walk() 
    assert wl.decisions == dwl.decisions
    assert wl.terminals == dwl.terminals
