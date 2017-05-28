"""
This module provides two RNN Walkers (one is random, the other deterministic) 
and an a simple torch-based RNN model. The RNN Walker and RNN model communicate
through an RNN Adapter, so that several kinds of models can be used with an RNN Walker.
"""
from collections import namedtuple
import numpy as np
import math

import torch.nn.init as init
import torch
import torch.nn as nn
from torch.autograd import Variable

from parsimonious.expressions import Compound, Sequence

from .grammar import Walker
from .types import Int


class RnnModel(nn.Module):
    """
    A simple LSTM torch based model.
    It implements `next_token` and `next_value` which are necessary
    to communicate with `RnnAdapter`.

    Parameters
    ----------
    
    vocab_size : int
        vocabulary size.
        Practically, it corresponds to the number of possible production
        rules. From the RNN perspective, each production rule is a token.
    
    emb_size : int
        Embedding size

    hidden_size : int
        number of hidden units of the lSTM

    nb_features : int
        number of real-valued features to predict when a value is needed.
        Practically, it corresponds to the number of statistics to predict 
        needed to sample from a `Type`. 
        
    """
    def __init__(self, vocab_size=10, emb_size=128, hidden_size=128, nb_features=1):
        super().__init__()   
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.hidden_size = hidden_size

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, batch_first=True)
        self.out_token  = nn.Linear(hidden_size, vocab_size)
        self.out_value = nn.Linear(hidden_size, nb_features)

    def next_token(self, inp, state):
        x = self.emb(inp)
        _, state = self.lstm(x, state)
        h, c = state
        h = h.view(h.size(0) * h.size(1), h.size(2))
        o = self.out_token(h)
        return o, state
    
    def next_value(self, inp, state):
        x = self.emb(inp)
        _, state = self.lstm(x, state)
        h, c = state
        h = h.view(h.size(0) * h.size(1), h.size(2))
        o = self.out_value(h)
        return o, state


class RnnAdapter:
    """
    `RnnAdapter` adapts a Model to communicate with an RNN `Walker` so that
    different Models, adapted with the same `RnnAdapter` can be used in an
    RNN `Walker`.

    Parameters
    ----------

    model : Model to adapt

    tok_to_id : dict of Token->Int
        Practically, `Token` corresponds to a token, which in our case is a 
        `parsimonious` `Expression`, which refers to a production rule.
        So this dict convert a production rule to an integer identifying the
        produciton rule.

    begin_tok : Token
        this Token (practically corresponding to a production rule, which is
        a `parsimonious` `Expression`) is given as input in the first step
        the RNN. If not provided, the token with ID 0 (that is, the `tok` for which tok_to_id[tok] ==0)
        is used.

    """
    def __init__(self, model, tok_to_id, begin_tok=None, random_state=None):
        assert model.vocab_size == len(tok_to_id)
        self.model = model
        self.tok_to_id = tok_to_id
        self.id_to_tok = {v: k for k, v in tok_to_id.items()}
        if begin_tok is None:
            begin_tok = self.id_to_tok[0]
        self.begin_tok = begin_tok
        self.rng = np.random.RandomState(random_state)

    def predict_next_token(self, inp, state):
        x = self._preprocess_input(inp)
        o, state = self.model.next_token(x, state)
        pr = nn.Softmax()(o)
        return pr, state

    def generate_next_token(self, pr, allowed=None):
        pr = pr[0].data.clone()
        if allowed is not None:
            full = set(self.tok_to_id.keys())
            forbidden = full - set(allowed)
            for tok in forbidden:
                pr[self.tok_to_id[tok]] = 0.
        pr = pr.tolist()
        pr = _normalize(pr)
        ids = list(range(len(pr)))
        next_id = self.rng.choice(ids, p=pr)
        tok = self.id_to_tok[next_id]
        return tok

    def predict_next_value(self, inp, state):
        x = self._preprocess_input(inp)
        o, state = self.model.next_value(x, state)
        return o, state

    def generate_next_value(self, stat, tok):
        stat = stat.data.tolist()
        if type(tok) == Int:
            mu = stat[0][0]
            mu = math.tanh(mu)
            mu = (mu + 1) / 2.
            mu = mu * (tok.high - tok.low) + tok.low
            val = self.rng.poisson(mu)
            val = min(val, tok.high)
            val = max(val, tok.low)
        else:
            raise ValueError('Unrecognized type : {}'.format(tok))
        return val

    def token_logp(self, tok, pred):
        """
        compute the log probability of a Token `tok` from a a set
        of predicted probabilities.
        """
        idx = self.tok_to_id[tok]
        return torch.log(pred[0, idx])

    def value_logp(self, tok, val, stats):
        """
        compute the log probability of a value `val` given the
        sufficient statistics `stats`. The probability distribution
        depends on the kind of the token `tok`.

        For `Int`, we use `poisson`.
        For `Float`, we use `Gaussian`.
        """
        if type(tok) == Int:
            mu = stats[0][0]
            # convert to [-1, 1]
            mu = torch.tanh(mu)
            # convert to [0, 1]
            mu = (mu + 1) / 2.
            # convert to [low, high] provided by the  Int `tok`
            mu = mu * (tok.high - tok.low) + tok.low
            # mu represents the mean of a poisson.
            # compute logp of a poisson with mean `mu`
            logp = val * torch.log(mu) - mu - _log_factorial(val)
            return logp
        else:
            raise ValueError('Unrecognized type : {}'.format(tok))

    def _preprocess_input(self, inp):
        # convert `inp` which is a Token to an integer
        # then to a torch `Variable` so that it is ready
        # to be an input of a torch model
        if inp is None:
            inp = self.begin_tok
        cur_id = self.tok_to_id[inp]
        x = torch.zeros(1, 1).long() # batch_size, nb_timesteps
        x.fill_(cur_id)
        x = Variable(x)
        return x


def _log_factorial(k):
    """
    compute $log(k!)$
    """
    return sum(math.log(i) for i in range(1, k + 1))


def _normalize(vals):
    """
    Normalize a list of values so that they sum up to 1.

    Assume that a set of elements have the exact value of 0.
    Divide by the sum, the compute the sum of vals[0:-1] then
    set vals[-1] to 1 - vals[0:-1].
    This wouldnt work if some values are 0, so the same operations
    defined by above are done for `vals` fitlered by non-zero values
    then the full `vals` is returned where values that were 0 are still
    zero, and the others are such that `vals` is normalized.
    
    Parameters
    ----------

    vals : list of float

    Returns
    -------

    list of float
    
    """
    s = sum(vals)
    vals = [v / s for v in vals]
    ivals = [(i, v) for i, v in enumerate(vals) if v > 0]
    first = sum(v for i, v in ivals[0:-1])
    if len(ivals) > 1:
        i, _ = ivals[-1]
        ivals[-1] = (i, 1 - first)
    vals = [0. for _ in range(len(vals))]
    for i, v in ivals:
        vals[i] = v
    return vals


Decision = namedtuple('Decision', ['action', 'given', 'pred', 'gen'])
class RnnWalker(Walker):

    """
    grammar : Grammar
        grammar where to walk
    rnn : RnnAdapter
        RNN Adapter used to communicate with a Model
    min_depth : int
        minimum depth of the parse tree.
    max_depth : int
        maximum depth of the parse tree.
        Note that it could exceed `max_depth` because when it reaches
        `max_depth` there is no garanthee that there would always be
        a terminal production rule to choose. The solution to this problem
        is that when `max_depth` is reached, non-terminal production rules
        stop from being candidates to be chosen.

    """
    def __init__(self, grammar, rnn, min_depth=1, max_depth=5):
        super().__init__(grammar)
        self.rnn = rnn
        self.min_depth = min_depth
        self.max_depth = max_depth

    def _init_walk(self):
        super()._init_walk()
        self._input = None
        self._state = None
        self._decisions = []
    
    def next_rule(self, rules, depth=0):
        pr, self._state = self.rnn.predict_next_token(self._input, self._state)
        rule = self._generate_rule(pr, rules, depth=depth)
        self._input = rule
        self._decisions.append(Decision(action='rule', given=rules, pred=pr, gen=rule))
        return rule
    
    def _generate_rule(self, pr, rules, depth=0):
        # use only non-terminals if we are belom `min_depth`
        # (only when possible, otherwise, when there are no terminals use the given rules as is)
        if depth <= self.min_depth:
            rules_ = [r for r in rules if isinstance(r, Compound)]
        # use only terminals if we are above `max_depth 
        # (only when possible, otherwise, when there are no terminals use the given rules as is)
        elif depth >= self.max_depth:
            rules_ = [r for r in rules if not isinstance(r, Compound)]
        else:
            rules_ = rules
        if len(rules_):
            rules = rules_
        return self.rnn.generate_next_token(pr, allowed=rules)

    def next_value(self, rule):
        stats, _ = self.rnn.predict_next_value(self._input, self._state)
        val = self._generate_value(stats, rule)
        self._decisions.append(Decision(action='value', given=rule, pred=stats, gen=val))
        return val

    def _generate_value(self, stats, rule):
        return self.rnn.generate_next_value(stats, rule)

    def compute_loss(self):
        """
        Compute the log probability of the sequence of decisions
        in _decisions that the RNN have taken.
        This is used to update the parameters of the `Model` to
        maximize the probability of the decisions that the RNN
        have taken.
        """
        loss_toks = 0.
        loss_vals = 0.
        for dc in self._decisions:
            if dc.action == 'rule':
                loss_toks += -self.rnn.token_logp(dc.gen, dc.pred)
            elif dc.action == 'value':
                loss_vals += -self.rnn.value_logp(dc.given, dc.gen, dc.pred)
        loss = loss_toks + loss_vals
        return loss


class RnnDeterministicWalker(RnnWalker):
    
    """
    RnnWalker but where we don't use the RNN to generate, we provide
    a groundtruth set of production rules and we collect the trace that
    the RNN model would produce if it had generated the groundtruth.
    This is used to compute the loss of a groundtruth expression,
    represented through `decisions`, which are `groundtruth` decisions
    that the RNN should have taken to generate an expression.
    `decisions` can be obtained by using a `DeterministicWalker`.
    """
    def __init__(self, grammar, rnn, decisions, min_depth=1, max_depth=5):
        super().__init__(grammar, rnn, min_depth=min_depth, max_depth=max_depth)
        self.decisions = decisions
    
    def _init_walk(self):
        super()._init_walk()
        self._external_decisions = self.decisions[::-1]
    
    def _generate_rule(self, pr, rules, depth=0):
        parent_rule, rule = self._external_decisions.pop()
        assert parent_rule.members == rules
        return rule

    def _generate_value(self, stats, rule):
        rule_, val = self._external_decisions.pop()
        assert rule_ == rule
        return val
