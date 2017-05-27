from collections import namedtuple
import numpy as np
import math

import torch.nn.init as init
import torch
import torch.nn as nn
from torch.autograd import Variable


from .grammar import Walker
from .types import Int


class RnnModel(nn.Module):
    
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
        idx = self.tok_to_id[tok]
        return torch.log(pred[0, idx])

    def value_logp(self, tok, val, stats):
        if type(tok) == Int:
            mu = stats[0][0]
            mu = torch.tanh(mu)
            mu = (mu + 1) / 2.
            mu = mu * (tok.high - tok.low) + tok.low
            logp = val * torch.log(mu) - mu - _log_factorial(val)
            return logp
        else:
            raise ValueError('Unrecognized type : {}'.format(tok))

    def _preprocess_input(self, inp):
        if inp is None:
            inp = self.begin_tok
        cur_id = self.tok_to_id[inp]
        x = torch.zeros(1, 1).long() # batch_size, nb_timesteps
        x.fill_(cur_id)
        x = Variable(x)
        return x

def _log_factorial(k):
    return sum(math.log(i) for i in range(1, k + 1))

def _normalize(vals):
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
    

    def __init__(self, rnn):
        self.rnn = rnn
        self.reset()

    def reset(self):
        super().reset()
        self._input = None
        self._state = None
        self._decisions = []
    
    def next_rule(self, rules):
        pr, self._state = self.rnn.predict_next_token(self._input, self._state)
        rule = self.rnn.generate_next_token(pr, allowed=rules)
        self._input = rule
        self._decisions.append(Decision(action='rule', given=rules, pred=pr, gen=rule))
        return rule

    def next_value(self, rule):
        stats, _ = self.rnn.predict_next_value(self._input, self._state)
        val = self.rnn.generate_next_value(stats, rule)
        self._decisions.append(Decision(action='value', given=rule, pred=stats, gen=val))
        return val

    def compute_loss(self):
        loss_toks = 0.
        loss_vals = 0.
        for dc in self._decisions:
            if dc.action == 'rule':
                loss_toks += -self.rnn.token_logp(dc.gen, dc.pred)
            elif dc.action == 'value':
                loss_vals += -self.rnn.value_logp(dc.given, dc.gen, dc.pred)
        loss = loss_toks + loss_vals
        return loss


class RnnWalkerWithGroundtruth(RnnWalker):
    
    def __init__(self, rnn):
        super().__init__(rnn)
        self._groundtruth = None

    def groundtruth_(self, gt):
        self._groundtruth = gt[::-1]

    def next_rule(self, rules):
        assert self._groundtruth is not None
        pr, self._state = self.rnn.predict_next_token(self._input, self._state)
        parent_rule, rule = self._groundtruth.pop()
        assert parent_rule.members == rules
        self._input = rule
        self._decisions.append(Decision(action='rule', given=rules, pred=pr, gen=rule))
        return rule

    def next_value(self, rule):
        assert self._groundtruth is not None
        stats, _ = self.rnn.predict_next_value(self._input, self._state)
        rule_, val = self._groundtruth.pop()
        assert rule_ == rule
        self._decisions.append(Decision(action='value', given=rule, pred=stats, gen=val))
        return val
