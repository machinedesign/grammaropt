import numpy as np

from parsimonious.expressions import Compound, Sequence
from .grammar import Walker
from .types import Int

class RandomWalker(Walker):
    
    def __init__(self, random_state=None):
        self.rng = np.random.RandomState(random_state)

    def next_rule(self, rules, depth=None):
        return self.rng.choice(rules)

    def next_value(self, rule):
        if type(rule) == Int:
            return self.rng.randint(rule.low, rule.high)
        else:
            raise ValueError('Unrecognized type : {}'.format(rule))
