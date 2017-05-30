"""
This module provides a simple random Walker which selects
production rules and values uniformly at random.
"""
from collections import defaultdict
import numpy as np
from parsimonious.expressions import Compound, Sequence
from .grammar import Walker
from .types import Int

class RandomWalker(Walker):
    """
    a random Walker that selects production rules and values uniformly
    at random. This is a very good baseline.

    grammar : Grammar
        grammar where to walk
    min_depth : int
        minimum depth of the parse tree.
    max_depth : int
        maximum depth of the parse tree.
        Note that it could exceed `max_depth` because when it reaches
        `max_depth` there is no garanthee that there would always be
        a terminal production rule to choose. The solution to this problem
        is that when `max_depth` is reached, non-terminal production rules
        stop from being candidates to be chosenm, but when only what we can
        choose are non-terminal production rules, we just choose one of them,
        even if `max_depth` is exceeded, otherwise the obtained string will
        not be a valid one according to the grammar.
    """
    def __init__(self, grammar, min_depth=1, max_depth=10, random_state=None):
        assert min_depth <= max_depth
        super().__init__(grammar)
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.rng = np.random.RandomState(random_state)

    def next_rule(self, rules, depth=0):
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
            return self.rng.choice(rules_)
        else:
            return self.rng.choice(rules)

    def next_value(self, rule):
        return rule.uniform_sample(self.rng)
