"""
This module provides a simple random Walker which selects
production rules and values uniformly at random.
"""
from collections import defaultdict
import numpy as np
from parsimonious.expressions import Compound, Sequence
from .grammar import Walker


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
    def __init__(self, grammar, min_depth=None, max_depth=None, strict_depth_limit=False, random_state=None):
        assert min_depth <= max_depth
        super().__init__(grammar, min_depth=min_depth, max_depth=max_depth, strict_depth_limit=strict_depth_limit)
        self.rng = np.random.RandomState(random_state)

    def next_rule(self, rules):
        return self.rng.choice(rules)

    def next_value(self, rule):
        return rule.uniform_sample(self.rng)
