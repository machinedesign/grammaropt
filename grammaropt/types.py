"""
This module contain types, which are special kinds of
grammar terminals that represent pre-defined types like int, float, etc.
The main goal of theses classes is not to provide functionality (they 
are basically a Regex) but rather to be identified as Type so that
some special treatment is reserved to them, because for inst. the RNN
based walker needs to know that there are types so that it predicts a value.
"""
import re

from parsimonious.expressions import Regex

class Type(Regex):
    pass


class Int(Type):
    """
    Integer `Type`, defined in an interval [low, high]
    (low and high are included).
    """
    def __init__(self, low, high):
        assert type(low) == int and type(high) == int
        assert low <= high
        super()
        self.name = ''
        self.low = low
        self.high = high
        self.re = re.compile('[0-9]+')#TODO do the actual regex
    
    def uniform_sample(self, rng):
        # assumes rng is `np.random.Random` rather than `random.Random`
        return rng.randint(self.low, self.high + 1)

    @staticmethod
    def from_str(s):
        return int(s)
