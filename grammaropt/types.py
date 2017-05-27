import re

import numpy as np

from parsimonious.expressions import Regex

class Type(Regex):
    pass


class Int(Type):

    def __init__(self, low, high):
        assert type(low) == int and type(high) == int
        assert low <= high
        super()
        self.name = ''
        self.low = low
        self.high = high
        self.re = re.compile('[0-9]+')#TODO do the actual regex

    def __deepcopy__(self, memo):
        # for StringWalker only...
        return Int(self.low, self.high)
