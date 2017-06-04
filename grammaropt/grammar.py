"""
This is the base module where the grammar can be built
and the base walkers are defined.
See `Walker` to know more about walkerrs.
"""
from types import MethodType
from random import Random
from copy import deepcopy
from collections import deque
from collections import namedtuple
from functools import partial
from functools import lru_cache

from parsimonious.grammar import Grammar
from parsimonious.expressions import Regex
from parsimonious.expressions import Sequence
from parsimonious.expressions import OneOf
from parsimonious.expressions import Compound
from parsimonious.expressions import Literal
from parsimonious.nodes import Node
from parsimonious.nodes import RegexNode

from .types import Type


def build_grammar(rules, types={}):
    """
    Build a Grammar object based on a set of rules and a set of types.
    The syntax of rules doc can be checked in `parsimoninous` documentation.
    Types are special kinds terminals that describe a type (for instance, Int or Float).
    Types are used so that walkers are aware of them and thus can do a special
    treatement to them (consider them as values instead of strings).

    Parameters
    ----------
    
    rules : str
        string following an EBNF-like syntax used by `parsimonious` to represent grammar rules.
    types : dict of str->Type
        maps a type name to a Type object.
        type names can be used inside the `rules` as terminals.

    Returns
    -------
    `parsimonious` Grammar object

    """
    more_rules = _build_type_rules(types)
    return Grammar(rules, **more_rules)


def extract_rules_from_grammar(grammar):
    """
    extract all the rules (including non-named ones) from a grammar

    Parameters
    ---------

    grammar : Grammar
        grammar from which to extract the rules

    Returns
    -------

    set of `parsimonious` `Expression` objects
    """
    full_rules = set()
    _extract_rules(grammar.values(), out=full_rules)
    return full_rules


def _extract_rules(rules, out=set()):
    """
    put recursively all rules from `rules` into `out`
    """
    for rule in rules:
        if rule not in out:
            out.add(rule)
            if isinstance(rule, Compound):
                _extract_rules(rule.members, out=out)


@lru_cache(maxsize=None)
def rule_depth(rule):
    return _rule_depth(rule, depths=None)


def _rule_depth(rule, depths=None):
    """
    get the minimal number of production rules to reach a terminal, starting
    from a rule `rule`.
    TODO : should be replaced by dijkstra algo for more efficiency
    """
    if depths is None:
        depths = {}
    # detecting and deal with cycles
    if rule in depths:
        return depths[rule]
    depths[rule] = float('inf')
    if isinstance(rule, OneOf):
        depth = 1 + min(map(partial(_rule_depth, depths=depths), rule.members))
    elif isinstance(rule, Sequence):
        depth = 1 + max(map(partial(_rule_depth, depths=depths), rule.members))
    else:
        depth = 0
    depths[rule] = depth
    return depth


def _build_type_rules(types):
    rules = {}
    for name, type in types.items():
        rules[name] = type
        rules[name].name = name #TODO is it necessary?
    return rules


class StopExpansion(Exception):
    pass


class Walker:
    """
    Walkers are objects that do a random walk (deterministic walks
    are a special case of random walks) on the production rules
    space of a given grammar : the responsability of a Walker 
    is to choose a production rule whenever it has to, and to 
    choose `Type` values whenever it has to. After the end of the walk,
    a trace is left describing the traversal of the space. The trace
    differs depending on the kind of Walker.

    Parameters
    ----------

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
        stop from being candidates to be chosen, but when only what we can
        choose are non-terminal production rules, we just choose one of them,
        even if `max_depth` is exceeded, otherwise the obtained string will
        not be a valid one according to the grammar.
    strict_depth_limit : bool
        if True, when `max_depth` is reached, forbid any further production rules 
        when a choice should be made.
        If False, even when `max_depth` is reached, choose terminals when terminals
        are available, otherwise keep applying production rules.

    """
    def __init__(self, grammar, min_depth=None, max_depth=None, strict_depth_limit=False):
        if min_depth and max_depth:
            assert min_depth <= max_depth
        self.grammar = grammar
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.strict_depth_limit = strict_depth_limit

    def next_rule(self, rules):
        """
        Given a set of production `rules`, choose the next one.
        Implemented by specific Walkers.
        `depth` is an `int` provides information about the current 
        depth of the parsetree.
        """
        raise NotImplementedError() 

    def next_value(self, rule):
        """
        Given a `Type` `rule`, choose its value.
        Implemented by specific Walkers.
        """
        raise NotImplementedError()

    def _init_walk(self):
        """
        Should be overloaded by specific Walkers.
        Called each time to initialize the walk, it can
        be used to empty the trace.
        """
        self.terminals = []

    def walk(self, start=None):
        """
        Do a random walk on the grammar.
        
        Parameters
        ----------

        start : str
            str that provides the starting rule name
            if `start` is not provides it uses the default rule of the grammar 
            (which is the first rule).
        """
        self._init_walk()
        grammar = self.grammar
        if start is None:
            rule = grammar.default_rule
        else:
            rule = grammar[start]
        # each element of the stack is (rule, depth)
        # top elements of the stack are at the right (first is older one, last newest one)
        stack = deque([(rule, 0)])
        while len(stack):
            rule, depth = stack.pop()
            if isinstance(rule, OneOf):
                members = self._filter_by_depth(rule.members, depth)
                if len(members):
                    chosen_rule = self.next_rule(members)
                    stack.append((chosen_rule, depth + 1))
            elif isinstance(rule, Sequence):
                members = rule.members
                members = [(m, depth + 1) for m in members]
                # put them in reverse order so that when popped the order
                # is the same than the order of `members` 
                stack.extend(members[::-1])
            elif isinstance(rule, Literal):
                val = rule.literal
                self.terminals.append(val)
            elif isinstance(rule, Type):
                val = self.next_value(rule)
                self.terminals.append(val)
    
    def _filter_by_depth(self, rules, depth):
        # use only non-terminals if we are belom `min_depth`
        # (only when possible, otherwise, when there are no terminals use the given rules as is)
        if self.min_depth is not None and depth <= self.min_depth:
            nonterminal_rules = [r for r in rules if isinstance(r, Compound)]
            return nonterminal_rules if len(nonterminal_rules) else rules
        # use only terminals if we are above `max_depth 
        # (only when possible, otherwise, when there are no terminals use the given rules as is
        #  and take the subset of rules with minimal rule depth (check _rule_depth))
        elif self.max_depth is not None and depth >= self.max_depth:
            terminal_rules = [r for r in rules if not isinstance(r, Compound)]
            if len(terminal_rules) == 0:
                if self.strict_depth_limit:
                    return []
                else:
                    depths = list(map(rule_depth, rules))
                    min_depth = min(depths)
                    rules = [r for r, d in zip(rules, depths) if d == min_depth]
                    return rules
            else:
                return terminal_rules
        else:
            return rules


_Decision =  namedtuple('Decision', ['rule', 'choice'])
class DeterministicWalker(Walker):
    """
    a very specific Walker that uses a given str expression `expr`, parse it
    usign the grammar `grammar`, then use the parse tree to force the next
    rule to choose and the next value to choose to correspond exactly to the
    expression `expr`. This very specific walker trace is used by the RNN walker
    to compute the loss for a given groundtruth expressions.
    Unfortunately I had to patch the uncached_match method of `parsimonious` `OneOf` and `Type`
    rules of the grammar to get some missing information required in the trace, 
    the missing information needed by `DeterministicWalker` was the parent rule that is used to 
    create a `Node`.
    """
    def __init__(self, grammar, expr):
        super().__init__(grammar)
        self.expr = expr
        self._init_walk()

    def _init_walk(self):
        super()._init_walk()
        self.decisions = []

    def walk(self):
        #WARNING : this function have a side effect on the grammar object `self.grammar`
        # but it cleans things up at the end of the call. I dont't find yet a better
        # way of doing that (see the description of `DeterministicWalker` to see why).
        grammar = self.grammar
        self._init_walk()
        rules = extract_rules_from_grammar(grammar)

        # patch OneOf and Type _uncached_match method
        for rule in rules:
            if isinstance(rule, OneOf):
                rule._uncached_match_orig = rule._uncached_match
                rule._uncached_match = MethodType(_patched_oneof_match, rule)
            elif isinstance(rule, Type):
                rule._uncached_match_orig = rule._uncached_match
                rule._uncached_match = MethodType(_patched_type_match, rule)
        node = grammar.parse(self.expr)
        stack = deque([node])
        while len(stack):
            node = stack.pop()
            # these are all nodes where choices have been made (that is, either `OneOf` or `Type`)
            if hasattr(node, 'parent_rule'):
                # OneOf nodes
                if isinstance(node.parent_rule, OneOf):
                    self.decisions.append(_Decision(rule=node.parent_rule, choice=node.rule))
                # Type nodes
                elif isinstance(node.parent_rule, Type):
                    val = node.parent_rule.from_str(node.text)
                    self.decisions.append(_Decision(rule=node.parent_rule, choice=val))
                else:
                    raise TypeError('Unrecognize parent rule : {}'.format(node.parent_rule))
            # terminals
            if len(node.children) == 0:
                val = node.text
                if hasattr(node, 'parent_rule') and isinstance(node.parent_rule, Type):
                    val = node.parent_rule.from_str(val)
                self.terminals.append(val)
            stack.extend(node.children[::-1])

        # unpatch OneOf and Type _uncached_match method
        for rule in rules:
            if hasattr(rule, '_uncached_match_orig'):
                rule._uncached_match = rule._uncached_match_orig


def _patched_oneof_match(self, text, pos, cache, error):
    # pasted from pasimonious source code (parsimonious.expressions.OneOf)
    # only difference is two lines
    for m in self.members:
        node = m.match_core(text, pos, cache, error)
        if node is not None:
            node = Node(self.name, text, pos, node.end, children=[node])
            node.parent_rule = self # newly addde line
            node.rule = m # newly added line
            return node


def _patched_type_match(self, text, pos , cache, error):
    # pasted from `parsimonious` source code (parsimonious.expressions.Regex)
    # only difference is one line
    m = self.re.match(text, pos)
    if m is not None:
        span = m.span()
        node = RegexNode(self.name, text, pos, pos + span[1] - span[0])
        node.match = m  # TODO: A terrible idea for cache size?
        node.parent_rule = self # newly added line
    return node


def as_str(terminals):
    return ''.join(map(str, terminals))
