from types import MethodType
from random import Random
from copy import deepcopy
from collections import deque

from parsimonious.grammar import Grammar
from parsimonious.expressions import Regex
from parsimonious.expressions import Sequence
from parsimonious.expressions import OneOf
from parsimonious.expressions import Compound
from parsimonious.expressions import Literal
from parsimonious.nodes import Node
from parsimonious.nodes import RegexNode

from .types import Type, Int


def build_grammar(rules, types={}):
    more_rules = _build_type_rules(types)
    return Grammar(rules, **more_rules)


def extract_rules_from_grammar(grammar):
    full_rules = set()
    _extract_rules(grammar.values(), out=full_rules)
    return full_rules


def _extract_rules(rules, out=set()):
    for rule in rules:
        if isinstance(rule, Compound) and rule not in out:
            out.add(rule)
            _extract_rules(rule.members, out=out)
        else:
            out.add(rule)


def _build_type_rules(types):
    rules = {}
    for name, type in types.items():
        rules[name] = type
        rules[name].name = name #TODO is it necessary?
    return rules


class Walker:
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.terminals = []

    def next_rule(self, rules):
        raise NotImplementedError() 

    def next_value(self, rule):
        raise NotImplementedError()
    
    def walk(self, grammar, start=None):
        self.reset()
        if start is None:
            rule = grammar.default_rule
        else:
            rule = grammar[start]
        stack = deque([rule])
        while len(stack):
            rule = stack.pop()
            if isinstance(rule, OneOf):
                chosen_rule = self.next_rule(rule.members)
                stack.append(chosen_rule)
            elif isinstance(rule, Sequence):
                stack.extend(rule.members[::-1])
            elif isinstance(rule, Literal):
                val = rule.literal
                self.terminals.append(val)
            elif isinstance(rule, Type):
                val = self.next_value(rule)
                self.terminals.append(val)


class StringWalker(Walker):

    def __init__(self, string):
        self.string = string
        self.reset()

    def reset(self):
        self.decisions = []

    def walk(self, grammar):
        self.reset()
        rules = extract_rules_from_grammar(grammar)
        for rule in rules:
            if isinstance(rule, OneOf):
                rule._uncached_match = MethodType(_patched_oneof_match, rule)
            elif isinstance(rule, Type):
                rule._uncached_match = MethodType(_patched_type_match, rule)
        node = grammar.parse(self.string)
        stack = deque([node])
        while len(stack):
            node = stack.pop()
            if hasattr(node, 'parent_rule'):
                if hasattr(node, 'rule'):
                    self.decisions.append((node.parent_rule, node.rule))
                else:
                    self.decisions.append((node.parent_rule, node.text))
            stack.extend(node.children[::-1])


def _patched_oneof_match(self, text, pos, cache, error):
    for m in self.members:
        node = m.match_core(text, pos, cache, error)
        if node is not None:
            node = Node(self.name, text, pos, node.end, children=[node])
            node.parent_rule = self
            node.rule = m
            return node

def _patched_type_match(self, text, pos , cache, error):
    m = self.re.match(text, pos)
    if m is not None:
        span = m.span()
        node = RegexNode(self.name, text, pos, pos + span[1] - span[0])
        node.match = m  # TODO: A terrible idea for cache size?
        node.parent_rule = self
    return node
