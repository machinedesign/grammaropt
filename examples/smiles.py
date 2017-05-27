from grammaropt.grammar import build_grammar
from grammaropt.random import RandomWalker

from molecules.molecule import is_valid

def main():
    rules = r"""
    smiles  =  chain
    atom  =  bracket_atom / aliphatic_organic / aromatic_organic
    aliphatic_organic  =  "B" / "C" / "N" / "O" / "S" / "P" / "F" / "I" / "Cl" / "Br"
    aromatic_organic  =  "c" / "n" / "o" / "s"
    bracket_atom  =  "[" BAI "]"
    BAI  =  (isotope symbol BAC) / (symbol BAC) / (isotope symbol) / symbol
    BAC  =  (chiral BAH) / BAH / chiral
    BAH  =  (hcount BACH) / BACH / hcount
    BACH  =  (charge class) / charge / class
    symbol  =  aliphatic_organic / aromatic_organic
    isotope  =  DIGIT / (DIGIT DIGIT) / (DIGIT DIGIT DIGIT)
    DIGIT  =  "1" / "2" / "3" / "4" / "5" / "6" / "7" / "8"
    chiral  =  "@" / "@@"
    hcount  =  "H" / ("H" DIGIT)
    charge  =  "-" / ("-" DIGIT) / ("-" DIGIT DIGIT) / "+" / ("+" DIGIT) / ("+" DIGIT DIGIT)
    bond  =  "-" / "=" / "#" / "/" / "\"
    ringbond  =  DIGIT / (bond DIGIT)
    branched_atom  =  atom / (atom RB) / (atom BB) / (atom RB BB)
    RB  =  (RB ringbond) / ringbond
    BB  =  (BB branch) / branch
    branch  = ("(" chain ")" ) / ( "(" bond chain ")" )
    chain  =  branched_atom / (chain branched_atom) / (chain bond branched_atom)
    class = ":" digits
    digits = DIGIT / digits
    """
    print(rules)
    grammar = build_grammar(rules)
    walker = RandomWalker()
    walker.walk(grammar)
    s = ''.join(walker.terminals)
    print(s) 
    print(is_valid(s))

if __name__ == "__main__":
    main()
