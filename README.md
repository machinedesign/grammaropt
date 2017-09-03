# GrammarOpt

Grammaropt is a python framework for defining domain specific languages (DSLs) using context-free grammars in order
to perform optimization over the space of instances of the DSL. 
For instance, it has been used to define and optimize over a grammar of scikit-learn machine learning pipelines 
in python (http://scikit-learn.org/stable/), similarly to TPOT (https://github.com/rhiever/tpot/).

The following are the typical steps that are performed to optimize overe a grammar: 

## 1) Define the grammar

First, a grammar is defined. Grammaropt uses parsimonious (https://github.com/erikrose/parsimonious/) to define
grammars.
A simple scikit-learn based grammar could be defined as:

```python

from grammaropt.grammar import build_grammar

rules = """
  pipeline = "make_pipeline" "(" elements "," estimator ")"
  elements = (element "," elements) / element
  element = pca / select_from_model
  pca = "PCA" "(" "n_components" "=" int ")"
  select_from_model = "SelectFromModel" "(" estimator ")"
  estimator = rf / logistic
  logistic = "LogisticRegression" "(" ")"
  rf = "RandomForestClassifier" "(" "max_depth" "=" int "," "max_features" "=" float ")"
  int = "10" / "20" / "30" / "40"
  float = "0.1" / "0.3" / "0.5"
"""

grammar = build_grammar(rules)
```
An example instance from the above defined grammar could be :

```python
make_pipeline(PCA(n_components=30),LogisticRegression())
```

## 2) Define the evaluation function

Second, an evaluation function that takes a string instance from the grammar and returns a value.
The goal is to find the string that has the maximum or minimum value of the evaluation function.
For machine learning pipelines, the evaluation can for instance return the accuracy of the model
on some dataset.


```python
from sklearn import datasets
from sklearn.model_selection import cross_val_score

digits = datasets.load_digits()
X, y = digits.data, digits.target

def evaluate(code):
    clf = eval(code)
    try:
        scores = cross_val_score(clf, X, y, cv=5)
    except Exception:
        return 0.
    else:
        return float(np.mean(scores))
```

## 3) Select the Walker and optimize

The main feature of this framework are the Walkers. Walkers are classes that implement different ways of exploring
the space of strings. The most straightforward Walker is the RandomWalker which samples uniformly from the grammar.
A more complex Walker is RnnWalker, which is a learnable Walker that can be updated in order to sample more often
strings that lead to high (or low) value of the evaluation function.


```python
from grammaropt.random import RandomWalker
from grammaropt.random import optimize
import numpy as np

walker = RandomWalker(grammar)
codes, scores = optimize(evaluate, walker, nb_iter=10)

idx = np.argmax(scores)
best_code = codes[idx]
best_score = scores[idx]
print('Best code : {}'.format(best_code))
print('Best score : {}'.format(best_score))
```

## Putting everything together

```python
import numpy as np

from sklearn import datasets
from sklearn.model_selection import cross_val_score

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from grammaropt.grammar import build_grammar
from grammaropt.random import RandomWalker
from grammaropt.random import optimize

rules = """
    pipeline = "make_pipeline" "(" elements "," estimator ")"
    elements = (element "," elements) / element
    element = pca / select_from_model
    pca = "PCA" "(" "n_components" "=" int ")"
    select_from_model = "SelectFromModel" "(" estimator ")"
    estimator = rf / logistic
    logistic = "LogisticRegression" "(" ")"
    rf = "RandomForestClassifier" "(" "max_depth" "=" int "," "max_features" "=" float ")"
    int = "10" / "20" / "30" / "40"
    float = "0.1" / "0.3" / "0.5"
"""

grammar = build_grammar(rules)

digits = datasets.load_digits()
X, y = digits.data, digits.target

def evaluate(code):
    clf = eval(code)
    try:
        scores = cross_val_score(clf, X, y, cv=5)
    except Exception:
        return 0.
    else:
        return float(np.mean(scores))

walker = RandomWalker(grammar)
codes, scores = optimize(evaluate, walker, nb_iter=10)

idx = np.argmax(scores)
best_code = codes[idx]
best_score = scores[idx]
print('Best code : {}'.format(best_code))
print('Best score : {}'.format(best_score))
```
