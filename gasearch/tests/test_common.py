import pytest
from scipy.stats import uniform
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.utils._testing import (MinimalClassifier, MinimalRegressor,
                                    MinimalTransformer)

from .._search import GeneticSearchCV

test_distributions = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l2', 'l1']
}

RANDOM_STATE = 1

X,y = make_classification(random_state=RANDOM_STATE)

@pytest.mark.parametrize(
    "estimator", [LogisticRegression(solver='liblinear')]
)
def test_genetic_search(estimator):
    gc = GeneticSearchCV(estimator, test_distributions, random_state=RANDOM_STATE, scoring='accuracy')
    gc.fit(X, y)
    assert gc.best_params_['penalty'] == 'l1'

@pytest.mark.parametrize(
    'estimator', [MinimalClassifier(), MinimalRegressor()]
)
def test_scoring(estimator):
    genetic_search = GeneticSearchCV(estimator, test_distributions, random_state=RANDOM_STATE)
    genetic_search.fit(X, y)
    genetic_search.score(X, y)

def test_predict_proba():
    genetic_search = GeneticSearchCV(MinimalClassifier(), test_distributions, random_state=RANDOM_STATE)
    genetic_search.fit(X, y)
    genetic_search.predict_proba(X)

def test_transform():
    genetic_search = GeneticSearchCV(MinimalTransformer(), test_distributions, scoring='accuracy', random_state=RANDOM_STATE)
    genetic_search.fit(X)
    genetic_search.transform(X)
