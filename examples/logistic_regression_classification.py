"""
========================================================
GA hyper-parameter search for binary classification task
========================================================

This example demonstrates how to use `gasearch` to discover
hyperparameters of a LogisticRegression model used for binary classification.

"""

import numpy as np
from scipy.stats import uniform
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from gasearch import GeneticSearchCV

RANDOM_STATE = 1

param_dists = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l2', 'l1']
}

X,y = make_classification(random_state=RANDOM_STATE)

gc = GeneticSearchCV(
    LogisticRegression(solver='liblinear'),
    param_dists,
    scoring='accuracy',
    random_state=RANDOM_STATE,
    selection_algorithm="tournament")

res = gc.fit(X, y)

print(gc.best_params_, gc.best_score_, np.mean(gc.cv_results_['mean_score_time']))
