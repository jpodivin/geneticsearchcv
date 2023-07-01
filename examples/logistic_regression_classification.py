import numpy as np
from scipy.stats import uniform
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from geneticsearchcv._search import GeneticSearchCV

param_dists = {
    'C': uniform(loc=0, scale=4),
    'penalty': ['l2', 'l1']
}

X,y = make_classification(random_state=10)
gc = GeneticSearchCV(LogisticRegression(solver='liblinear'), param_dists, scoring='accuracy')
res = gc.fit(X, y)
print(gc.best_params_, gc.best_score_, np.mean(gc.cv_results_['mean_score_time']))
