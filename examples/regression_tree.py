import numpy as np
from scipy.stats import uniform
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from geneticsearchcv import GeneticSearchCV

RANDOM_STATE = 1

param_dists = {
    'max_depth': [i for i in range(1, 50)],
    'min_samples_leaf': [i for i in range(3, 15)],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    'ccp_alpha': uniform(0, 1)
}
X,y = make_regression(random_state=RANDOM_STATE)
gc = GeneticSearchCV(
    DecisionTreeRegressor(),
    param_dists,
    scoring='neg_mean_squared_error',
    n_iter=10,
    random_state=RANDOM_STATE,
    pop_size=100)

res = gc.fit(X, y)
print(gc.best_params_, gc.best_score_, np.mean(gc.cv_results_['mean_score_time']))
