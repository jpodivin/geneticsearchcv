import numpy as np
from scipy.stats import uniform
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor

from geneticsearchcv._search import GeneticSearchCV

param_dists = {
    'max_depth': [i for i in range(1, 50)],
    'min_samples_leaf': [i for i in range(3, 15)],
    'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],
    'ccp_alpha': uniform(0, 1)
}
X,y = make_regression(random_state=10)
gc = GeneticSearchCV(
    DecisionTreeRegressor(),
    param_dists,
    verbose=3,
    scoring='neg_mean_squared_error',
    n_iter=50)
res = gc.fit(X, y)
print(gc.best_params_, gc.best_score_, np.mean(gc.cv_results_['mean_score_time']))
