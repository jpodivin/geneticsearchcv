"""
===============================================================
GA hyper-parameter search for pipeline with k-nearest neighbors
===============================================================

This example demonstrates how to use `gasearch` to discover
hyperparameters of a k-nearest neighbors classifier in a pipeline
with pre-processing step.

"""
from sklearn.neighbors import KNeighborsRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from gasearch import GeneticSearchCV
import numpy as np

X, y = make_regression(n_samples=1000, n_features=50)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

feature_selector = VarianceThreshold()
estimator = KNeighborsRegressor()
pipeline = Pipeline([('features', feature_selector), ('estimator', estimator)])

params = {
    "estimator__n_neighbors": np.arange(5, 30, 2),
    "estimator__weights": ["uniform", "distance"],
    "estimator__p": np.arange(2, 7)
}
optimizer = GeneticSearchCV(pipeline, params, n_iter=100)

optimizer.fit(X_train, y_train)

optimizer.best_params_

optimizer.best_estimator_.score(X_test, y_test)
