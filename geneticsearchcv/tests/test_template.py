import pytest
import numpy as np

from sklearn.datasets import load_iris
from numpy.testing import assert_array_equal
from numpy.testing import assert_allclose

from geneticsearchcv import GeneticSearchCV


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_search(data):
    est = GeneticSearchCV()
    assert est.demo_param == 'demo_param'

    est.fit(*data)
    assert hasattr(est, 'is_fitted_')

    X = data[0]
    y_pred = est.predict(X)
    assert_array_equal(y_pred, np.ones(X.shape[0], dtype=np.int64))
