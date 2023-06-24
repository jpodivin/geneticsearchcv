import pytest

from sklearn.utils.estimator_checks import check_estimator

from geneticsearchcv import GeneticSearchCV


@pytest.mark.parametrize(
    "estimator",
    [GeneticSearchCV()]
)
def test_all_estimators(estimator):
    return check_estimator(estimator)
