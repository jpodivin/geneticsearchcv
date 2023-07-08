from scipy.stats import uniform
from sklearn.model_selection._search import ParameterSampler

from .._search import Mutator

param_dists = {
    'foo': uniform(loc=0, scale=4),
    'bar': ['a', 'b', 'c']
}

RANDOM_STATE = 1


def test_init():
    m = Mutator(RANDOM_STATE, param_dists)
    assert m.random_state == RANDOM_STATE
    assert m.param_distributions == param_dists

def test_mutate():
    m = Mutator(RANDOM_STATE, param_dists)
    individual = list(ParameterSampler(
                param_dists, 1, random_state=RANDOM_STATE
            ))[0]

    mutated = m.mutate(individual.copy())

    assert individual.values != mutated.values
