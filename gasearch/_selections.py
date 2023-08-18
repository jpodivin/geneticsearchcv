import numpy as np
import pandas as pd

def proportional(results : dict, pop_size : int, rng : np.random.RandomState, **args) -> list:
    """Use proportional selection to ensure presence of high fitness individuals."""
    results = pd.DataFrame(results).sort_values(by='mean_test_score')
    fitness = 1/results['mean_test_score'].to_numpy()
    # Normalizing to [0,1]
    if fitness.max() != fitness.min():
        fitness = (fitness - fitness.min()) / (fitness.max() - fitness.min())
    else:
        fitness = np.full_like(fitness, 1e-15)
    probs = fitness/np.sum(fitness)
    new_population = rng.choice(results['params'], size=pop_size, p=probs)
    return new_population

def tournament(results : dict, pop_size : int, rng : np.random.RandomState, **args) -> list:
    """Use tournament selection to ensure presence of high fitness individuals"""
    new_population = []
    results = pd.DataFrame(results)
    while len(new_population) < pop_size:
        new_population.append(
            results.sample(frac=0.1).sort_values(by="mean_test_score")['params'].iloc[0])
    return new_population

SELECTION_ALGOS = {
    "proportional": proportional,
    "tournament": tournament
}
