import numpy as np
import pandas as pd


class Selection:

    def __call__(self, results) -> list:
        raise NotImplementedError

class Proportional(Selection):

    def __call__(self, results, pop_size, rng):
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


SELECTION_ALGOS = {
    "proportional": Proportional()
}
