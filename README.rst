.. -*- mode: rst -*-

|ReadTheDocs|_

.. |ReadTheDocs| image:: https://readthedocs.org/projects/geneticsearchcv/badge/?version=latest
.. _ReadTheDocs: https://geneticsearchcv.readthedocs.io/en/latest/?badge=latest

gasearch - Genetic algorithm based hyperparameter tuner
==============================================================

.. _scikit-learn: https://scikit-learn.org

**gasearch** - Finding hyperparameters the way nature intended

Hyperparameter search isn't an exact science. We've all heard that one.

gasearch searches the parameter space using genetic algorithm,
with multiple solutions in each generation competing for inclusion in the next.

Mutation of the existing solutions introduces new characteristics and crossover
helps the beneficial traits spread across the population.
At the same time, proportional selection prevents early convergence of algorithm on local
minimum, keeping the population diverse and dynamic.

However, since genetic algorithms are heuristic, there can not be any guarantee that
the solution delivered is the optimal one for a given problem.
Regardless of the number of iterations.

The package follows scikit-learn API conventions and can be readily integrated with
existing pipelines.

TODO:
    * Implement alternative selection algorithms
    * Publish more examples
    * Implement alternative crossover operations
    * Improve docs
    * Optimize, optimize, optimize ...
