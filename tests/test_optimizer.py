from random import Random
import numpy as np

from optimizer import adaptive_mutate, evaluate_solution, exact_solution, multi_point_crossover, run_optimization


def test_genetic_operators_preserve_permutation():
    rng = Random(2)
    p1, p2 = [0, 1, 2, 3, 4], [4, 2, 0, 3, 1]
    for _ in range(40):
        child = multi_point_crossover(p1, p2, rng)
        assert sorted(child) == list(range(5))
        assert sorted(adaptive_mutate(child, 9, 10, rng, 1)) == list(range(5))
    assert p1 == [0, 1, 2, 3, 4]


def test_bounded_search_matches_objective_and_exact_lower_bound():
    X = np.random.default_rng(2).normal(0, .01, (8, 4))
    result = run_optimization(X, seed=2, population_size=10, generations=8, seconds=3)
    assert result['score'] == evaluate_solution(X, result['permutation'])
    assert all(b <= a for a, b in zip(result['history'], result['history'][1:]))
    _, exact = exact_solution(X)
    assert result['score'] >= exact - 1e-12
    repeated = run_optimization(X, seed=2, population_size=10, generations=8, seconds=3)
    assert result == repeated
