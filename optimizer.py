"""Genetic search over a permutation of matrix columns."""
import argparse
import json
from itertools import permutations
from pathlib import Path
from random import Random
import time

import numpy as np


def evaluate_solution(X, solution):
    z = X[:, solution[0]] * 100
    for index in solution[1:]:
        z += np.sin(10 * X[:, index] + z)
    return float(np.sum(z ** 2))


def multi_point_crossover(p1, p2, rng):
    start, end = sorted(rng.sample(range(len(p1) + 1), 2))
    child = [None] * len(p1)
    child[start:end] = p1[start:end]
    genes = (gene for gene in p2 if gene not in child)
    for index in range(len(child)):
        if child[index] is None:
            child[index] = next(genes)
    return child


def adaptive_mutate(individual, generation, max_generations, rng, rate=.3):
    individual = individual.copy()
    if rng.random() < min(1., rate * (1 + generation / max_generations)):
        a, b = sorted(rng.sample(range(len(individual)), 2))
        if rng.random() < .5:
            individual[a], individual[b] = individual[b], individual[a]
        else:
            individual[a:b+1] = reversed(individual[a:b+1])
    return individual


def select_parent(population, scores, rng, tournament_size=5):
    candidates = rng.sample(range(len(population)), min(tournament_size, len(population)))
    return population[min(candidates, key=lambda i: scores[i])]


def local_search(X, solution, score, rng, max_iter=20, deadline=float('inf')):
    best_solution = solution.copy()
    for _ in range(max_iter):
        if time.monotonic() >= deadline:
            break
        a, b = rng.sample(range(len(solution)), 2)
        candidate = best_solution.copy()
        candidate[a], candidate[b] = candidate[b], candidate[a]
        candidate_score = evaluate_solution(X, candidate)
        if candidate_score < score:
            best_solution, score = candidate, candidate_score
    return best_solution, score


def run_optimization(X, seed=17, population_size=40, generations=100, seconds=5.):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] < 2 or not np.isfinite(X).all():
        raise ValueError('Expected a finite nonempty matrix with at least two columns')
    if population_size < 2 or generations < 1 or seconds <= 0 or not np.isfinite(seconds):
        raise ValueError('Expected population >= 2 and positive generations and time limit')
    rng = Random(seed)
    deadline = time.monotonic() + seconds
    population = [rng.sample(range(X.shape[1]), X.shape[1]) for _ in range(population_size)]
    best_solution, best_score = population[0].copy(), evaluate_solution(X, population[0])
    history = []
    for generation in range(generations):
        if generation and time.monotonic() >= deadline:
            break
        scores = [evaluate_solution(X, individual) for individual in population]
        index = min(range(len(scores)), key=scores.__getitem__)
        if scores[index] < best_score:
            best_solution, best_score = population[index].copy(), scores[index]
        best_solution, best_score = local_search(X, best_solution, best_score, rng, deadline=deadline)
        history.append(best_score)
        if time.monotonic() >= deadline:
            break
        elite_count = max(1, population_size // 5)
        elite = sorted(range(len(scores)), key=scores.__getitem__)[:elite_count]
        new_population = [best_solution.copy()] + [population[i].copy() for i in elite]
        while len(new_population) < population_size:
            p1 = select_parent(population, scores, rng)
            p2 = select_parent(population, scores, rng)
            child = multi_point_crossover(p1, p2, rng)
            new_population.append(adaptive_mutate(child, generation, generations, rng))
        population = new_population[:population_size]
    return {'permutation': best_solution, 'score': best_score, 'history': history, 'seed': seed}


def exact_solution(X):
    if X.shape[1] > 8:
        raise ValueError('Exact enumeration is limited to eight columns')
    solution = min(permutations(range(X.shape[1])), key=lambda p: evaluate_solution(X, p))
    return list(solution), evaluate_solution(X, solution)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, help='Numeric CSV without a header; default: synthetic example')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--generations', type=int, default=100)
    parser.add_argument('--population', type=int, default=40)
    parser.add_argument('--seconds', type=float, default=5.)
    parser.add_argument('--output', type=Path, help='Optional NEW result file')
    args = parser.parse_args()
    X = np.loadtxt(args.data, delimiter=',', ndmin=2) if args.data else np.random.default_rng(args.seed).normal(0, .01, (12, 6))
    result = run_optimization(X, args.seed, args.population, args.generations, args.seconds)
    text = json.dumps(result, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(text)
    print(text)


if __name__ == '__main__':
    main()
