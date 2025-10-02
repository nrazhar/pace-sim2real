from __future__ import annotations

import cmaes


class CMAESOptimizer:
    def __init__(self, initial_params, sigma, population_size):
        self.initial_params = initial_params
        self.sigma = sigma
        self.population_size = population_size
        self.optimizer = None

    def ask(self):
        return self.optimizer.ask()

    def tell(self, solutions, fitnesses):
        for solution, fitness in zip(solutions, fitnesses):
            self.optimizer.tell(solution, fitness)

    def best_solution(self):
        return self.optimizer.best_solution
