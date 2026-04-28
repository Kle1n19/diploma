"""
Softmax (Boltzmann) bandit agent.
"""

import math
import random

from .base import TuningAgent


class SoftmaxAgent(TuningAgent):
    """
    Parameters
    ----------
    n_actions       : number of configs in the search space
    temperature     : initial Boltzmann temperature
    learning_rate   : step size for Q-value update
    anneal_rate     : fractional temperature decay per step (0 = no annealing)
    min_temperature : lower clamp for temperature annealing
    """

    def __init__(
        self,
        n_actions: int,
        temperature: float = 1.0,
        learning_rate: float = 0.1,
        anneal_rate: float = 0.0,
        min_temperature: float = 0.01
    ):
        self.n_actions = n_actions
        self.temperature = temperature
        self.learning_rate = learning_rate
        self.anneal_rate = anneal_rate
        self.min_temperature = min_temperature
        self._initial_temperature = temperature
        self.q_values = [0.0] * n_actions
        self.counts = [0] * n_actions

    def _probabilities(self) -> list[float]:
        T = max(self.temperature, 1e-9)
        max_q = max(self.q_values)
        exps = [math.exp((q - max_q) / T) for q in self.q_values]
        total = sum(exps)
        return [e / total for e in exps]

    def select_action(self) -> int:
        probs = self._probabilities()
        r = random.random()
        cumulative = 0.0
        for i, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return i
        return self.n_actions - 1

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        q = self.q_values[action]
        self.q_values[action] = q + self.learning_rate * (reward - q)
        if self.anneal_rate > 0:
            self.temperature = max(self.min_temperature, self.temperature * (1.0 - self.anneal_rate))

    def best_action(self) -> int:
        return max(range(self.n_actions), key=lambda i: self.q_values[i])

    def reset(self) -> None:
        self.q_values = [0.0] * self.n_actions
        self.counts = [0] * self.n_actions
        self.temperature = self._initial_temperature
