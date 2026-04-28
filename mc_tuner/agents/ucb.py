"""
UCB1 (Upper Confidence Bound) agent. https://en.wikipedia.org/wiki/Upper_Confidence_Bound
"""

import math
from .base import TuningAgent


class UCB1Agent(TuningAgent):
    """
    Parameters
    ----------
    n_actions : number of configs in the search space
    c: exploration constant (default sqrt(2) ≈ 1.414)
    """

    def __init__(self, n_actions: int, c: float = 1.414):
        self.n_actions = n_actions
        self.c = c
        self.q_values = [0.0] * n_actions
        self.counts = [0] * n_actions
        self.t = 0

    def _ucb(self, i: int) -> float:
        if self.counts[i] == 0:
            return float("inf")
        return self.q_values[i] + self.c * math.sqrt(math.log(self.t) / self.counts[i])

    def select_action(self) -> int:
        self.t += 1
        return max(range(self.n_actions), key=self._ucb)

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        n = self.counts[action]
        self.q_values[action] += (reward - self.q_values[action]) / n

    def best_action(self) -> int:
        return max(range(self.n_actions), key=lambda i: self.q_values[i])

    def reset(self) -> None:
        self.q_values = [0.0] * self.n_actions
        self.counts = [0] * self.n_actions
        self.t = 0
