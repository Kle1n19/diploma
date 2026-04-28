import random

from .base import TuningAgent


class EpsilonGreedyAgent(TuningAgent):
    def __init__(self, n_actions: int, epsilon: float = 0.2, learning_rate: float = 0.1):
        self.n_actions = n_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.q_values = [0.0] * n_actions
        self.counts = [0] * n_actions

    def select_action(self) -> int:
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        return self.best_action()

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        q = self.q_values[action]
        self.q_values[action] = q + self.learning_rate * (reward - q)

    def best_action(self) -> int:
        return max(range(self.n_actions), key=lambda i: self.q_values[i])

    def reset(self) -> None:
        self.q_values = [0.0] * self.n_actions
        self.counts = [0] * self.n_actions
