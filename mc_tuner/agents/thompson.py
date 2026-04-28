"""
Thompson Sampling agent with Normal-Normal conjugate model.
"""

import math
import random
from .base import TuningAgent


class ThompsonSamplingAgent(TuningAgent):
    """
    Parameters
    ----------
    n_actions  : number of configs in the search space
    prior_mean : prior mean for all action reward distributions
    prior_std  : prior standard deviation (uncertainty before any data)
    noise_std  : assumed reward noise standard deviation
    """

    def __init__(
        self,
        n_actions: int,
        prior_mean: float = 0.0,
        prior_std: float = 1.0,
        noise_std: float = 1.0,
    ):
        self.n_actions = n_actions
        self.prior_mean = prior_mean
        self.prior_var = prior_std ** 2
        self.noise_var = noise_std ** 2

        self._mu = [prior_mean] * n_actions
        self._var = [prior_std**2]  * n_actions
        self._reward_sums = [0.0] * n_actions
        self.counts = [0] * n_actions

    def _sample(self, i: int) -> float:
        u1 = random.random() + 1e-12
        u2 = random.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        return self._mu[i] + math.sqrt(self._var[i]) * z

    def select_action(self) -> int:
        return max(range(self.n_actions), key=self._sample)

    def update(self, action: int, reward: float) -> None:
        self.counts[action] += 1
        self._reward_sums[action] += reward
        n = self.counts[action]

        precision_prior = 1.0 / self.prior_var
        precision_likelihood = n / self.noise_var
        posterior_var = 1.0 / (precision_prior + precision_likelihood)
        posterior_mean = posterior_var * (self.prior_mean / self.prior_var + self._reward_sums[action] / self.noise_var)
        self._mu[action] = posterior_mean
        self._var[action] = posterior_var

    def best_action(self) -> int:
        return max(range(self.n_actions), key=lambda i: self._mu[i])

    def reset(self) -> None:
        self._mu = [self.prior_mean] * self.n_actions
        self._var = [self.prior_var] * self.n_actions
        self._reward_sums = [0.0] * self.n_actions
        self.counts = [0] * self.n_actions
