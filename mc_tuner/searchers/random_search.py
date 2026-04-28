import random

from ..evaluator import Evaluator
from .base import Searcher


class RandomSearcher(Searcher):
    """
    Parameters
    ----------
    n: number of configs to sample and evaluate
    seed: random seed for reproducibility (None = non-deterministic)
    verbose: print one line per config evaluated
    """

    def __init__(self, n: int, seed: int | None = None, verbose: bool = True):
        self.n = n
        self.seed = seed
        self.verbose = verbose

    def search(self, evaluator: Evaluator, configs: list[dict]) -> tuple[dict, dict, list[dict]]:
        rng = random.Random(self.seed)
        sample = (
            rng.sample(configs, self.n)
            if self.n <= len(configs)
            else [rng.choice(configs) for _ in range(self.n)]
)

        best_params = None
        best_metrics = None
        best_reward = float("-inf")
        history = []

        for i, params in enumerate(sample, 1):
            try:
                metrics = evaluator.evaluate(params)
                history.append({"params": params, "metrics": metrics})

                if metrics["reward"] > best_reward:
                    best_reward  = metrics["reward"]
                    best_params  = params
                    best_metrics = metrics

                if self.verbose:
                    flag = "OK " if metrics["precision_ok"] else "BAD"
                    print(
                        f"[{i:>3}/{self.n}] [{flag}]"
                        f"  speedup={metrics['speedup']:.2f}x"
                        f"  reward={metrics['reward']:>8.3f}"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"[{i:>3}/{self.n}] [ERR]  {e}")

        return best_params, best_metrics, history
