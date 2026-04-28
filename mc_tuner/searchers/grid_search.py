from ..evaluator import Evaluator
from .base import Searcher


class GridSearcher(Searcher):
    """
    Parameters
    ----------
    verbose: print one line per config evaluated
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose

    def search(self, evaluator: Evaluator, configs: list[dict]) -> tuple[dict, dict, list[dict]]:
        best_params = None
        best_metrics = None
        best_reward = float("-inf")
        history: list[dict] = []
        total = len(configs)

        for i, params in enumerate(configs, 1):
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
                        f"[{i:>4}/{total}] [{flag}]"
                        f"  speedup={metrics['speedup']:.2f}x"
                        f"  reward={metrics['reward']:>8.3f}"
                    )

            except Exception as e:
                if self.verbose:
                    print(f"[{i:>4}/{total}] [ERR]  {e}")

        return best_params, best_metrics, history
