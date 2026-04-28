"""
RL-guided search using any TuningAgent.
"""

from ..agents.base import TuningAgent
from ..evaluator import Evaluator
from .base import Searcher


class RLSearcher(Searcher):
    """
    Parameters
    ----------
    agent: any TuningAgent implementation
    episodes: number of evaluate-and-update cycles
    verbose: print one line per episode
    """

    def __init__(self, agent: TuningAgent, episodes: int = 1000, verbose: bool = True):
        self.agent = agent
        self.episodes = episodes
        self.verbose = verbose

    def search(self, evaluator: Evaluator, configs: list[dict]) -> tuple[dict, dict, list[dict]]:
        best_params = None
        best_metrics = None
        best_reward  = float("-inf")
        history: list[dict] = []

        for episode in range(1, self.episodes + 1):
            idx = self.agent.select_action()
            params = configs[idx]
            metrics = evaluator.evaluate(params)
            self.agent.update(idx, metrics["reward"])
            history.append({"params": params, "metrics": metrics})

            if metrics["reward"] > best_reward:
                best_reward = metrics["reward"]
                best_params = params
                best_metrics = metrics

            if self.verbose:
                flag = "OK " if metrics["precision_ok"] else "BAD"
                print(
                    f"[ep {episode:>4}/{self.episodes}] [{flag}]"
                    f"  speedup={metrics['speedup']:.2f}x"
                    f"  reward={metrics['reward']:>8.3f}"
                    f"  {_fmt_params(params)}"
                )

        return best_params, best_metrics, history


def _fmt_params(p: dict) -> str:
    parts = []
    for k, v in p.items():
        if hasattr(v, "dtype"):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return "  ".join(parts)
