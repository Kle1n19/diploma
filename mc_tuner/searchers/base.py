"""
Abstract base class for all search strategies.
Add new - subclass Searcher and implement search().
"""

from abc import ABC, abstractmethod

from ..evaluator import Evaluator


class Searcher(ABC):

    @abstractmethod
    def search(self, evaluator: Evaluator, configs: list[dict]) -> tuple[dict, dict, list[dict]]:
        """
        Parameters
        ----------
        evaluato: Evaluator  — knows how to time and score a config
        configs: list[dict] — full set of parameter dicts to consider

        Returns
        -------
        best_params: dict         — config with highest reward found
        best_metrics: dict         — evaluation result for best_params
        history: list[dict]   — every {"params": ..., "metrics": ...} tried
        """
