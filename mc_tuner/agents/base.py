from abc import ABC, abstractmethod


class TuningAgent(ABC):

    @abstractmethod
    def select_action(self) -> int:
        """Return the index of the next configuration to evaluate"""

    @abstractmethod
    def update(self, action: int, reward: float) -> None:
        """Update internal state after observing reward"""

    @abstractmethod
    def best_action(self) -> int:
        """Return the index of the currently estimated best configuration"""

    @abstractmethod
    def reset(self) -> None:
        """Reset to initial state"""
