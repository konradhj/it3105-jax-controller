from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def reset(self, cfg):
        """Return initial plant state."""
        raise NotImplementedError

    @abstractmethod
    def output(self, state, cfg):
        """Return plant output Y from state."""
        raise NotImplementedError

    @abstractmethod
    def step(self, state, u, d, cfg):
        """Return next plant state given control and disturbance."""
        raise NotImplementedError
