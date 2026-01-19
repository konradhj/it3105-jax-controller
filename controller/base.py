from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def init_params(self, cfg, key):
        """Return initial controller parameters."""
        raise NotImplementedError

    @abstractmethod
    def init_state(self, cfg):
        """Return initial controller state (e.g., error history)."""
        raise NotImplementedError

    @abstractmethod
    def step(self, params, state, error, cfg):
        """Compute control output and next controller state."""
        raise NotImplementedError
