from abc import ABC, abstractmethod


class Controller(ABC):
    @abstractmethod
    def init_params(self, cfg, key):
        raise NotImplementedError

    @abstractmethod
    def init_state(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def step(self, params, state, error, cfg):
        raise NotImplementedError
