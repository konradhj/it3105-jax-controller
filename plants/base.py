from abc import ABC, abstractmethod


class Plant(ABC):
    @abstractmethod
    def reset(self, cfg):
        raise NotImplementedError

    @abstractmethod
    def output(self, state, cfg):
        raise NotImplementedError

    @abstractmethod
    def step(self, state, u, d, cfg):
        raise NotImplementedError
