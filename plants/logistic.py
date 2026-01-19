from plants.base import Plant


class LogisticPopulation(Plant):
    def reset(self, cfg):
        # TODO: read N0 from cfg
        raise NotImplementedError

    def output(self, state, cfg):
        # TODO: return population size
        raise NotImplementedError

    def step(self, state, u, d, cfg):
        # TODO: implement logistic growth + control + disturbance
        raise NotImplementedError
