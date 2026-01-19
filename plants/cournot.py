from plants.base import Plant


class Cournot(Plant):
    def reset(self, cfg):
        # TODO: initialize q1, q2
        raise NotImplementedError

    def output(self, state, cfg):
        # TODO: compute profit for producer 1
        raise NotImplementedError

    def step(self, state, u, d, cfg):
        # TODO: update q1, q2 and clamp to [0,1]
        raise NotImplementedError
