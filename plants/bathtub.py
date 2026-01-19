from plants.base import Plant


g = 9.81


class Bathtub(Plant):
    def reset(self, cfg):
        # TODO: read H0 from cfg
        raise NotImplementedError

    def output(self, state, cfg):
        # TODO: return height
        raise NotImplementedError

    def step(self, state, u, d, cfg):
        # TODO: implement bathtub dynamics
        raise NotImplementedError
