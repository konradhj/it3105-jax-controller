from controller.base import Controller


class PIDClassic(Controller):
    def init_params(self, cfg, key):
        # TODO: initialize kp, ki, kd from cfg
        raise NotImplementedError

    def init_state(self, cfg):
        # TODO: initialize integral and prev_error
        raise NotImplementedError

    def step(self, params, state, error, cfg):
        # TODO: compute U and update error history
        raise NotImplementedError
