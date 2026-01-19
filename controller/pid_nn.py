from controller.base import Controller


class PIDNN(Controller):
    def init_params(self, cfg, key):
        # TODO: initialize NN weights/biases from cfg
        raise NotImplementedError

    def init_state(self, cfg):
        # TODO: initialize integral and prev_error
        raise NotImplementedError

    def step(self, params, state, error, cfg):
        # TODO: compute U via NN and update error history
        raise NotImplementedError
