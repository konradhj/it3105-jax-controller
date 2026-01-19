import jax.numpy as jnp

from controller.base import Controller


class PIDClassic(Controller):
    def init_params(self, cfg, key):
        init = cfg["controller"]["init"]
        return {
            "kp": jnp.asarray(init["kp"], dtype=jnp.float32),
            "ki": jnp.asarray(init["ki"], dtype=jnp.float32),
            "kd": jnp.asarray(init["kd"], dtype=jnp.float32),
        }

    def init_state(self, cfg):
        return {
            "integral": jnp.asarray(0.0, dtype=jnp.float32),
            "prev_error": jnp.asarray(0.0, dtype=jnp.float32),
        }

    def step(self, params, state, error, cfg):
        integral = state["integral"] + error
        derivative = error - state["prev_error"]
        u = params["kp"] * error + params["ki"] * integral + params["kd"] * derivative
        new_state = {"integral": integral, "prev_error": error}
        return u, new_state
