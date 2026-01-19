import jax
import jax.numpy as jnp

from controller.base import Controller


def _activation(name: str):
    name = name.lower()
    if name == "sigmoid":
        return jax.nn.sigmoid
    if name == "tanh":
        return jnp.tanh
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation '{name}'")


class PIDNN(Controller):
    def init_params(self, cfg, key):
        nn_cfg = cfg["controller"]["nn"]
        layers = nn_cfg["layers"]
        init_min = nn_cfg["init_min"]
        init_max = nn_cfg["init_max"]

        sizes = [3] + layers + [1]
        params = []
        keys = jax.random.split(key, len(sizes) - 1)
        for k, (din, dout) in zip(keys, zip(sizes[:-1], sizes[1:])):
            w_key, b_key = jax.random.split(k)
            W = jax.random.uniform(
                w_key, shape=(din, dout), minval=init_min, maxval=init_max, dtype=jnp.float32
            )
            b = jax.random.uniform(
                b_key, shape=(dout,), minval=init_min, maxval=init_max, dtype=jnp.float32
            )
            params.append({"W": W, "b": b})
        return {"layers": params}

    def init_state(self, cfg):
        return {
            "integral": jnp.asarray(0.0, dtype=jnp.float32),
            "prev_error": jnp.asarray(0.0, dtype=jnp.float32),
        }

    def step(self, params, state, error, cfg):
        integral = state["integral"] + error
        derivative = error - state["prev_error"]
        x = jnp.array([error, integral, derivative], dtype=jnp.float32)

        nn_cfg = cfg["controller"]["nn"]
        activations = nn_cfg["activations"]
        layers = params["layers"]

        for i, layer in enumerate(layers):
            x = jnp.dot(x, layer["W"]) + layer["b"]
            if i < len(layers) - 1:
                act = _activation(activations[i])
                x = act(x)
        u = jnp.squeeze(x)

        new_state = {"integral": integral, "prev_error": error}
        return u, new_state
