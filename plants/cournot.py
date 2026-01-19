import jax.numpy as jnp

from plants.base import Plant


class Cournot(Plant):
    def reset(self, cfg):
        p = cfg["plant"]["params"]
        q1 = jnp.asarray(p["q1_0"], dtype=jnp.float32)
        q2 = jnp.asarray(p["q2_0"], dtype=jnp.float32)
        return jnp.array([q1, q2], dtype=jnp.float32)

    def output(self, state, cfg):
        q1, q2 = state[0], state[1]
        p = cfg["plant"]["params"]
        pmax = jnp.asarray(p["pmax"], dtype=jnp.float32)
        cm = jnp.asarray(p["cm"], dtype=jnp.float32)
        q = q1 + q2
        price = pmax - q
        profit = q1 * (price - cm)
        return profit

    def step(self, state, u, d, cfg):
        q1, q2 = state[0], state[1]
        q1_next = jnp.clip(q1 + u, 0.0, 1.0)
        q2_next = jnp.clip(q2 + d, 0.0, 1.0)
        return jnp.array([q1_next, q2_next], dtype=jnp.float32)
