import jax.numpy as jnp

from plants.base import Plant


class LogisticPopulation(Plant):
    def reset(self, cfg):
        p = cfg["plant"]["params"]
        n0 = jnp.asarray(p["N0"], dtype=jnp.float32)
        return n0

    def output(self, state, cfg):
        return state

    def step(self, state, u, d, cfg):
        p = cfg["plant"]["params"]
        r = jnp.asarray(p["r"], dtype=jnp.float32)
        k = jnp.asarray(p["K"], dtype=jnp.float32)
        dt = jnp.asarray(p.get("dt", 1.0), dtype=jnp.float32)

        n = jnp.maximum(state, 0.0)
        growth = r * n * (1.0 - n / k)
        n_next = n + dt * (growth + u + d)
        return jnp.maximum(n_next, 0.0)
