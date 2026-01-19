import jax.numpy as jnp

from plants.base import Plant

g = 9.81


class Bathtub(Plant):
    def reset(self, cfg):
        h0 = cfg["plant"]["params"]["H0"]
        return jnp.asarray(h0, dtype=jnp.float32)

    def output(self, state, cfg):
        return state

    def step(self, state, u, d, cfg):
        p = cfg["plant"]["params"]
        a = jnp.asarray(p["A"], dtype=jnp.float32)
        c = jnp.asarray(p["C"], dtype=jnp.float32)
        dt = jnp.asarray(p.get("dt", 1.0), dtype=jnp.float32)

        h = jnp.maximum(state, 0.0)
        h_safe = jnp.maximum(h, 1e-6)
        v = jnp.sqrt(2.0 * g * h_safe)
        q = v * c

        dh = (u + d - q) / a
        h_next = jnp.maximum(h + dt * dh, 0.0)
        return h_next
