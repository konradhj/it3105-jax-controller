import jax
import jax.numpy as jnp

from controller import get_controller
from plants import get_plant


def sample_disturbance(cfg, key, timesteps):
    d_cfg = cfg["disturbance"]
    d_min = jnp.asarray(d_cfg["d_min"], dtype=jnp.float32)
    d_max = jnp.asarray(d_cfg["d_max"], dtype=jnp.float32)
    return jax.random.uniform(key, shape=(timesteps,), minval=d_min, maxval=d_max, dtype=jnp.float32)


def simulate(cfg):
    # TODO: run one epoch (timesteps) and return trajectories + MSE
    raise NotImplementedError
