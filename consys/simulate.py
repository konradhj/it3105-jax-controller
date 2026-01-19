import jax
import jax.numpy as jnp

from plants import get_plant


def sample_disturbance(cfg, key, timesteps):
    d_cfg = cfg["disturbance"]
    d_min = jnp.asarray(d_cfg["d_min"], dtype=jnp.float32)
    d_max = jnp.asarray(d_cfg["d_max"], dtype=jnp.float32)
    return jax.random.uniform(key, shape=(timesteps,), minval=d_min, maxval=d_max, dtype=jnp.float32)


def simulate(cfg):
    timesteps = int(cfg['train']['timesteps'])
    target = jnp.asarray(cfg["plant"]["target"], dtype=jnp.float32)
    
    plant_name = cfg["plant"]["name"]
    plant_reset, plant_step = get_plant(plant_name)

    seed = int(cfg["run"]["seed"])
    key = jax.random.PRNGKey(seed)
    
    state = plant_reset(cfg)

    # sample all disturbances up front
    key, subkey = jax.random.split(key)
    D_traj = sample_disturbance(cfg, subkey, timesteps)

    # logs
    H_traj = []
    U_traj = []
    e_traj = []

    for timestep in range(timesteps):
        e = target - state

        # no controller yet
        U = jnp.asarray(0.0, dtype=jnp.float32)

        state = plant_step(state, U, D_traj[timestep], cfg)

        H_traj.append(state)
        U_traj.append(U)
        e_traj.append(e)

    H_traj = jnp.stack(H_traj)
    U_traj = jnp.stack(U_traj)
    e_traj = jnp.stack(e_traj)

    mse = jnp.mean(e_traj ** 2)

    return {"H": H_traj, "U": U_traj, "D": D_traj, "e": e_traj, "mse": mse, "final_H": state}
