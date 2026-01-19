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
    timesteps = int(cfg["train"]["timesteps"])
    target = jnp.asarray(cfg["plant"]["target"], dtype=jnp.float32)

    plant = get_plant(cfg["plant"]["name"])
    controller = get_controller(cfg["controller"]["name"])

    seed = int(cfg["run"]["seed"])
    key = jax.random.PRNGKey(seed)
    key, key_params, key_dist = jax.random.split(key, 3)

    params = controller.init_params(cfg, key_params)
    ctrl_state = controller.init_state(cfg)
    plant_state = plant.reset(cfg)

    # sample all disturbances up front
    D_traj = sample_disturbance(cfg, key_dist, timesteps)

    # logs
    H_traj = []
    U_traj = []
    e_traj = []

    for timestep in range(timesteps):
        y = plant.output(plant_state, cfg)
        e = target - y

        u, ctrl_state = controller.step(params, ctrl_state, e, cfg)
        plant_state = plant.step(plant_state, u, D_traj[timestep], cfg)

        H_traj.append(plant_state)
        U_traj.append(u)
        e_traj.append(e)

    H_traj = jnp.stack(H_traj)
    U_traj = jnp.stack(U_traj)
    e_traj = jnp.stack(e_traj)

    mse = jnp.mean(e_traj ** 2)

    final_y = plant.output(plant_state, cfg)
    return {
        "H": H_traj,
        "U": U_traj,
        "D": D_traj,
        "e": e_traj,
        "mse": mse,
        "final_y": final_y,
        "final_state": plant_state,
    }
