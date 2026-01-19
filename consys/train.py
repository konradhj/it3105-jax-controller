import jax
import jax.numpy as jnp

from controller import get_controller
from plants import get_plant
from consys.simulate import sample_disturbance


def _simulate_epoch_jax(cfg, plant, controller, params, key):
    timesteps = int(cfg["train"]["timesteps"])
    target = jnp.asarray(cfg["plant"]["target"], dtype=jnp.float32)

    plant_state = plant.reset(cfg)
    ctrl_state = controller.init_state(cfg)
    d_traj = sample_disturbance(cfg, key, timesteps)

    def step_fn(carry, d_t):
        p_state, c_state = carry
        y = plant.output(p_state, cfg)
        e = target - y
        u, c_state = controller.step(params, c_state, e, cfg)
        p_state = plant.step(p_state, u, d_t, cfg)
        return (p_state, c_state), (e, u, y)

    (plant_state, ctrl_state), (e_traj, u_traj, y_traj) = jax.lax.scan(
        step_fn, (plant_state, ctrl_state), d_traj
    )
    mse = jnp.mean(e_traj ** 2)
    return mse, {"e": e_traj, "u": u_traj, "y": y_traj, "final_state": plant_state}


def train(cfg):
    epochs = int(cfg["train"]["epochs"])
    lr = jnp.asarray(cfg["train"]["lr"], dtype=jnp.float32)

    plant = get_plant(cfg["plant"]["name"])
    controller = get_controller(cfg["controller"]["name"])

    seed = int(cfg["run"]["seed"])
    key = jax.random.PRNGKey(seed)
    key, key_params = jax.random.split(key)
    params = controller.init_params(cfg, key_params)

    mse_hist = []
    kp_hist = []
    ki_hist = []
    kd_hist = []

    for _ in range(epochs):
        key, subkey = jax.random.split(key)

        def loss_fn(p):
            mse, _ = _simulate_epoch_jax(cfg, plant, controller, p, subkey)
            return mse

        mse = loss_fn(params)
        grads = jax.grad(loss_fn)(params)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)

        mse_hist.append(mse)
        if "kp" in params:
            kp_hist.append(params["kp"])
            ki_hist.append(params["ki"])
            kd_hist.append(params["kd"])

    result = {
        "mse": jnp.stack(mse_hist),
        "params": params,
    }
    if kp_hist:
        result["kp"] = jnp.stack(kp_hist)
        result["ki"] = jnp.stack(ki_hist)
        result["kd"] = jnp.stack(kd_hist)
    return result
