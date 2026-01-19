import jax.numpy as jnp

g = 9.81
# Velocity of water exiting through drain V = sqrt(2*g*H) H is height

def reset(cfg):
    H0 = cfg['plant']['params']['H0']
    return jnp.asarray(H0, dtype=jnp.float32)


def step(H, U, D, cfg):
    """
    One timestep of the bathtub dynamics.

    Args:
        H: current water height
        U: controller inflow (faucet) this step
        D: disturbance/noise inflow (can be negative) this step
        cfg: config dict containing plant parameters

    Returns:
        H_next: next height
    """

    p = cfg["plant"]["params"]
    A = jnp.asarray(p["A"], dtype=jnp.float32)
    C = jnp.asarray(p["C"], dtype=jnp.float32)
    dt = jnp.asarray(p.get("dt", 1.0), dtype=jnp.float32)

    # Prevent negative height (and sqrt issues)
    H = jnp.maximum(H, 0.0)

    V = jnp.sqrt(2.0 * g * H)   # drain speed
    Q = V * C                   # outflow rate

    dH = (U + D - Q) / A        # from the equation in the PDF
    H_next = jnp.maximum(H + dt * dH, 0.0)

    return H_next
