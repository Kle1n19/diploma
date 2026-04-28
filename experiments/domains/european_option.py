import jax
import jax.numpy as jnp

S0 = 100.0
K = 100.0
r = 0.05
sigma = 0.20
T = 1.0


def make_fns(n_steps: int = 252):
    dt = T / n_steps
    log_drift = (r - 0.5 * sigma**2) * dt
    vol_sqrt = sigma * jnp.sqrt(dt)

    def step_fn(carry, _):
        S, key = carry
        key, subkey = jax.random.split(key)
        Z = jax.random.normal(subkey, dtype=S.dtype)
        S_new = S * jnp.exp(jnp.array(log_drift, dtype=S.dtype) + jnp.array(vol_sqrt,  dtype=S.dtype) * Z)
        return (S_new, key), S_new

    def init_carry_fn(key, x0):
        return (jnp.full((), S0, dtype=x0.dtype), key)

    return step_fn, init_carry_fn
