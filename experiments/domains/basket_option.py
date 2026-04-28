import jax
import jax.numpy as jnp
import numpy as np

S0_EACH = 100.0
r       = 0.05
sigma   = 0.20
T       = 1.0

_CORR = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]], dtype=np.float32)
_CHOL = np.linalg.cholesky(_CORR).astype(np.float32)


def make_fns(n_steps: int = 252):
    dt        = T / n_steps
    log_drift = float((r - 0.5 * sigma**2) * dt)
    vol_sqrt  = float(sigma * np.sqrt(dt))
    L         = jnp.array(_CHOL)

    def step_fn(carry, _):
        prices, key = carry
        key, subkey = jax.random.split(key)
        Z       = jax.random.normal(subkey, shape=(3,), dtype=prices.dtype)
        corr_Z  = L.astype(prices.dtype) @ Z
        prices_new = prices * jnp.exp(
            jnp.array(log_drift, dtype=prices.dtype)
            + jnp.array(vol_sqrt, dtype=prices.dtype) * corr_Z
        )
        return (prices_new, key), jnp.mean(prices_new)

    def init_carry_fn(key, x0):
        prices = jnp.full((3,), S0_EACH, dtype=x0.dtype)
        return (prices, key)

    return step_fn, init_carry_fn
