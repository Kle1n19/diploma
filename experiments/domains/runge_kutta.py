import jax
import jax.numpy as jnp
import numpy as np

DIM = 32
DT  = 0.01


def _make_stable_matrix(dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((dim, dim)).astype(np.float32)
    A = (A - A.T) / 2.0
    A -= np.eye(dim, dtype=np.float32) * 0.1
    return A


def make_fns(dim: int = DIM, dt: float = DT):
    A = jnp.array(_make_stable_matrix(dim))

    def step_fn(carry, _):
        y, t = carry
        k1 = A @ y
        k2 = A @ (y + dt / 2 * k1)
        k3 = A @ (y + dt / 2 * k2)
        k4 = A @ (y + dt * k3)
        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t_new = t + dt
        return (y_new, t_new), jnp.linalg.norm(y_new)

    def init_carry_fn(key, x0):
        y0 = jnp.ones(dim, dtype=x0.dtype)
        t0 = jnp.zeros((), dtype=x0.dtype)
        return (y0, t0)

    return step_fn, init_carry_fn
