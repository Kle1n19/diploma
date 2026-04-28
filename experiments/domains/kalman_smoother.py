import jax
import jax.numpy as jnp

STATE_DIM = 8


def make_fns(state_dim: int = STATE_DIM, n_steps: int = 200):
    F = jnp.eye(state_dim) * 0.95
    Q = jnp.eye(state_dim) * 0.1

    def step_fn(carry, t_and_data):
        smoothed, (filt_means, filt_covs) = carry, t_and_data
        m_filt = filt_means[smoothed.shape[0] - 1] if False else filt_means
        m_filt, P_filt = t_and_data
        P_pred = F @ P_filt @ F.T + Q
        G = P_filt @ F.T @ jnp.linalg.inv(P_pred)
        m_smooth = m_filt + G @ (carry - F @ m_filt)
        return m_smooth, jnp.linalg.norm(m_smooth)

    def init_carry_fn(key, x0):
        subkeys = jax.random.split(key, n_steps)
        filt_means = jax.vmap(lambda k: jax.random.normal(k, (state_dim,)))(subkeys)
        filt_covs  = jax.vmap(lambda k: jnp.eye(state_dim) + jax.random.normal(k, (state_dim, state_dim)) * 0.01)(subkeys)
        init_smooth = filt_means[-1]
        return init_smooth, filt_means, filt_covs

    def step_fn_flat(carry, _):
        smoothed, filt_means, filt_covs, idx = carry
        m_filt = filt_means[idx]
        P_filt = filt_covs[idx]
        P_pred = F @ P_filt @ F.T + Q
        G = P_filt @ F.T @ jnp.linalg.inv(P_pred)
        m_smooth = m_filt + G @ (smoothed - F @ m_filt)
        new_idx = idx - 1
        return (m_smooth, filt_means, filt_covs, new_idx), jnp.linalg.norm(m_smooth)

    def init_carry_fn_flat(key, x0):
        subkeys = jax.random.split(key, n_steps)
        filt_means = jax.vmap(lambda k: jax.random.normal(k, (state_dim,), dtype=x0.dtype))(subkeys)
        filt_covs = jax.vmap(lambda k: (jnp.eye(state_dim, dtype=x0.dtype) + jax.random.normal(k, (state_dim, state_dim), dtype=x0.dtype) * 0.01))(subkeys)
        init_smooth = filt_means[-1]
        return (init_smooth, filt_means, filt_covs, jnp.array(n_steps - 2, dtype=jnp.int32))

    return step_fn_flat, init_carry_fn_flat
