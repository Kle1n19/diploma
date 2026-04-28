import jax
import jax.numpy as jnp

RATE_RIGHT = 0.7
RATE_LEFT = 0.3
P_RIGHT = RATE_RIGHT / (RATE_RIGHT + RATE_LEFT)


def make_fns():
    def step_fn(carry, _):
        pos, key = carry
        key, subkey = jax.random.split(key)
        go_right = jax.random.bernoulli(subkey, p=jnp.array(P_RIGHT, dtype=pos.dtype))
        delta = jnp.where(go_right, jnp.ones_like(pos), -jnp.ones_like(pos))
        new_pos = pos + delta
        return (new_pos, key), new_pos

    def init_carry_fn(key, x0):
        return (jnp.zeros((), dtype=x0.dtype), key)

    return step_fn, init_carry_fn
