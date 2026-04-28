import os
from functools import lru_cache, partial

import jax
import jax.numpy as jnp


def _simulate_raw(
    keys,
    init_states,
    step_fn,
    init_carry_fn,
    n_steps,
    scan_unroll,
    scan_reverse,
    dtype,
    xla_flags: str = ""
):
    init_states = init_states.astype(dtype)

    def single_chain(key, x0):
        carry0 = init_carry_fn(key, x0)
        carry_final, outputs = jax.lax.scan(
            step_fn, carry0, None,
            length=n_steps,
            unroll=scan_unroll,
            reverse=scan_reverse,
        )
        return carry_final, outputs

    carries, outputs = jax.vmap(single_chain)(keys, init_states)
    return carries, outputs


@lru_cache(maxsize=8)
def _get_run_simulation(donate_argnums: tuple):
    """
    Return a JIT-compiled simulation function with the given donate_argnums.
    """
    return jax.jit(
        _simulate_raw,
        static_argnames=["step_fn", "init_carry_fn", "n_steps", "scan_unroll", "scan_reverse", "dtype", "xla_flags"],
        donate_argnums=donate_argnums)


def run_simulation(
    step_fn,
    init_carry_fn,
    keys,
    init_states,
    n_steps,
    scan_unroll,
    scan_reverse,
    dtype,
    xla_flags: str = ""
):
    """
    Public entry point — no buffer donation (default behaviour).
    """
    fn = _get_run_simulation(())
    return fn(
        keys, init_states,
        step_fn=step_fn, init_carry_fn=init_carry_fn,
        n_steps=n_steps, scan_unroll=scan_unroll, scan_reverse=scan_reverse,
        dtype=dtype, xla_flags=xla_flags,
    )
