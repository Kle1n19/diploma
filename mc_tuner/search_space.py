"""
JAX injection-level search space and parameter helpers.
"""

import itertools
import jax
import jax.numpy as jnp


JAX_INJECT_SPACE: dict = {
    "scan_unroll":        [1, 2, 4, 8, 16],
    "scan_reverse":       [False, True],
    "jit_donate_argnums": [(), (0,), (1,), (0,1)],
    "dot_precision":      [None, "high", "highest"],
    "matmul_precision":   [None, "high", "highest"],
    "autotune_level":     [0, 1, 2, 3, 4],
    "map_chunk_size":     [1, 4, 8, 16, None],
    "checkpoint_policy":  [
        None,
        jax.checkpoint_policies.nothing_saveable,
        jax.checkpoint_policies.everything_saveable,
        jax.checkpoint_policies.dots_with_no_batch_dims_saveable,
    ],
}

JAX_INJECT_DEFAULTS: dict = {
    "scan_unroll":        1,
    "scan_reverse":       False,
    "jit_donate_argnums": (),
    "dot_precision":      None,
    "matmul_precision":   None,
    "autotune_level":     0,
    "map_chunk_size":     None,
    "checkpoint_policy":  None,
}


def all_combinations(search_space: dict) -> list[dict]:
    """Cartesian product of all parameter lists → list of config dicts."""
    keys = list(search_space.keys())
    values = list(search_space.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def make_inputs(master_key, batch_size: int, state_shape: tuple = (2,)):
    """Split master_key into per-chain keys and sample Gaussian init states."""
    keys = jax.random.split(master_key, batch_size)
    init_states = jax.random.normal(master_key, shape=(batch_size, *state_shape))
    return keys, init_states
