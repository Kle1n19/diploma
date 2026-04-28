"""Params cinfig helper"""
import json
import os
import jax
import jax.numpy as jnp


_DTYPE_MAP: dict = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}

_POLICY_MAP: dict = {
    "nothing_saveable": jax.checkpoint_policies.nothing_saveable,
    "everything_saveable": jax.checkpoint_policies.everything_saveable,
    "dots_with_no_batch_dims_saveable": jax.checkpoint_policies.dots_with_no_batch_dims_saveable
}


def dtype_to_str(dtype) -> str:
    for name, dt in _DTYPE_MAP.items():
        if dtype == dt:
            return name
    return str(dtype)


def dtype_from_str(name):
    if name not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype '{name}'. Known: {list(_DTYPE_MAP)}")
    return _DTYPE_MAP[name]


def policy_to_str(policy) -> str | None:
    if policy is None:
        return None
    for name, pol in _POLICY_MAP.items():
        if policy is pol:
            return name
    return str(policy)


def policy_from_str(name):
    if name is None:
        return None
    if name not in _POLICY_MAP:
        raise ValueError(f"Unknown checkpoint policy '{name}'. Known: {list(_POLICY_MAP)}")
    return _POLICY_MAP[name]


def _serialise_value(key, value):
    if key == "dtype":
        return dtype_to_str(value)
    if key == "checkpoint_policy":
        return policy_to_str(value)
    if key == "jit_donate_argnums":
        return list(value)
    return value


def _deserialise_value(key, value):
    if key == "dtype":
        return dtype_from_str(value)
    if key == "checkpoint_policy":
        return policy_from_str(value)
    if key == "jit_donate_argnums":
        return tuple(value)
    return value


def save_config(params, path) -> None:
    serial = {k: _serialise_value(k, v) for k, v in params.items()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(serial, f, indent=2)


def load_config(path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {k: _deserialise_value(k, v) for k, v in raw.items()}
