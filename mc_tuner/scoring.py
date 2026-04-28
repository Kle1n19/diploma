import math
import jax
import jax.numpy as jnp


PRECISION_PENALTY = -100.0

PRECISION_MODES = ("absolute", "relative", "zscore", "wasserstein", "quantile", "soft")


def default_metric(outputs) -> float:
    """Mean of all output values across all pytree leaves (float32-safe)."""
    leaves = jax.tree_util.tree_leaves(outputs)
    flat = jnp.concatenate([jnp.ravel(leaf.astype(jnp.float32)) for leaf in leaves])
    return float(jnp.mean(flat))


def default_std(outputs) -> float:
    """Std of all output values across all pytree leaves (float32-safe)."""
    leaves = jax.tree_util.tree_leaves(outputs)
    flat = jnp.concatenate([jnp.ravel(leaf.astype(jnp.float32)) for leaf in leaves])
    return float(jnp.std(flat))


def _flat_f32(outputs) -> jnp.ndarray:
    leaves = jax.tree_util.tree_leaves(outputs)
    return jnp.concatenate([jnp.ravel(leaf.astype(jnp.float32)) for leaf in leaves])


#Drift
def _drift_absolute(candidate_metric, baseline_metric) -> float:
    return abs(candidate_metric - baseline_metric)


def _drift_relative(candidate_metric, baseline_metric) -> float:
    denom = abs(baseline_metric) if baseline_metric != 0.0 else 1.0
    return abs(candidate_metric - baseline_metric) / denom


def _drift_zscore(candidate_metric, baseline_metric, baseline_std, batch_size) -> float:
    se = (baseline_std / math.sqrt(max(batch_size, 1))) or 1e-9
    return abs(candidate_metric - baseline_metric) / se


def _drift_wasserstein(candidate_outputs, baseline_outputs) -> float:
    a = jnp.sort(_flat_f32(baseline_outputs))
    b = jnp.sort(_flat_f32(candidate_outputs))
    return float(jnp.mean(jnp.abs(a - b)))


def _drift_quantile(
    candidate_outputs,
    baseline_outputs,
    quantiles = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)
) -> float:
    q = jnp.array(quantiles)
    bq = jnp.quantile(_flat_f32(baseline_outputs),  q)
    cq = jnp.quantile(_flat_f32(candidate_outputs), q)
    denom = jnp.abs(bq) + 1e-9
    return float(jnp.max(jnp.abs(cq - bq) / denom))

def compute_drift(
    mode,
    candidate_metric,
    baseline_metric,
    baseline_std = 0.0,
    batch_size = 1,
    candidate_outputs = None,
    baseline_outputs = None
) -> float:
    if mode in ("absolute", "soft"):
        return _drift_absolute(candidate_metric, baseline_metric)
    if mode == "relative":
        return _drift_relative(candidate_metric, baseline_metric)
    if mode == "zscore":
        return _drift_zscore(candidate_metric, baseline_metric, baseline_std, batch_size)
    if mode == "wasserstein":
        if candidate_outputs is None or baseline_outputs is None:
            return _drift_absolute(candidate_metric, baseline_metric)
        return _drift_wasserstein(candidate_outputs, baseline_outputs)
    if mode == "quantile":
        if candidate_outputs is None or baseline_outputs is None:
            return _drift_absolute(candidate_metric, baseline_metric)
        return _drift_quantile(candidate_outputs, baseline_outputs)
    raise ValueError(f"Unknown precision_mode '{mode}'. Choose from: {PRECISION_MODES}")


def precision_ok(drift, threshold, mode = "absolute") -> bool:
    if mode == "soft":
        return True
    return drift <= threshold

def reward(speedup, drift, threshold, mode = "absolute") -> float:
    if mode == "soft":
        gate = math.exp(-((drift / threshold) ** 2)) if threshold > 0 else float(drift == 0)
        return speedup * gate
    if drift > threshold:
        return PRECISION_PENALTY + speedup
    return speedup

def compute_speedup( candidate_time, candidate_work, baseline_time, baseline_work) -> float:
    return (candidate_work / candidate_time) / (baseline_work / baseline_time)
