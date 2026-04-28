"""
FnEvaluator: benchmarks any complete JAX function without decomposing it.
Unlike Evaluator (which expects step_fn + init_carry_fn for scan), this
wraps a full callable with jax.vmap for batching and times it directly.
"""

import time
from functools import partial
import jax
import jax.numpy as jnp

from .scoring import (default_metric, default_std, compute_drift, precision_ok, reward as compute_reward, PRECISION_MODES)


def _unwrap_jit(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn

@partial(jax.jit, static_argnames=["_fn"])
def _vmapped_call(_fn, inputs):
    return jax.vmap(_fn)(inputs)


class FnEvaluator:
    """
    Parameters
    ----------
    fn               : the JAX function to benchmark, signature fn(x, **kwargs)
    generate_inputs  : (master_key, batch_size) -> jax array of per-sample inputs
    fixed_kwargs     : kwargs always forwarded to fn unchanged (e.g. K, T, r, sigma)
    param_map        : maps tuner param names to fn kwarg names
                       e.g. {"n_steps": "steps"} — default assumes fn accepts `steps`
    baseline_params  : reference config for speedup and precision comparison
    master_key       : JAX PRNGKey
    precision      : max allowed deviation — meaning depends on precision_mode
    precision_mode : one of "absolute" | "relative" | "zscore" |
                     "wasserstein" | "quantile" | "soft"
    n_repeats      : timing repeats (min wall time is reported)
    """

    def __init__(
        self,
        fn,
        generate_inputs,
        fixed_kwargs: dict,
        param_map: dict,
        baseline_params: dict,
        master_key,
        precision: float = 0.001,
        precision_mode: str = "absolute",
        n_repeats: int = 3,
    ):
        if precision_mode not in PRECISION_MODES:
            raise ValueError(f"Unknown precision_mode '{precision_mode}'. Choose from: {PRECISION_MODES}")
        self.fn = fn
        self.generate_inputs = generate_inputs
        self.fixed_kwargs = fixed_kwargs
        self.param_map = param_map
        self.baseline_params = baseline_params
        self.master_key = master_key
        self.precision = precision
        self.precision_mode = precision_mode
        self.n_repeats = n_repeats
        self._baseline: dict | None = None

    def _resolve_fn(self, params: dict):
        """Bind current tuner params to fn kwargs via param_map."""
        mapped = {fn_key: params[tuner_key]
                  for tuner_key, fn_key in self.param_map.items()
                  if tuner_key in params}
        raw_fn = _unwrap_jit(self.fn)
        return partial(raw_fn, **self.fixed_kwargs, **mapped)

    def _measure(self, params: dict) -> tuple[float, float, any]:
        fn_bound = self._resolve_fn(params)
        dtype = params.get("dtype", jnp.float32)
        inputs = self.generate_inputs(self.master_key, params["batch_size"]).astype(dtype)

        outputs = _vmapped_call(fn_bound, inputs)
        jax.block_until_ready(outputs)
        metric = default_metric(outputs)

        times = []
        for _ in range(self.n_repeats):
            t0 = time.perf_counter()
            outputs = _vmapped_call(fn_bound, inputs)
            jax.block_until_ready(outputs)
            times.append(time.perf_counter() - t0)
        return min(times), metric, outputs

    def warm_up(self) -> None:
        if self._baseline is not None:
            return
        t, metric, outputs = self._measure(self.baseline_params)
        self._baseline = {"time": t, "metric": metric, "std": default_std(outputs), "outputs": outputs}

    def evaluate(self, params: dict) -> dict:
        self.warm_up()
        t, metric, outputs = self._measure(params)
        bl = self._baseline
        speedup = bl["time"] / t
        drift = compute_drift(
            mode = self.precision_mode,
            candidate_metric = metric,
            baseline_metric = bl["metric"],
            baseline_std = bl["std"],
            batch_size = params.get("batch_size", 1),
            candidate_outputs = outputs,
            baseline_outputs  = bl["outputs"])
        ok = precision_ok(drift, self.precision, self.precision_mode)
        rew = compute_reward(speedup, drift, self.precision, self.precision_mode)
        return {"time": t, "metric": metric, "drift": drift, "speedup": speedup, "precision_ok": ok, "reward": rew}

    @property
    def baseline(self) -> dict:
        self.warm_up()
        return self._baseline
