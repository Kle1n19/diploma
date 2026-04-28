"""
Evaluator: wraps run_simulation with timing, baseline management, and precision checks.
"""

import os
import time

import jax
import jax.numpy as jnp

from .simulation import _get_run_simulation
from .search_space import make_inputs, JAX_INJECT_DEFAULTS
from .scoring import (default_metric, default_std, compute_drift, precision_ok, reward as compute_reward, PRECISION_MODES)


class Evaluator:
    """
    Parameters
    ----------
    step_fn: (carry, _) -> (carry, output)
    init_carry_fn: (key, x0) -> carry
    master_key: JAX PRNGKey
    batch_size: number of parallel chains (fixed across all candidates)
    n_steps: scan length (fixed across all candidates)
    dtype: compute dtype (fixed across all candidates)
    state_shape: shape of one state vector, e.g. (2,) for 2-D problems
    precision: threshold value — meaning depends on precision_mode
    precision_mode: one of "absolute" | "relative" | "zscore" | "wasserstein" | "quantile" | "soft"
    n_repeats: number of timed runs (min is taken)
    """

    def __init__(
        self,
        step_fn,
        init_carry_fn,
        master_key,
        batch_size = 256,
        n_steps = 100,
        dtype = jnp.float32,
        state_shape = (2,),
        precision = 0.001,
        precision_mode = "absolute",
        n_repeats = 3,
    ):
        if precision_mode not in PRECISION_MODES:
            raise ValueError(f"Unknown precision_mode '{precision_mode}'. Choose from: {PRECISION_MODES}")
        self.step_fn = step_fn
        self.init_carry_fn = init_carry_fn
        self.master_key = master_key
        self.batch_size = batch_size
        self.n_steps = n_steps
        self.dtype = dtype
        self.state_shape = state_shape
        self.precision = precision
        self.precision_mode = precision_mode
        self.n_repeats = n_repeats
        self._baseline = None

        self._keys, self._inits = make_inputs(master_key, batch_size, state_shape)

    def _run_once(self, params: dict):
        flags = params.get("xla_flags", "")
        donate = params.get("jit_donate_argnums", ())
        if isinstance(donate, str):
            donate = tuple(int(x) for x in donate.strip("()").split(",") if x.strip())
        os.environ["XLA_FLAGS"] = flags
        run_fn = _get_run_simulation(donate)
        keys = jnp.array(self._keys)  if 0 in donate else self._keys
        inits = jnp.array(self._inits) if 1 in donate else self._inits
        return run_fn(
            keys, inits,
            step_fn = self.step_fn,
            init_carry_fn = self.init_carry_fn,
            n_steps = self.n_steps,
            scan_unroll = params.get("scan_unroll", JAX_INJECT_DEFAULTS["scan_unroll"]),
            scan_reverse = params.get("scan_reverse", JAX_INJECT_DEFAULTS["scan_reverse"]),
            dtype = self.dtype,
            xla_flags = flags
        )

    def _measure(self, params: dict) -> tuple[float, float, any]:
        """
        Returns (min_wall_time, mean_metric, outputs).
        """
        carries, outputs = self._run_once(params)
        jax.block_until_ready(carries)
        metric = default_metric(outputs)

        times = []
        for _ in range(self.n_repeats):
            t0 = time.perf_counter()
            carries, _ = self._run_once(params)
            jax.block_until_ready(carries)
            times.append(time.perf_counter() - t0)

        return min(times), metric, outputs


    def warm_up(self) -> None:
        """Pre-compute the baseline using JAX_INJECT_DEFAULTS with no XLA flags."""
        if self._baseline is not None:
            return
        os.environ["XLA_FLAGS"] = ""
        t, metric, outputs = self._measure({**JAX_INJECT_DEFAULTS, "xla_flags": ""})
        self._baseline = {"time": t, "metric": metric, "std": default_std(outputs), "outputs": outputs}

    def evaluate(self, params: dict) -> dict:
        """
        Returns
        -------
        dict with keys:
            time: wall time (seconds)
            metric: mean of outputs
            drift: precision drift value (mode-dependent units)
            speedup: baseline_time / candidate_time
            precision_ok: bool — drift within threshold
            reward: optimisation objective
        """
        self.warm_up()
        t, metric, outputs = self._measure(params)
        bl = self._baseline

        speedup = bl["time"] / t
        drift = compute_drift(
            mode = self.precision_mode,
            candidate_metric = metric,
            baseline_metric = bl["metric"],
            baseline_std = bl["std"],
            batch_size = self.batch_size,
            candidate_outputs = outputs,
            baseline_outputs = bl["outputs"]
        )
        ok = precision_ok(drift, self.precision, self.precision_mode)
        rew = compute_reward(speedup, drift, self.precision, self.precision_mode)

        return {
            "time": t,
            "metric": metric,
            "drift": drift,
            "speedup": speedup,
            "precision_ok": ok,
            "reward": rew
        }

    def trace(self, params: dict, output_dir: str = "./traces", n_runs: int = 3) -> str:
        """
        Run the simulation under Perfetto tracing and save the trace.
        Open the returned path at https://ui.perfetto.dev
        """
        import jax.profiler

        os.makedirs(output_dir, exist_ok=True)
        self.warm_up()

        carries, _ = self._run_once(params)
        jax.block_until_ready(carries)

        with jax.profiler.trace(output_dir, create_perfetto_trace=True):
            for _ in range(n_runs):
                carries, _ = self._run_once(params)
                jax.block_until_ready(carries)

        return output_dir

    @property
    def baseline(self) -> dict:
        """Baseline run result (lazy — triggers warm_up on first access)."""
        self.warm_up()
        return self._baseline
