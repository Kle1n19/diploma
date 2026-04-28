import inspect
import textwrap
import time
from dataclasses import dataclass, field
from functools import partial
from typing import Callable

import jax
import jax.numpy as jnp

from .instrumentor import from_fn as _instrument_from_fn, Finding
from .search_space import all_combinations, JAX_INJECT_DEFAULTS
from .scoring import default_metric, compute_speedup, reward as compute_reward


class InstrumentationError(RuntimeError):
    """In case of failure to parse and inject the target function."""


@dataclass
class AutoTunerResult:
    best_params: dict
    best_metrics: dict
    baseline_metrics: dict
    history: list[dict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)



@partial(jax.jit, static_argnames=["_fn"])
def _vmapped_call(_fn, inputs):
    return jax.vmap(_fn)(inputs)


class _AutoTunerEvaluator:
    """
    Adapts an (original_fn, instrumented_fn) pair to the Searcher protocol.
    evaluate(params) returns:
        {"time", "metric", "speedup", "precision_ok", "reward"}
    """

    def __init__(
        self,
        original_fn: Callable,
        instrumented_fn: Callable,
        generate_inputs: Callable,
        fixed_kwargs: dict,
        master_key,
        precision: float,
        n_repeats: int,
        batch_size: int,
    ):
        self._original_fn = original_fn
        self._instrumented_fn = instrumented_fn
        self._generate_inputs = generate_inputs
        self._fixed_kwargs = fixed_kwargs or {}
        self._master_key = master_key
        self._precision = precision
        self._n_repeats = n_repeats
        self._batch_size = batch_size
        self._baseline_cache = None

    def _strip_jit(self, fn):
        while hasattr(fn, "__wrapped__"):
            fn = fn.__wrapped__
        return fn

    def _measure(self, fn: Callable, inject_params: dict) -> tuple[float, float]:
        """Run fn (bound with inject_params + fixed_kwargs), return (min_time, metric)."""
        raw = self._strip_jit(fn)
        bound = partial(raw, **self._fixed_kwargs, **inject_params)
        inputs = self._generate_inputs(self._master_key, self._batch_size)

        # warmup
        outputs = _vmapped_call(bound, inputs)
        jax.block_until_ready(outputs)
        metric = default_metric(outputs)

        times = []
        for _ in range(self._n_repeats):
            t0 = time.perf_counter()
            out = _vmapped_call(bound, inputs)
            jax.block_until_ready(out)
            times.append(time.perf_counter() - t0)

        return max(times), metric

    def warm_up(self) -> None:
        if self._baseline_cache is not None:
            return
        t, metric = self._measure(self._original_fn, {})
        self._baseline_cache = {"time": t, "metric": metric}

    @property
    def baseline(self) -> dict:
        self.warm_up()
        return self._baseline_cache

    def evaluate(self, params: dict) -> dict:
        self.warm_up()
        t, metric = self._measure(self._instrumented_fn, params)
        bl = self._baseline_cache
        speedup = bl["time"] / t if t > 0 else 1.0
        ok = abs(metric - bl["metric"]) <= self._precision
        rew = compute_reward(speedup, metric, bl["metric"], self._precision)
        return {"time": t, "metric": metric, "speedup": speedup, "precision_ok": ok, "reward": rew}


class AutoTuner:
    """
    End-to-end JAX parameter tuner driven by LibCST code injection..
    """

    def __init__(
        self,
        original_fn: Callable,
        instrumented_fn: Callable,
        search_space: dict,
        param_map: dict,
        findings: list[Finding],
        generate_inputs: Callable,
        fixed_kwargs: dict | None,
        master_key,
        precision: float,
        n_repeats: int,
    ):
        self._original_fn = original_fn
        self._instrumented_fn = instrumented_fn
        self.search_space = search_space
        self.param_map = param_map
        self.findings = findings
        self._generate_inputs = generate_inputs
        self._fixed_kwargs = fixed_kwargs or {}
        self._master_key = master_key
        self._precision = precision
        self._n_repeats = n_repeats
        self._result: AutoTunerResult | None = None


    @classmethod
    def from_fn(
        cls,
        fn: Callable,
        generate_inputs: Callable,
        fixed_kwargs: dict | None = None,
        master_key=None,
        precision: float = 0.05,
        n_repeats: int = 3,
        include_autotune: bool = False,
        skip_present: bool = True,
    ) -> "AutoTuner":
        """
        Parameters
        ----------
        fn: JAX function to optimise (must be in a .py file)
        generate_inputs: (master_key, batch_size) -> jnp array of per-sample inputs
        fixed_kwargs: kwargs forwarded unchanged to fn on every call
        master_key: JAX PRNGKey (default: PRNGKey(0))
        precision: max allowed |metric_candidate - metric_baseline|
        n_repeats: timing repeats (min wall-time is used)
        include_autotune: inject autotune_level XLA env flag param
        skip_present: skip injecting params that already exist in calls
        """
        raw = fn
        while hasattr(raw, "__wrapped__"):
            raw = raw.__wrapped__

        try:
            src = textwrap.dedent(inspect.getsource(raw))
        except OSError as exc:
            raise InstrumentationError(
                f"Cannot retrieve source for {fn!r}. "
                "AutoTuner requires a function defined in a .py file, not a REPL."
                ) from exc

        instrumented_src, param_map, search_space = _instrument_from_fn(raw, skip_present=skip_present, include_autotune=include_autotune)

        if not search_space:
            raise InstrumentationError(
                f"No tunable patterns found in {getattr(raw, '__name__', repr(raw))!r}. "
                "Ensure the function uses lax.scan, jnp.dot, jnp.matmul, lax.map, "
                "@jax.jit, or @jax.checkpoint."
            )

        globs = {**raw.__globals__}
        if "__builtins__" not in globs:
            import builtins
            globs["__builtins__"] = builtins
        compiled = compile(instrumented_src, f"<instrumented:{raw.__name__}>", "exec")
        exec(compiled, globs)
        instrumented_fn = globs[raw.__name__]

        return cls(
            original_fn=raw,
            instrumented_fn=instrumented_fn,
            search_space=search_space,
            param_map=param_map,
            findings=[],
            generate_inputs=generate_inputs,
            fixed_kwargs=fixed_kwargs,
            master_key=master_key if master_key is not None else jax.random.PRNGKey(0),
            precision=precision,
            n_repeats=n_repeats,
        )

    def run(
        self,
        method = "random",
        agent = "ucb",
        episodes = 30,
        n = 20,
        batch_size = 64,
        verbose = True
    ) -> dict:
        """
        Search over injected parameter space.

        Parameters
        ----------
        method: "random" | "grid" | "rl"
        agent: (rl only) "epsilon_greedy" | "ucb" | "softmax" | "thompson"
        episodes: (rl only) number of evaluate-update cycles
        n: (random only) number of configs to sample
        batch_size: number of samples per timing run
        verbose: print progress
        """
        evaluator = _AutoTunerEvaluator(
            original_fn=self._original_fn,
            instrumented_fn=self._instrumented_fn,
            generate_inputs=self._generate_inputs,
            fixed_kwargs=self._fixed_kwargs,
            master_key=self._master_key,
            precision=self._precision,
            n_repeats=self._n_repeats,
            batch_size=batch_size,
        )

        if verbose:
            print(f"[AutoTuner] Warming up baseline ({self._original_fn.__name__})...")
        evaluator.warm_up()
        bl = evaluator.baseline
        if verbose:
            print(f"[AutoTuner] Baseline: time={bl['time']:.4f}s  metric={bl['metric']:.6f}")

        configs = all_combinations(self.search_space)
        if verbose:
            print(f"[AutoTuner] Search space: {len(configs)} configs via method='{method}'")

        searcher = self._build_searcher(method, agent, episodes, n, verbose)
        best_params, best_metrics, history = searcher.search(evaluator, configs)

        self._result = AutoTunerResult(
            best_params=best_params,
            best_metrics=best_metrics,
            baseline_metrics=bl,
            history=history,
            findings=self.findings,
        )
        return best_params

    def compare(self, batch_size: int = 64) -> None:
        if self._result is None:
            print("[AutoTuner] No results yet. Call run() first.")
            return

        bl = self._result.baseline_metrics
        best = self._result.best_metrics
        params = self._result.best_params

        print("\n--- Metrics (Baseline -> Best) ---")
        print(f"wall time (s): {bl['time']:.4f} -> {best.get('time', '?'):.4f}")
        print(f"speedup: 1.00x -> {best.get('speedup', 1.0):.2f}x")
        print(f"output metric: {bl['metric']:.5f} -> {best.get('metric', '?'):.5f}")
        print(f"precision_ok: True -> {best.get('precision_ok', '?')}")

        print("\n--- Parameters (Default -> Best) ---")
        for k, v in params.items():
            default = JAX_INJECT_DEFAULTS.get(k, "—")
            print(f"{k}: {default} -> {v}")
        print()


    def _build_searcher(self, method, agent_name, episodes, n, verbose):
        from .searchers.grid_search import GridSearcher
        from .searchers.random_search import RandomSearcher
        from .searchers.rl_search import RLSearcher
        from .agents.bandit import EpsilonGreedyAgent
        from .agents.ucb import UCB1Agent
        from .agents.softmax import SoftmaxAgent
        from .agents.thompson import ThompsonSamplingAgent

        if method == "grid":
            return GridSearcher(verbose=verbose)
        if method == "random":
            return RandomSearcher(n=n, verbose=verbose)
        if method == "rl":
            registry = {
                "epsilon_greedy": EpsilonGreedyAgent,
                "ucb": UCB1Agent,
                "softmax": SoftmaxAgent,
                "thompson": ThompsonSamplingAgent,
            }
            agent_cls = registry[agent_name]
            configs = all_combinations(self.search_space)
            return RLSearcher(agent=agent_cls(n_actions=len(configs)), episodes=episodes, verbose=verbose)
        raise ValueError(f"Unknown search method '{method}'. Choose: 'random', 'grid', 'rl'.")
