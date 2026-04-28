import jax
import jax.numpy as jnp

from .evaluator import Evaluator
from .fn_evaluator import FnEvaluator
from .search_space import JAX_INJECT_SPACE, JAX_INJECT_DEFAULTS, all_combinations
from .hardware import xla_flag_space
from .searchers.rl_search import RLSearcher
from .searchers.grid_search import GridSearcher
from .searchers.random_search import RandomSearcher
from .agents.bandit import EpsilonGreedyAgent
from .agents.ucb import UCB1Agent
from .agents.softmax import SoftmaxAgent
from .agents.thompson import ThompsonSamplingAgent
from .agents.autosampler import AutoSamplerAgent
from .agents.sac import SACAgent
from .agents.grpo import GRPOAgent
from .io import save_config, load_config


_AGENT_REGISTRY = {
    "epsilon_greedy": EpsilonGreedyAgent,
    "ucb": UCB1Agent,
    "softmax":  SoftmaxAgent,
    "thompson": ThompsonSamplingAgent,
    "autosampler": AutoSamplerAgent,
    "sac": SACAgent,
    "grpo": GRPOAgent
}


class Tuner:
    """
    Parameters
    ----------
    step_fn            : (carry, _) -> (carry, output)   scan-compatible step function
    init_carry_fn      : (key, x0)  -> carry             build initial carry per chain
    search_space       : params to tune; merged with JAX_INJECT_SPACE.
                         Pass {} to tune inject params only.
    batch_size         : number of parallel chains — fixed across all candidates
    n_steps            : scan length — fixed across all candidates
    dtype              : compute dtype — fixed across all candidates
    state_shape        : shape of one state vector, e.g. (2,) for 2-D problems
    precision          : threshold value — meaning depends on precision_mode
    precision_mode     : "absolute" | "relative" | "zscore" | "wasserstein" | "quantile" | "soft"
    tune_xla_flags     : if True, add XLA_FLAGS combinations to the search space.
                         Each candidate gets its own compiled JIT slot.
    master_key         : JAX PRNGKey (default: PRNGKey(0))
    n_repeats          : timing repeats per evaluation (min wall time is used)

    Usage
    -----
    tuner = Tuner(step_fn=my_step, init_carry_fn=my_init, batch_size=512, n_steps=200)
    best_params, best_metrics = tuner.run(method="rl", agent="autosampler", episodes=60)

    # with XLA flag tuning:
    tuner = Tuner(step_fn=my_step, init_carry_fn=my_init, tune_xla_flags=True)
    """

    def __init__(
        self,
        step_fn,
        init_carry_fn,
        search_space = None,
        batch_size = 256,
        n_steps = 100,
        dtype = jnp.float32,
        state_shape = (2,),
        precision = 0.001,
        precision_mode = "absolute",
        tune_xla_flags = False,
        master_key = None,
        n_repeats = 3,
    ):
        base = dict(JAX_INJECT_SPACE)
        if tune_xla_flags:
            base["xla_flags"] = xla_flag_space(jax.default_backend())
        self.search_space = {**base, **(search_space or {})}
        self.configs = all_combinations(self.search_space)

        master_key = master_key if master_key is not None else jax.random.PRNGKey(0)
        self.evaluator = Evaluator(
            step_fn = step_fn,
            init_carry_fn = init_carry_fn,
            master_key = master_key,
            batch_size = batch_size,
            n_steps = n_steps,
            dtype = dtype,
            state_shape = state_shape,
            precision = precision,
            precision_mode = precision_mode,
            n_repeats = n_repeats,
        )

    def run(
        self,
        method = "rl",
        agent = "epsilon_greedy",
        episodes = 40,
        n = 50,
        verbose = True,
        trace_best = False,
        trace_dir = "./traces",
        **agent_kwargs,
    ) -> tuple[dict, dict]:
        """
        Run the search and return (best_params, best_metrics).

        Parameters
        ----------
        method     : "rl" | "grid" | "random"
        agent      : (rl only) agent name
        episodes   : (rl only) number of evaluate-update cycles
        n          : (random only) number of configs to sample
        verbose    : print progress lines
        trace_best : if True, run Perfetto tracing on the best config after search
        trace_dir  : directory to write the Perfetto trace (default: ./traces)
        """
        searcher = self._build_searcher(method, agent, episodes, n, verbose, **agent_kwargs)
        best_params, best_metrics, _ = searcher.search(self.evaluator, self.configs)
        if trace_best:
            path = self.evaluator.trace(best_params, output_dir=trace_dir)
            print(f"\nTrace saved {path}")
        return best_params, best_metrics

    def run_full(
        self,
        method = "rl",
        agent = "epsilon_greedy",
        episodes = 40,
        n = 50,
        verbose = True,
        trace_best = False,
        trace_dir = "./traces",
        **agent_kwargs,
    ) -> tuple[dict, dict, list[dict]]:
        """full evaluation history."""
        searcher = self._build_searcher(method, agent, episodes, n, verbose, **agent_kwargs)
        best_params, best_metrics, history = searcher.search(self.evaluator, self.configs)
        if trace_best:
            path = self.evaluator.trace(best_params, output_dir=trace_dir)
            print(f"\nTrace saved {path}")
        return best_params, best_metrics, history

    def trace(
        self,
        params,
        output_dir = "./traces",
        n_runs = 3,
    ) -> str:
        """
        Trace any config under Perfetto and return the trace directory path.

        Parameters
        ----------
        params     : config dict to trace — e.g. best_params, JAX_INJECT_DEFAULTS,
                     or any hand-crafted dict
        output_dir : where to write the trace files
        n_runs     : repetitions captured inside the trace window

        Example
        -------
        path = tuner.trace(best_params, output_dir="./traces/best")
        # then open path at https://ui.perfetto.dev
        """
        path = self.evaluator.trace(params, output_dir=output_dir, n_runs=n_runs)
        print(f"Perfetto trace saved {path}")
        return path

    def save(self, params: dict, path: str) -> None:
        save_config(params, path)

    def load(self, path: str) -> dict:
        return load_config(path)

    @property
    def n_configs(self) -> int:
        return len(self.configs)

    @property
    def baseline(self) -> dict:
        return self.evaluator.baseline

    @classmethod
    def from_callable(
        cls,
        fn,
        generate_inputs,
        fixed_kwargs = None,
        param_map = None,
        search_space = None,
        baseline_params = None,
        precision = 0.001,
        precision_mode = "absolute",
        master_key = None,
        n_repeats = 3,
    ) -> "Tuner":
        """
        Build a Tuner from any complete JAX function — no step decomposition needed.

        Parameters
        ----------
        fn               : JAX function with signature fn(x, **kwargs)
        generate_inputs  : (master_key, batch_size) -> jax array, shape (batch_size, ...)
        fixed_kwargs     : kwargs forwarded to fn unchanged (e.g. {"K": 100.0, "T": 1.0})
        param_map        : maps tuner param names to fn kwarg names, e.g. {"n_steps": "steps"}
        search_space     : override/extend the default search space
        baseline_params  : reference config for the fn evaluator
        precision        : max allowed deviation — meaning depends on precision_mode
        precision_mode   : "absolute" | "relative" | "zscore" | "wasserstein" | "quantile" | "soft"
        master_key       : JAX PRNGKey (default: PRNGKey(0))
        n_repeats        : timing repeats per evaluation
        """
        from .search_space import JAX_INJECT_SPACE
        merged_space = {**JAX_INJECT_SPACE, **(search_space or {})}
        master_key = master_key if master_key is not None else jax.random.PRNGKey(0)

        evaluator = FnEvaluator(
            fn = fn,
            generate_inputs = generate_inputs,
            fixed_kwargs = fixed_kwargs or {},
            param_map = param_map if param_map is not None else {"n_steps": "steps"},
            baseline_params = baseline_params or JAX_INJECT_DEFAULTS.copy(),
            master_key = master_key,
            precision = precision,
            precision_mode = precision_mode,
            n_repeats = n_repeats
        )

        instance = cls.__new__(cls)
        instance.search_space = merged_space
        instance.configs = all_combinations(merged_space)
        instance.evaluator = evaluator
        return instance

    def _build_searcher(self, method, agent_name, episodes, n, verbose, **agent_kwargs):
        if method == "grid":
            return GridSearcher(verbose=verbose)

        if method == "random":
            return RandomSearcher(n=n, verbose=verbose)

        if method == "rl":
            agent_cls = _AGENT_REGISTRY.get(agent_name)
            if agent_cls is None:
                raise ValueError(
                    f"Unknown agent '{agent_name}'. Choose from: {list(_AGENT_REGISTRY)}"
                )
            extra = {"configs": self.configs} if agent_name == "autosampler" else {}
            agent_obj = agent_cls(n_actions=self.n_configs, **extra, **agent_kwargs)
            return RLSearcher(agent=agent_obj, episodes=episodes, verbose=verbose)

        raise ValueError(f"Unknown method '{method}'. Choose from: 'rl', 'grid', 'random'")
