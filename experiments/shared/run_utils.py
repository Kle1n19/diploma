import time
import traceback

def device_banner(exp_name: str = "") -> None:
    import jax
    import platform
    backend = jax.default_backend().upper()
    title   = f"  {exp_name}  " if exp_name else "  mc_tuner experiment  "

    if backend == "CPU":
        detail = platform.processor() or platform.machine() or "CPU"
    else:
        detail = backend

    n_dev = jax.device_count()
    print(f"{'━'*4}{title:^52}{'━'*4}")
    print(f"  Device : {backend}  —  {detail}")
    print(f"  JAX    : {jax.__version__}  |  {n_dev} device(s) visible")

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import all_combinations, JAX_INJECT_SPACE
from mc_tuner.searchers.grid_search import GridSearcher
from mc_tuner.searchers.random_search import RandomSearcher
from mc_tuner.searchers.rl_search import RLSearcher
from mc_tuner.agents.bandit import EpsilonGreedyAgent
from mc_tuner.agents.ucb import UCB1Agent
from mc_tuner.agents.softmax import SoftmaxAgent
from mc_tuner.agents.thompson import ThompsonSamplingAgent
from mc_tuner.agents.autosampler import AutoSamplerAgent
from mc_tuner.agents.sac import SACAgent
from mc_tuner.agents.grpo import GRPOAgent
from mc_tuner.hardware import xla_flag_space


def cummax(values: list[float]) -> list[float]:
    best, out = float("-inf"), []
    for v in values:
        best = max(best, v)
        out.append(best)
    return out


def serialize_params(p: dict) -> dict:
    out = {}
    for k, v in p.items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _build_agent(name: str, n: int, configs: list[dict], budget: int):
    if name == "ε-greedy":
        return EpsilonGreedyAgent(n_actions=n)
    if name == "UCB1":
        return UCB1Agent(n_actions=n)
    if name == "Softmax":
        return SoftmaxAgent(n_actions=n)
    if name == "Thompson":
        return ThompsonSamplingAgent(n_actions=n)
    if name == "TPE":
        return AutoSamplerAgent(n_actions=n, configs=configs, n_startup_trials=max(5, budget // 5))
    if name == "SAC":
        return SACAgent(n_actions=n, max_arms=min(n, 256), warmup_steps=min(10, budget // 4))
    if name == "GRPO":
        return GRPOAgent(n_actions=n, max_arms=min(n, 256), group_size=4, warmup_steps=min(8, budget // 5))
    raise ValueError(name)


def run_method(
    name: str,
    evaluator: Evaluator,
    configs: list[dict],
    budget: int,
    verbose: bool = True,
) -> dict:
    n = len(configs)
    print(f"  {name}  (budget={budget if name != 'Grid' else n})")

    t0 = time.perf_counter()
    try:
        if name == "Grid":
            searcher = GridSearcher(verbose=verbose)
            _, _, history = searcher.search(evaluator, configs)
        elif name == "Random":
            searcher = RandomSearcher(n=budget, seed=42, verbose=verbose)
            _, _, history = searcher.search(evaluator, configs)
        else:
            agent = _build_agent(name, n, configs, budget)
            searcher = RLSearcher(agent=agent, episodes=budget, verbose=verbose)
            _, _, history = searcher.search(evaluator, configs)
    except Exception:
        print(f"[{name}] FAILED:")
        traceback.print_exc()
        return {"name": name, "error": traceback.format_exc(), "history": []}

    elapsed = time.perf_counter() - t0
    rewards = [h["metrics"]["reward"] for h in history]
    speedups = [h["metrics"]["speedup"] for h in history]
    ok_flags = [h["metrics"]["precision_ok"] for h in history]

    result = {
        "name":  name,
        "n_evals": len(history),
        "elapsed_s": round(elapsed, 2),
        "rewards": rewards,
        "speedups": speedups,
        "ok_flags": ok_flags,
        "cum_best": cummax(rewards),
        "best_reward": max(rewards)  if rewards  else None,
        "best_speedup": max((s for s, ok in zip(speedups, ok_flags) if ok), default=None),
        "precision_rate": sum(ok_flags) / len(ok_flags) if ok_flags else 0.0,
        "best_params": serialize_params(history[rewards.index(max(rewards))]["params"] if rewards else {})
    }

    print(f"\n  → best reward={result['best_reward']:.3f}  "
          f"best speedup={result['best_speedup']:.3f}x  "
          f"precision_ok={result['precision_rate']*100:.0f}%  "
          f"({elapsed:.1f}s)")
    return result


def build_spaces(step_fn=None, sample_carry=None) -> list[tuple[str, list[dict]]]:
    import jax
    backend = jax.default_backend()
    base_configs = all_combinations(JAX_INJECT_SPACE)
    flags = xla_flag_space(backend)
    flag_configs = all_combinations({**JAX_INJECT_SPACE, "xla_flags": flags})
    return [("Base", base_configs), ("XLA flags", flag_configs)]
