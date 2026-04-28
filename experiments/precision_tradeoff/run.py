import sys, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import all_combinations, JAX_INJECT_SPACE
from experiments.shared.run_utils import run_method, device_banner
from experiments.domains.american_option import make_fns as option_fns

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

THRESHOLDS = [0.0001, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
BUDGET = 1000
N_STEPS = 252
BATCH_SIZE = 512
N_REPEATS = 3


def run_threshold(threshold, step_fn, init_carry_fn, configs, batch_size, n_steps, n_repeats, budget):
    evaluator = Evaluator(
        step_fn = step_fn,
        init_carry_fn = init_carry_fn,
        master_key = jax.random.PRNGKey(0),
        batch_size = batch_size,
        n_steps = n_steps,
        state_shape = (1,),
        precision = threshold,
        precision_mode = "relative",
        n_repeats = n_repeats
    )
    result = run_method("TPE", evaluator, configs, budget, verbose=False)
    result["threshold"] = threshold
    return result


def plot_results(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if not r.get("error") and r.get("best_speedup")]

    thresholds = [r["threshold"]      for r in valid]
    speedups = [r["best_speedup"]   for r in valid]
    ok_rates = [r["precision_rate"] for r in valid]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    ax.plot(thresholds, speedups, "o-", color="#4e79a7", linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Precision threshold (relative)", fontsize=12)
    ax.set_ylabel("Best speedup", fontsize=12)
    ax.set_title("Speedup vs Precision Threshold", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ax.plot(thresholds, [r * 100 for r in ok_rates], "s-", color="#f28e2b", linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Precision threshold (relative)", fontsize=12)
    ax.set_ylabel("Precision-OK rate (%)", fontsize=12)
    ax.set_title("Precision-OK Rate vs Threshold", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    sc = ax.scatter(thresholds, speedups, c=np.log10(thresholds), cmap="viridis", s=100, zorder=3)
    for t, s in zip(thresholds, speedups):
        ax.annotate(f"{t}", (t, s), textcoords="offset points", xytext=(5, 5), fontsize=8, alpha=0.8)
    ax.set_xscale("log")
    ax.set_xlabel("Threshold (log scale)", fontsize=12)
    ax.set_ylabel("Best speedup", fontsize=12)
    ax.set_title("Pareto Frontier: Accuracy vs Speed", fontsize=13, fontweight="bold")
    plt.colorbar(sc, ax=ax, label="log10(threshold)")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Precision–Speedup Tradeoff — American Option Pricing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "01_precision_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / '01_precision_tradeoff.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    args = parser.parse_args()
    device_banner("Exp — Precision Tradeoff")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    if args.plot_only:
        with open(results_path) as f:
            payload = json.load(f)
        plot_results(payload["results"], FIGURES_DIR)
        return

    batch_size = 64 if args.fast else BATCH_SIZE
    n_steps = 32 if args.fast else (args.n_steps or N_STEPS)
    n_repeats = 1 if args.fast else (args.n_repeats or N_REPEATS)
    budget = 10 if args.fast else args.budget

    step_fn, init_carry_fn = option_fns(n_steps=n_steps)
    configs = all_combinations(JAX_INJECT_SPACE)
    print(f"Search space: {len(configs)} configs")

    thresholds = [0.05, 0.1, 0.5] if args.fast else THRESHOLDS
    all_results = []
    for threshold in thresholds:
        print(f"\n  threshold={threshold}")
        r = run_threshold(threshold, step_fn, init_carry_fn, configs, batch_size, n_steps, n_repeats, budget)
        all_results.append(r)
        print(f"  speedup={r.get('best_speedup', 'N/A'):.3f}x  "
              f"ok_rate={r.get('precision_rate', 0)*100:.0f}%")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "budget": budget,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "backend": jax.default_backend(),
        "results": all_results
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved  {results_path}")
    plot_results(all_results, FIGURES_DIR)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
