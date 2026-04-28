import sys, json, time, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import all_combinations, JAX_INJECT_DEFAULTS
from experiments.shared.run_utils import run_method, device_banner
from experiments.domains.american_option import make_fns as american_fns
from experiments.domains.european_option import make_fns as european_fns
from experiments.domains.basket_option import make_fns as basket_fns
from experiments.domains.runge_kutta import make_fns as rk_fns
from experiments.domains.kalman_smoother import make_fns as kalman_fns

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

DOMAINS = [
    ("American Option", "MC", american_fns, (1,), 0.05, "relative"),
    ("European Option", "MC", european_fns, (1,), 0.05, "relative"),
    ("Basket Option", "MC", basket_fns, (1,), 0.05, "relative"),
    ("Runge-Kutta ODE", "Non-MC", rk_fns, (1,), 0.001, "absolute"),
    ("Kalman Smoother", "Non-MC", kalman_fns, (1,), 0.001, "absolute"),
]

BUDGET = 4096
N_STEPS = 252
BATCH_SIZE = 1024
N_REPEATS = 3

DOMAIN_COLORS = {"MC": "#4e79a7", "Non-MC": "#f28e2b"}


def run_domain(name, family, make_fns_fn, state_shape, precision, prec_mode, batch_size, n_steps, n_repeats, budget):
    print(f"  Domain: {name}  [{family}]")

    step_fn, init_carry_fn = make_fns_fn(n_steps) if name != "Runge-Kutta ODE" else make_fns_fn()
    if name == "Kalman Smoother":
        step_fn, init_carry_fn = make_fns_fn(n_steps=n_steps)

    evaluator = Evaluator(
        step_fn = step_fn,
        init_carry_fn = init_carry_fn,
        master_key = jax.random.PRNGKey(42),
        batch_size = batch_size,
        n_steps = n_steps,
        state_shape = state_shape,
        precision = precision,
        precision_mode = prec_mode,
        n_repeats = n_repeats
    )
    baseline_time = evaluator.baseline["time"]

    from mc_tuner.search_space import JAX_INJECT_SPACE
    configs = all_combinations(JAX_INJECT_SPACE)

    result = run_method("TPE", evaluator, configs, budget, verbose=False)
    result["domain"] = name
    result["family"] = family
    result["baseline_ms"] = round(baseline_time * 1000, 3)
    result["tuned_ms"] = round(baseline_time / result["best_speedup"] * 1000, 3) if result.get("best_speedup") else None
    result["n_configs"] = len(configs)
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

    names = [r["domain"] for r in valid]
    speedups = [r["best_speedup"] for r in valid]
    colors = [DOMAIN_COLORS.get(r["family"], "#888") for r in valid]

    fig, ax = plt.subplots(figsize=(10, 5))
    y = np.arange(len(names))
    bars = ax.barh(y, speedups, color=colors, edgecolor="white", alpha=0.88)
    ax.bar_label(bars, fmt="%.2fx", padding=4, fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.axvline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("Wall-time speedup (higher = faster)", fontsize=12)
    ax.set_title("Speedup Magnitude by Domain\n(TPE agent, full search space)", fontsize=13, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elems = [Patch(facecolor=c, label=k) for k, c in DOMAIN_COLORS.items()]
    ax.legend(handles=legend_elems, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / "01_speedup_magnitude.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / '01_speedup_magnitude.png'}")

    print(f"\n{'Domain':<22} {'Family':<8} {'Baseline ms':>12} {'Tuned ms':>10} {'Speedup':>9} {'Configs':>8}")
    for r in results:
        if r.get("error"):
            print(f"{r['domain']:<22}  ERROR")
        else:
            print(f"{r['domain']:<22} {r['family']:<8} {r['baseline_ms']:>12.2f} "
                  f"{r['tuned_ms'] or 0:>10.2f} {r['best_speedup']:>8.3f}x "
                  f"{r['n_configs']:>8}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--budget", type=int, default=BUDGET)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--n-repeats", type=int, default=None)
    args = parser.parse_args()
    device_banner("Exp — Speedup Magnitude")
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

    all_results = []
    for name, family, make_fns_fn, state_shape, precision, prec_mode in DOMAINS:
        r = run_domain(name, family, make_fns_fn, state_shape, precision, prec_mode, batch_size, n_steps, n_repeats, budget)
        all_results.append(r)

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
    print(f"\nResults saved → {results_path}")

    print("\nGenerating plots…")
    plot_results(all_results, FIGURES_DIR)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
