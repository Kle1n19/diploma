import sys, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp
import numpy as np

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import JAX_INJECT_DEFAULTS, JAX_INJECT_SPACE, all_combinations
from mc_tuner.simulation import run_simulation
from mc_tuner.scoring import _drift_wasserstein, _drift_quantile
from experiments.shared.run_utils import run_method, device_banner
from experiments.domains.american_option import make_fns as option_fns

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

SPEEDUP_RESULTS = ROOT / "experiments/speedup_magnitude/results/results.json"

N_CHAINS = 10_000
N_STEPS = 252
N_REPEATS = 3
BUDGET = 1000
QUANTILES = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]


def _get_best_params(step_fn, init_carry_fn, batch_size, n_steps, n_repeats, budget):
    if SPEEDUP_RESULTS.exists():
        with open(SPEEDUP_RESULTS) as f:
            payload = json.load(f)
        for r in payload.get("results", []):
            if "American" in r.get("domain", "") and r.get("best_params"):
                print(f"  Using cached best_params from speedup_magnitude run.")
                return r["best_params"]

    evaluator = Evaluator(
        step_fn=step_fn, init_carry_fn=init_carry_fn,
        master_key=jax.random.PRNGKey(0),
        batch_size=batch_size, n_steps=n_steps,
        state_shape=(1,), precision=0.05, precision_mode="relative",
        n_repeats=n_repeats,
    )
    configs = all_combinations(JAX_INJECT_SPACE)
    result = run_method("TPE", evaluator, configs, budget, verbose=False)
    return result.get("best_params", JAX_INJECT_DEFAULTS)


def _collect_outputs(step_fn, init_carry_fn, params, master_key, n_chains, n_steps):
    evaluator = Evaluator(
        step_fn = step_fn,
        init_carry_fn = init_carry_fn,
        master_key = master_key,
        batch_size = n_chains,
        n_steps = n_steps,
        state_shape = (1,),
        precision = 1e9,
        n_repeats = 1,
    )
    _, outputs = evaluator._run_once(params)
    jax.block_until_ready(outputs)
    flat = jnp.ravel(outputs.astype(jnp.float32))
    return np.array(flat)


def plot_results(baseline_vals, tuned_vals, metrics, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    b_sorted = np.sort(baseline_vals)
    t_sorted = np.sort(tuned_vals)
    p = np.linspace(0, 1, len(b_sorted))
    ax.plot(b_sorted, p, label="Baseline", color="#4e79a7", linewidth=2)
    ax.plot(t_sorted, p, label="Tuned", color="#f28e2b", linewidth=2, linestyle="--")
    ax.set_xlabel("Output value", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("CDF Overlay (Baseline vs Tuned)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    min_len = min(len(b_sorted), len(t_sorted))
    idx = np.linspace(0, min_len - 1, 500, dtype=int)
    ax.scatter(b_sorted[idx], t_sorted[idx], s=10, alpha=0.4, color="#4e79a7")
    mn, mx = min(b_sorted.min(), t_sorted.min()), max(b_sorted.max(), t_sorted.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="y=x (perfect)")
    ax.set_xlabel("Baseline quantiles", fontsize=12)
    ax.set_ylabel("Tuned quantiles", fontsize=12)
    ax.set_title("Q-Q Plot", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    q_labels = [f"q{int(q*100)}" for q in QUANTILES]
    ax.bar(q_labels, metrics["quantile_drifts"], color="#e15759", edgecolor="white", alpha=0.85)
    ax.set_ylabel("Relative drift", fontsize=12)
    ax.set_title("Per-Quantile Drift (Tuned vs Baseline)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ["W1 distance",f"{metrics['wasserstein']:.4f}"],
        ["Max quantile drift",f"{metrics['max_q_drift']:.4f}"],
        ["Mean quantile drift",f"{metrics['mean_q_drift']:.4f}"],
        ["N chains", str(metrics["n_chains"])],
    ]
    tbl = ax.table(cellText=rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.2, 1.8)
    ax.set_title("Summary Metrics", fontsize=13, fontweight="bold")

    fig.suptitle("Distributional Integrity — American Option Pricing", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "01_distributional_integrity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / '01_distributional_integrity.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    device_banner("Exp — Distributional Integrity")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    n_chains = 256 if args.fast else N_CHAINS
    n_steps = 32 if args.fast else N_STEPS
    n_repeats = 1 if args.fast else N_REPEATS
    budget = 10 if args.fast else BUDGET

    if args.plot_only:
        with open(results_path) as f:
            payload = json.load(f)
        b = np.array(payload["baseline_vals"])
        t = np.array(payload["tuned_vals"])
        plot_results(b, t, payload["metrics"], FIGURES_DIR)
        return

    step_fn, init_carry_fn = option_fns(n_steps=n_steps)
    root_key = jax.random.PRNGKey(7)
    baseline_key, tuned_key = jax.random.split(root_key)

    print("Finding best tuned params…")
    best_params = _get_best_params(step_fn, init_carry_fn, 512, n_steps, n_repeats, budget)
    print(f"  best_params: {best_params}")

    print(f"\nCollecting baseline outputs ({n_chains} chains)…")
    baseline_vals = _collect_outputs(step_fn, init_carry_fn, JAX_INJECT_DEFAULTS, baseline_key, n_chains, n_steps)

    print(f"Collecting tuned outputs ({n_chains} chains)…")
    tuned_vals = _collect_outputs(step_fn, init_carry_fn, best_params, tuned_key, n_chains, n_steps)

    b_jnp = jnp.array(baseline_vals)
    t_jnp = jnp.array(tuned_vals)
    w1 = _drift_wasserstein(t_jnp, b_jnp)
    q_arr = jnp.array(QUANTILES)
    bq = np.quantile(baseline_vals, QUANTILES)
    tq = np.quantile(tuned_vals, QUANTILES)
    q_drifts = np.abs(tq - bq) / (np.abs(bq) + 1e-9)

    metrics = {
        "wasserstein": float(w1),
        "max_q_drift": float(q_drifts.max()),
        "mean_q_drift": float(q_drifts.mean()),
        "quantile_drifts": q_drifts.tolist(),
        "n_chains": n_chains,
    }

    print(f"\n  W1 distance:       {metrics['wasserstein']:.4f}")
    print(f"  Max quantile drift: {metrics['max_q_drift']:.4f}")
    print(f"  Mean quantile drift:{metrics['mean_q_drift']:.4f}")

    payload = {
        "timestamp": datetime.now().isoformat(),
        "n_chains": n_chains,
        "n_steps": n_steps,
        "best_params": best_params,
        "metrics": metrics,
        "baseline_vals": baseline_vals[:5000].tolist(),
        "tuned_vals": tuned_vals[:5000].tolist(),
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved  {results_path}")

    print("\nGenerating plots…")
    plot_results(baseline_vals, tuned_vals, metrics, FIGURES_DIR)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
