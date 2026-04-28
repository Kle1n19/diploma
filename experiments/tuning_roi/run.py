import sys, json, argparse, time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import all_combinations, JAX_INJECT_SPACE
from experiments.shared.run_utils import run_method, device_banner
from experiments.domains.american_option import make_fns as american_fns
from experiments.domains.european_option import make_fns as european_fns
from experiments.domains.basket_option import make_fns as basket_fns
from experiments.domains.runge_kutta import make_fns as rk_fns
from experiments.domains.kalman_smoother import make_fns as kalman_fns

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

BUDGET = 1000
N_STEPS = 252
BATCH_SIZE = 512
N_REPEATS = 3

DOMAINS = [
    ("American Option", "MC", american_fns, (1,), 0.05, "relative"),
    ("European Option", "MC", european_fns, (1,), 0.05, "relative"),
    ("Basket Option", "MC", basket_fns, (1,), 0.05, "relative"),
    ("Runge-Kutta ODE", "Non-MC", rk_fns, (1,), 0.001, "absolute"),
    ("Kalman Smoother", "Non-MC", kalman_fns, (1,), 0.001, "absolute"),
]

FAMILY_COLORS = {"MC": "#4e79a7", "Non-MC": "#f28e2b"}


def _make_fns_safe(make_fns_fn, n_steps):
    try:
        return make_fns_fn(n_steps=n_steps)
    except TypeError:
        return make_fns_fn()


def run_domain_roi(name, family, make_fns_fn, state_shape, precision, prec_mode, batch_size, n_steps, n_repeats, budget):
    print(f"  Domain: {name}  [{family}]")

    step_fn, init_carry_fn = _make_fns_safe(make_fns_fn, n_steps)

    evaluator = Evaluator(
        step_fn=step_fn, init_carry_fn=init_carry_fn,
        master_key=jax.random.PRNGKey(42),
        batch_size=batch_size, n_steps=n_steps,
        state_shape=state_shape, precision=precision,
        precision_mode=prec_mode, n_repeats=n_repeats)
    baseline_time = evaluator.baseline["time"]

    configs = all_combinations(JAX_INJECT_SPACE)

    t0 = time.perf_counter()
    result = run_method("TPE", evaluator, configs, budget, verbose=False)
    tuning_time = time.perf_counter() - t0

    best_speedup = result.get("best_speedup") or 1.0
    tuned_time = baseline_time / best_speedup
    saved_per_run = baseline_time - tuned_time

    if saved_per_run > 0:
        break_even = tuning_time / saved_per_run
    else:
        break_even = float("inf")

    hours_saved_1k = (saved_per_run * 1000) / 3600

    roi = {
        "domain": name,
        "family": family,
        "baseline_ms": round(baseline_time * 1000, 3),
        "tuned_ms": round(tuned_time * 1000, 3),
        "saved_ms": round(saved_per_run * 1000, 3),
        "speedup": round(best_speedup, 4),
        "tuning_time_s": round(tuning_time, 2),
        "break_even_runs": round(break_even, 1),
        "hours_saved_1k": round(hours_saved_1k, 4),
        "n_configs": len(configs)
    }

    print(f"  Baseline:    {roi['baseline_ms']:.2f} ms")
    print(f"  Tuned:       {roi['tuned_ms']:.2f} ms  ({roi['speedup']:.3f}x)")
    print(f"  Tuning time: {roi['tuning_time_s']:.1f} s")
    print(f"  Break-even:  {roi['break_even_runs']:.0f} runs")
    print(f"  Saved 1k runs: {roi['hours_saved_1k']:.4f} hours")
    return roi


def plot_results(results, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if r.get("break_even_runs") != float("inf")]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    sorted_r = sorted(valid, key=lambda r: r["break_even_runs"])
    names = [r["domain"] for r in sorted_r]
    bes = [r["break_even_runs"] for r in sorted_r]
    colors = [FAMILY_COLORS.get(r["family"], "#888") for r in sorted_r]
    y = np.arange(len(names))
    bars = ax.barh(y, bes, color=colors, edgecolor="white", alpha=0.88)
    ax.bar_label(bars, fmt="%.0f runs", padding=4, fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Break-even (number of production runs)", fontsize=12)
    ax.set_title("How Many Runs Until Tuning Pays Off?", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=k) for k, c in FAMILY_COLORS.items()], fontsize=11)

    ax = axes[1]
    xs = np.arange(len(sorted_r))
    tt = [r["tuning_time_s"] for r in sorted_r]
    saved = [r["hours_saved_1k"] * 3600 for r in sorted_r]

    ax.bar(xs, tt, label="Tuning time (s)", color="#e15759", alpha=0.85, edgecolor="white")
    ax.bar(xs, saved, label="Savings over 1k runs (s)", color="#59a14f", alpha=0.75, edgecolor="white", bottom=tt)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Seconds", fontsize=12)
    ax.set_title("Tuning Cost vs Savings (1000-run horizon)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("mc_tuner ROI Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "01_tuning_roi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / '01_tuning_roi.png'}")

    print(f"\n{'Domain':<22} {'Speedup':>8} {'Tuning(s)':>10} {'Break-even':>12} {'Saved 1k(h)':>12}")
    print("─" * 70)
    for r in results:
        print(f"{r['domain']:<22} {r['speedup']:>8.3f}x {r['tuning_time_s']:>10.1f} "
              f"{r['break_even_runs']:>12.0f} {r['hours_saved_1k']:>12.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--budget", type=int, default=BUDGET)
    args = parser.parse_args()
    device_banner("Exp — Tuning ROI")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    if args.plot_only:
        with open(results_path) as f:
            payload = json.load(f)
        plot_results(payload["results"], FIGURES_DIR)
        return

    batch_size = 64  if args.fast else BATCH_SIZE
    n_steps = 32  if args.fast else N_STEPS
    n_repeats = 1   if args.fast else N_REPEATS
    budget = 10  if args.fast else args.budget

    all_results = []
    for name, family, make_fns_fn, state_shape, precision, prec_mode in DOMAINS:
        r = run_domain_roi(name, family, make_fns_fn, state_shape, precision, prec_mode, batch_size, n_steps, n_repeats, budget)
        all_results.append(r)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "budget": budget,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "backend": jax.default_backend(),
        "results": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {results_path}")

    print("\nGenerating plots…")
    plot_results(all_results, FIGURES_DIR)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
