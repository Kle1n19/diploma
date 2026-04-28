import sys
import os
import json
import time
import argparse
import traceback
from pathlib import Path
from datetime import datetime

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import all_combinations, JAX_INJECT_SPACE
from experiments.domains.american_option import make_fns as option_fns
from experiments.shared.run_utils import cummax, serialize_params, run_method, _build_agent, build_spaces, device_banner
from experiments.shared.plot_style import COLORS, LINE_STYLES, set_style, save_fig

SPACE = JAX_INJECT_SPACE
GRID_MAX_CONFIGS = None

BUDGET = 4096
N_STEPS    = 252
BATCH_SIZE = 1024
N_REPEATS  = 3

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"


def evals_to_fraction(cum_best: list[float], target_fraction: float) -> int | None:
    """Return first eval index where cum_best >= target_fraction * cum_best[-1]."""
    target = target_fraction * cum_best[-1]
    for i, v in enumerate(cum_best):
        if v >= target:
            return i + 1
    return None



def plot_results(results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import numpy as np
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    valid = [r for r in results if not r.get("error") and r["rewards"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    for r in valid:
        xs = list(range(1, len(r["cum_best"]) + 1))
        ax.plot(xs, r["cum_best"],
                label=r["name"],
                color=COLORS.get(r["name"], "#333333"),
                linestyle=LINE_STYLES.get(r["name"], "solid"),
                linewidth=2.2,
                marker="o" if len(xs) <= 50 else None,
                markersize=4,
                alpha=0.9)

    ax.set_xlabel("Evaluations", fontsize=13)
    ax.set_ylabel("Best reward found so far", fontsize=13)
    ax.set_title("Convergence Speed — American Option Pricing (GBM paths)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "01_convergence_curves.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '01_convergence_curves.png'}")

    names = [r["name"] for r in valid]
    speedups = [r["best_speedup"] for r in valid]
    colors = [COLORS.get(n, "#333333") for n in names]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, speedups, color=colors, edgecolor="white", linewidth=1.2, alpha=0.88)
    ax.bar_label(bars, fmt="%.2fx", padding=4, fontsize=10)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5, label="baseline (1.0×)")
    ax.set_ylabel("Best speedup achieved", fontsize=13)
    ax.set_title("Final Best Speedup by Search Method", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "02_final_speedup.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '02_final_speedup.png'}")

    global_best = max(r["best_reward"] for r in valid)
    thresholds = [0.80, 0.90, 0.95]
    t_labels = ["80%", "90%", "95%"]
    n_methods = len(valid)
    x = np.arange(n_methods)
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 5))
    for ti, (frac, label) in enumerate(zip(thresholds, t_labels)):
        vals = []
        for r in valid:
            target = frac * global_best
            idx = next((i + 1 for i, v in enumerate(r["cum_best"]) if v >= target), None)
            vals.append(idx if idx is not None else r["n_evals"] * 1.05)
        bars = ax.bar(x + ti * width, vals, width, label=f"≥{label} of global best", alpha=0.82, edgecolor="white")

    ax.set_xticks(x + width)
    ax.set_xticklabels([r["name"] for r in valid], fontsize=11)
    ax.set_ylabel("Evaluations needed", fontsize=13)
    ax.set_title("Evaluations to Reach X% of Global Best Reward", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "03_evals_to_threshold.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '03_evals_to_threshold.png'}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    data = [r["rewards"] for r in valid]
    labels = [r["name"] for r in valid]
    vp = ax.violinplot(data, showmedians=True, showextrema=True)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(COLORS.get(labels[i], "#aaa"))
        body.set_alpha(0.7)
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Reward", fontsize=12)
    ax.set_title("Reward Distribution (all evaluations)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    ok_rates = [r["precision_rate"] * 100 for r in valid]
    col = [COLORS.get(r["name"], "#333") for r in valid]
    bars = ax.bar(labels, ok_rates, color=col, edgecolor="white", alpha=0.88)
    ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=10)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Precision-OK rate (%)", fontsize=12)
    ax.set_title("Fraction of Numerically Valid Configs Found", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_dir / "04_distributions.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '04_distributions.png'}")

    fig = plt.figure(figsize=(16, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :])
    for r in valid:
        xs = list(range(1, len(r["cum_best"]) + 1))
        ax0.plot(xs, r["cum_best"],
                 label=r["name"],
                 color=COLORS.get(r["name"], "#333"),
                 linestyle=LINE_STYLES.get(r["name"], "solid"),
                 linewidth=2.2, alpha=0.9)
    ax0.set_xlabel("Evaluations", fontsize=12)
    ax0.set_ylabel("Best reward", fontsize=12)
    ax0.set_title("Convergence Curves", fontsize=13, fontweight="bold")
    ax0.legend(ncol=3, fontsize=10, framealpha=0.9)
    ax0.grid(True, alpha=0.3)
    ax0.spines[["top", "right"]].set_visible(False)

    ax1 = fig.add_subplot(gs[1, 0])
    bars = ax1.bar(names, speedups, color=colors, edgecolor="white", alpha=0.88)
    ax1.bar_label(bars, fmt="%.2fx", padding=3, fontsize=9)
    ax1.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax1.set_ylabel("Best speedup", fontsize=12)
    ax1.set_title("Final Speedup", fontsize=13, fontweight="bold")
    ax1.grid(axis="y", alpha=0.3)
    ax1.spines[["top", "right"]].set_visible(False)
    plt.setp(ax1.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    ax2 = fig.add_subplot(gs[1, 1])
    ok_bars = ax2.bar(labels, ok_rates, color=col, edgecolor="white", alpha=0.88)
    ax2.bar_label(ok_bars, fmt="%.0f%%", padding=3, fontsize=9)
    ax2.set_ylim(0, 115)
    ax2.set_ylabel("Precision-OK (%)", fontsize=12)
    ax2.set_title("Numerical Validity Rate", fontsize=13, fontweight="bold")
    ax2.grid(axis="y", alpha=0.3)
    ax2.spines[["top", "right"]].set_visible(False)
    plt.setp(ax2.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    fig.suptitle(
        f"mc_tuner — Convergence Speed Experiment\n"
        f"Domain: American Option Pricing (GBM)  |  "
        f"Space: {len(all_combinations(SPACE))} configs  |  Budget: {BUDGET} evals",
        fontsize=14, fontweight="bold", y=1.01,
    )
    fig.savefig(out_dir / "00_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / '00_summary.png'}")


def plot_compare_modes(modes_results: list[dict], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_labels  = [m["mode"] for m in modes_results]
    method_names = [r["name"] for r in modes_results[0]["results"] if not r.get("error") and r.get("rewards")]

    n_modes = len(mode_labels)
    n_methods = len(method_names)
    x = np.arange(n_methods)
    width = 0.22
    offsets = np.linspace(-(n_modes - 1) / 2, (n_modes - 1) / 2, n_modes) * width
    MODE_COLORS = ["#4e79a7", "#f28e2b", "#e15759"]

    fig, ax = plt.subplots(figsize=(13, 6))
    for mi, (mode_data, color, offset) in enumerate(
            zip(modes_results, MODE_COLORS, offsets)):
        valid = {r["name"]: r for r in mode_data["results"] if not r.get("error") and r.get("rewards")}
        vals  = [valid.get(n, {}).get("best_speedup", 0) or 0 for n in method_names]
        bars  = ax.bar(x + offset, vals, width, label=mode_data["mode"], color=color, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, fmt="%.2fx", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(method_names, fontsize=11)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.4)
    ax.set_ylabel("Best speedup achieved", fontsize=13)
    ax.set_title("Search Mode Comparison — Best Speedup per Agent", fontsize=14, fontweight="bold")
    ax.legend(fontsize=12, framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "10_mode_speedup_comparison.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '10_mode_speedup_comparison.png'}")

    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5), sharey=True)
    if n_modes == 1:
        axes = [axes]

    for ax, mode_data, color in zip(axes, modes_results, MODE_COLORS):
        valid = [r for r in mode_data["results"] if not r.get("error") and r.get("rewards")]
        for r in valid:
            xs = list(range(1, len(r["cum_best"]) + 1))
            ax.plot(xs, r["cum_best"],
                    label=r["name"],
                    color=COLORS.get(r["name"], "#333"),
                    linestyle=LINE_STYLES.get(r["name"], "solid"),
                    linewidth=2, alpha=0.9)
        n_cfg = mode_data["n_configs"]
        ax.set_title(f"{mode_data['mode']}\n({n_cfg:,} configs)", fontsize=12, fontweight="bold", color=color)
        ax.set_xlabel("Evaluations", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(fontsize=9, loc="lower right")

    axes[0].set_ylabel("Best reward found so far", fontsize=12)
    fig.suptitle("Convergence Curves by Search Mode — American Option Pricing", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "11_mode_convergence.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '11_mode_convergence.png'}")

    fig, ax = plt.subplots(figsize=(9, 5))
    for mode_data, color in zip(modes_results, MODE_COLORS):
        valid = [r for r in mode_data["results"] if not r.get("error") and r.get("rewards")]
        xs = [mode_data["n_configs"]] * len(valid)
        ys = [r["best_speedup"] for r in valid]
        ns = [r["name"] for r in valid]
        ax.scatter(xs, ys, color=color, s=80, alpha=0.8, zorder=3)
        for x_pt, y_pt, n in zip(xs, ys, ns):
            ax.annotate(n, (x_pt, y_pt), textcoords="offset points",
                        xytext=(6, 2), fontsize=8, alpha=0.8)

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='o', color='w',
                            markerfacecolor=c, markersize=10, label=m["mode"])
                    for m, c in zip(modes_results, MODE_COLORS)]
    ax.legend(handles=legend_elems, fontsize=11)
    ax.set_xscale("log")
    ax.set_xlabel("Search space size (log scale)", fontsize=12)
    ax.set_ylabel("Best speedup found", fontsize=12)
    ax.set_title("Space Size vs Best Speedup Found (same budget)", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_dir / "12_space_vs_reward.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / '12_space_vs_reward.png'}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true",
                        help="Smoke-test: small batch, few repeats, skip slow agents")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip evaluation, re-plot last saved results")
    parser.add_argument("--grid", action="store_true",
                        help="Include exhaustive grid search (slow — excluded by default)")
    parser.add_argument("--budget", type=int, default=BUDGET,
                        help=f"Evaluation budget for RL/Random (default {BUDGET})")
    parser.add_argument("--n-steps", type=int, default=None,
                        help="Override simulation steps per eval (lower = faster evals)")
    parser.add_argument("--n-repeats", type=int, default=None,
                        help="Override timing repeats per config (default 3)")
    parser.add_argument("--compare-modes", action="store_true")
    args = parser.parse_args()

    device_banner("Exp 1 — Convergence Speed")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    if args.plot_only:
        if not results_path.exists():
            print(f"No results found at {results_path}. Run without --plot-only first.")
            sys.exit(1)
        with open(results_path) as f:
            payload = json.load(f)
        all_results = payload["results"] if "results" in payload else payload
        print("\nRe-plotting saved results…")
        plot_results(all_results, FIGURES_DIR)
        return

    budget = args.budget
    batch_size = 64 if args.fast else BATCH_SIZE
    n_repeats = 1 if args.fast else (args.n_repeats or N_REPEATS)
    n_steps = 32 if args.fast else (args.n_steps or N_STEPS)

    step_fn, init_carry_fn = option_fns(n_steps=n_steps)
    sample_carry = init_carry_fn(jax.random.PRNGKey(0), jnp.zeros(()))

    evaluator = Evaluator(
        step_fn = step_fn,
        init_carry_fn = init_carry_fn,
        master_key = jax.random.PRNGKey(0),
        batch_size = batch_size,
        n_steps = n_steps,
        state_shape = (1,),
        precision = 0.05,
        precision_mode = "relative",
        n_repeats = n_repeats
    )
    print("\nWarming up baseline (JIT compile)…", end=" ", flush=True)
    t_warm = time.perf_counter()
    _ = evaluator.baseline
    print(f"done ({time.perf_counter()-t_warm:.1f}s)")
    methods = ["ε-greedy", "UCB1", "Softmax", "Thompson", "TPE", "SAC", "GRPO", "Random"]
    if args.fast:
        methods = ["ε-greedy", "UCB1", "Random"]

    if args.compare_modes:
        compare_path = RESULTS_DIR / "compare_modes.json"
        spaces = build_spaces(step_fn, sample_carry=sample_carry)
        modes_results = []
        base_size = len(spaces[0][1])
        for mode_label, mode_configs in spaces:
            n_cfg = len(mode_configs)
            scale = n_cfg / base_size
            mode_budget = max(budget, int(budget * scale))
            skip_grid = not args.grid or (GRID_MAX_CONFIGS is not None and n_cfg > GRID_MAX_CONFIGS)
            mode_methods = methods + ([] if skip_grid else ["Grid"])

            print(f"  MODE: {mode_label}  ({n_cfg:,} configs, "
                  f"{mode_budget/n_cfg*100:.2f}% sampled, budget={mode_budget})")

            mode_res = []
            for name in mode_methods:
                result = run_method(name, evaluator, mode_configs, mode_budget, verbose=False)
                mode_res.append(result)

            modes_results.append({
                "mode": mode_label,
                "n_configs": n_cfg,
                "budget": mode_budget,
                "results": mode_res,
            })

        payload = {
            "timestamp": datetime.now().isoformat(),
            "mode": "compare_modes",
            "budget": budget,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "backend": jax.default_backend(),
            "modes": modes_results,
        }
        with open(compare_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved  {compare_path}")

        print(f"\n{'Mode':<20} {'Method':<12} {'Speedup':>10} {'Reward':>10}")
        print("─" * 56)
        for md in modes_results:
            for r in md["results"]:
                if not r.get("error") and r.get("rewards"):
                    print(f"{md['mode']:<20} {r['name']:<12} "
                          f"{r['best_speedup']:>9.3f}x {r['best_reward']:>10.3f}")

        print("\nGenerating mode-comparison plots…")
        plot_compare_modes(modes_results, FIGURES_DIR)
        print(f"\nDone. Figures in {FIGURES_DIR}/")
        return

    configs = all_combinations(SPACE)
    space_tag = "full JAX_INJECT_SPACE"

    n_configs = len(configs)
    skip_grid = not args.grid or (GRID_MAX_CONFIGS is not None and n_configs > GRID_MAX_CONFIGS)

    print(f"  Search space: {n_configs:,} configs  — {space_tag}")
    print(f"  Budget: {budget} evaluations per RL/Random method")
    print(f"                 ({budget/n_configs*100:.2f}% of the full space sampled)")
    print(f"  Grid search: {'SKIPPED (pass --grid to enable)' if skip_grid else f'{n_configs} evals (exhaustive)'}")
    print(f"  Domain: American Option Pricing — {n_steps} steps × {batch_size} chains")
    print(f"  Device: {jax.default_backend().upper()}  ({jax.device_count()} device(s))")

    if not skip_grid:
        methods.append("Grid")

    all_results = []
    for name in methods:
        result = run_method(name, evaluator, configs, budget, verbose=False)
        all_results.append(result)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "space_size": n_configs,
        "space_tag": space_tag,
        "budget": budget,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "n_repeats": n_repeats,
        "backend": jax.default_backend(),
        "results": all_results,
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved  {results_path}")

    print(f"\n{'Method':<12} {'Evals':>6} {'Best reward':>12} {'Best speedup':>13} {'Precision-OK':>13}")
    for r in all_results:
        if r.get("error"):
            print(f"{r['name']:<12}  ERROR")
        else:
            print(f"{r['name']:<12} {r['n_evals']:>6} {r['best_reward']:>12.3f}"
                  f" {r['best_speedup']:>12.3f}x {r['precision_rate']*100:>12.0f}%")

    print("\nGenerating plots…")
    plot_results(all_results, FIGURES_DIR)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
