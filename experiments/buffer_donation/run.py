import sys, json, argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import jax
import jax.numpy as jnp

from mc_tuner.evaluator import Evaluator
from mc_tuner.search_space import JAX_INJECT_DEFAULTS
from experiments.domains.runge_kutta import make_fns as rk_fns

FIGURES_DIR = Path(__file__).parent / "figures"
RESULTS_DIR = Path(__file__).parent / "results"

STATE_DIMS = [4, 16, 64, 256, 512]
DONATIONS = [(), (0,), (0, 1)]
N_STEPS = 256
BATCH_SIZE = 512
N_REPEATS = 5

DONATION_LABELS = {(): "No donation", (0,): "donate carry", (0, 1): "donate carry+xs"}
DONATION_COLORS = {(): "#7f7f7f", (0,): "#4e79a7", (0, 1): "#f28e2b"}


def measure_throughput(step_fn, init_carry_fn, donate, batch_size, n_steps, n_repeats):
    evaluator = Evaluator(
        step_fn = step_fn,
        init_carry_fn = init_carry_fn,
        master_key = jax.random.PRNGKey(0),
        batch_size = batch_size,
        n_steps = n_steps,
        state_shape = (1,),
        precision = 1e9,
        n_repeats = n_repeats
    )
    params = {**JAX_INJECT_DEFAULTS, "jit_donate_argnums": donate}
    try:
        metrics = evaluator.evaluate(params)
        wall_time = evaluator.baseline["time"] / metrics["speedup"]
        throughput = batch_size * n_steps / wall_time
        return throughput, metrics["speedup"]
    except Exception as e:
        print(f"    [WARN] {e}")
        return None, None


def run_experiment(batch_size, n_steps, n_repeats, state_dims):
    results = {}

    for dim in state_dims:
        print(f"\n  state_dim={dim}")
        step_fn, init_carry_fn = rk_fns(dim=dim)
        results[dim] = {}

        baseline_tp = None
        for donate in DONATIONS:
            tp, sp = measure_throughput(step_fn, init_carry_fn, donate, batch_size, n_steps, n_repeats)
            results[dim][str(donate)] = {"throughput": tp, "speedup": sp}
            if donate == () and tp is not None:
                baseline_tp = tp
            donation_sp = tp / baseline_tp if (tp and baseline_tp) else None
            print(f"    donate={str(donate):<12}  "
                  f"throughput={tp:.0f} samples/s  "
                  f"donation_speedup={donation_sp:.3f}x" if tp else
                  f"    donate={str(donate)}  FAILED")

    return results


def plot_results(results, out_dir, state_dims=None):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    if state_dims is None:
        state_dims = sorted(int(k) for k in results.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    for donate in DONATIONS:
        label = DONATION_LABELS[donate]
        color = DONATION_COLORS[donate]
        tps = [results.get(str(dim), {}).get(str(donate), {}).get("throughput") for dim in state_dims]
        valid = [(d, t) for d, t in zip(state_dims, tps) if t is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "o-", label=label, color=color, linewidth=2, markersize=6)

    ax.set_xlabel("State dimension", fontsize=12)
    ax.set_ylabel("Throughput (samples/sec)", fontsize=12)
    ax.set_title("Throughput vs State Dimension", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    for donate in DONATIONS:
        if not donate:
            continue
        label = DONATION_LABELS[donate]
        color = DONATION_COLORS[donate]
        ratios = []
        for dim in state_dims:
            base_tp = results.get(str(dim), {}).get("()", {}).get("throughput")
            don_tp = results.get(str(dim), {}).get(str(donate), {}).get("throughput")
            ratios.append(don_tp / base_tp if (base_tp and don_tp) else None)
        valid = [(d, r) for d, r in zip(state_dims, ratios) if r is not None]
        if valid:
            xs, ys = zip(*valid)
            ax.plot(xs, ys, "s-", label=label, color=color, linewidth=2, markersize=6)

    ax.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax.set_xlabel("State dimension", fontsize=12)
    ax.set_ylabel("Donation speedup ratio", fontsize=12)
    ax.set_title("Buffer Donation Benefit vs State Size", fontsize=13, fontweight="bold")
    ax.set_xscale("log")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Buffer Donation — Runge-Kutta ODE Solver", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "01_buffer_donation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_dir / '01_buffer_donation.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    from experiments.shared.run_utils import device_banner
    device_banner("Exp — Buffer Donation")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = RESULTS_DIR / "results.json"

    if args.plot_only:
        with open(results_path) as f:
            payload = json.load(f)
        plot_results(payload["results"], FIGURES_DIR, payload.get("state_dims"))
        return

    state_dims = [4, 64, 512] if args.fast else STATE_DIMS
    batch_size = 64 if args.fast else BATCH_SIZE
    n_steps = 64 if args.fast else N_STEPS
    n_repeats = 2 if args.fast else N_REPEATS

    print(f"Backend: {jax.default_backend().upper()}")
    results = run_experiment(batch_size, n_steps, n_repeats, state_dims)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "backend": jax.default_backend(),
        "state_dims": state_dims,
        "results": {str(k): v for k, v in results.items()},
    }
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved  {results_path}")

    print("\nGenerating plots…")
    plot_results({str(k): v for k, v in results.items()}, FIGURES_DIR, state_dims)
    print(f"\nDone. Figures in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
