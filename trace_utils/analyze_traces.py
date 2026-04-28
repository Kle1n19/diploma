import gzip, json, glob, argparse, os
import numpy as np
from collections import defaultdict
from contextlib import redirect_stdout

GPU_CFG = dict(
    baseline_glob = "traces/baseline/plugins/profile/**/*.trace.json.gz",
    tuned_glob = "traces/tuned/plugins/profile/**/*.trace.json.gz",
    xla_module_key = "jit_run_simulation:XLA GPU module",
    pjit_key = "PjitFunction(run_simulation)",
    dispatch_stack = [
        "PjitFunction(run_simulation)",
        "PjRtCApiLoadedExecutable::Execute",
        "PJRT_LoadedExecutable_Execute",
        "CommonPjRtLoadedExecutable::Execute (jit_run_simulation)",
        "CommonPjRtLoadedExecutable::ExecuteHelperOnSingleDevice",
        "[0] PjRtStreamExecutorRawLoadedExecutable::Execute",
        "[0] GpuExecutable::ExecuteThunks",
        "jit_run_simulation:XLA GPU module"
    ],
    fusion_prefixes = ("loop_", "wrapped_", "command_buffer", "cuGraphLaunch"),
    while_prefix = "while",
    memcpy_names = {"MemcpyD2D", "memcpy32_post", "copy.18", "BUFFER_FLUSH"},
    xla_label = "XLA GPU module"
)

TPU_CFG = dict(
    baseline_glob = "tpu/traces/baseline/plugins/profile/**/*.trace.json.gz",
    tuned_glob = "tpu/traces/tuned/plugins/profile/**/*.trace.json.gz",
    xla_module_key = None,
    pjit_key = "PjitFunction(run_simulation)",
    dispatch_stack = [
        "PjitFunction(run_simulation)",
        "PJRT_LoadedExecutable_Execute",
        "TpuLoadedExecutable::Execute",
        "TpuLoadedExecutable::ExecuteHelperOnSingleDevice",
        "TpuLoadedExecutable::ExecuteLaunch",
        "tpu::System::Execute"
    ],
    fusion_prefixes = ("add_", "select_", "pad_", "fusion", "while", "copy_bitcast"),
    while_prefix = "while",
    memcpy_names = {"AllocateOutputBufferWithInputReuse", "copy-start", "copy-start.1", "copy-start.2", "copy-done", "copy-done.1", "copy-done.2", "copy.4", "copy.2"},
    xla_label = "XLA TPU module",
)

PROFILER_NOISE = {
    "$profiler.py:271 trace", "$profiler.py:106 start_trace",
    "$profiler.py:115 start_trace", "$profiler.py:218 stop_trace",
    "$profiler.py:235 stop_trace", "$contextlib.py:132 __enter__",
    "$contextlib.py:136 __enter__", "$contextlib.py:141 __exit__",
    "$contextlib.py:145 __exit__"
}


def load(path):
    with gzip.open(path, "rt") as f:
        data = json.load(f)
    return [e for e in data.get("traceEvents", [])
            if isinstance(e, dict) and e.get("ph") == "X" and e.get("dur", 0) > 0
            and e.get("name") not in PROFILER_NOISE]


def resolve_xla_key(by_name, cfg):
    if cfg["xla_module_key"] is not None and cfg["xla_module_key"] in by_name:
        return cfg["xla_module_key"]
    for name in by_name:
        if name.startswith("jit_run_simulation("):
            return name
    for name in by_name:
        if "GpuExecutable::ExecuteThunks" in name:
            return name
    return None


def section(title):
    print(f"  {title}")


def analyze(events, label, cfg):
    by_name = defaultdict(list)
    for e in events:
        by_name[e["name"]].append(e["dur"])

    xla_key = resolve_xla_key(by_name, cfg)
    section(label)

    print("\n  [1] Dispatch stack")
    print(f" {'Layer':<58} {'n':>3}  {'avg_ms':>8}")
    for name in cfg["dispatch_stack"]:
        durs = by_name.get(name, [])
        if durs:
            print(f" {name:<58} {len(durs):>3}  {np.mean(durs)/1e3:>8.3f}")

    fusion_events = {k: v for k, v in by_name.items() if any(k.startswith(p) for p in cfg["fusion_prefixes"])}
    while_events  = {k: v for k, v in by_name.items() if k.startswith(cfg["while_prefix"])}
    memcpy_events = {k: v for k, v in by_name.items() if k in cfg["memcpy_names"]}

    print(f"\n [2] XLA fused kernels")
    print(f" {'Kernel name':<45} {'calls':>6}  {'avg_ms':>8}  {'total_ms':>10}")
    for name, durs in sorted(fusion_events.items(), key=lambda x: -sum(x[1])):
        print(f" {name:<45} {len(durs):>6}  "
              f"{np.mean(durs)/1e3:>8.4f}  {sum(durs)/1e3:>10.4f}")

    n_fused = sum(len(v) for v in fusion_events.values())
    n_while = sum(len(v) for v in while_events.values())
    n_memcpy = sum(len(v) for v in memcpy_events.values())
    total_fused_ms = sum(sum(v) for v in fusion_events.values()) / 1e3

    if while_events:
        print(f"\n [3] While-loop body kernels (lax.scan → XLA WhileOp)")
        for name, durs in while_events.items():
            print(f" {name:<45} {len(durs):>6}  "
                  f"{np.mean(durs)/1e3:>8.4f} {sum(durs)/1e3:>10.4f}")

    if memcpy_events:
        print(f"\n [4] Memory transfers / buffer ops")
        for name, durs in memcpy_events.items():
            print(f" {name:<45} {len(durs):>6}  "
                  f"{np.mean(durs)/1e3:>8.4f}  {sum(durs)/1e3:>10.4f}")

    xla_durs = by_name.get(xla_key, []) if xla_key else []
    pjit_durs = by_name.get(cfg["pjit_key"], [])

    print(f"\n [5] Run-to-run jitter ({cfg['xla_label']}, {len(xla_durs)} runs)")
    if xla_durs:
        for i, d in enumerate(xla_durs):
            print(f" run {i+1}: {d/1e3:.4f} ms")
        print(f" std dev : {np.std(xla_durs)/1e3:.4f} ms  "
              f"({np.std(xla_durs)/np.mean(xla_durs)*100:.1f}% CV)")

    overhead = 0
    if pjit_durs and xla_durs:
        matched_pjit = sorted(pjit_durs, reverse=True)[:len(xla_durs)]
        avg_pjit = np.mean(matched_pjit)
        avg_xla = np.mean(xla_durs)
        overhead = max((avg_pjit - avg_xla) / avg_pjit * 100, 0)
        print(f"\n  [6] Dispatch overhead breakdown")
        print(f" Total call time (PjitFunction, top-{len(xla_durs)}) : {avg_pjit/1e3:.3f} ms")
        print(f" Actual XLA time ({cfg['xla_label']}): {avg_xla/1e3:.3f} ms")
        print(f" Python/JAX dispatch overhead: {(avg_pjit-avg_xla)/1e3:.3f} ms  ({overhead:.1f}%)")

    return {
        "xla_avg_us": float(np.mean(xla_durs)) if xla_durs else 0,
        "xla_std_us": float(np.std(xla_durs))  if xla_durs else 0,
        "n_fused": n_fused,
        "n_while": n_while,
        "n_memcpy": n_memcpy,
        "total_fused_ms": total_fused_ms,
        "dispatch_overhead_pct": overhead
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", action="store_true", help="Analyse TPU traces instead of GPU")
    parser.add_argument("--label", default="", help="Device label appended to output filename (e.g. A100, H100, V100)")
    args = parser.parse_args()

    cfg = TPU_CFG if args.tpu else GPU_CFG

    baseline_files = sorted(glob.glob(cfg["baseline_glob"], recursive=True))
    tuned_files = sorted(glob.glob(cfg["tuned_glob"],    recursive=True))

    if not baseline_files or not tuned_files:
        raise FileNotFoundError(f"No trace files found.\n  baseline: {cfg['baseline_glob']}\n  tuned: {cfg['tuned_glob']}")

    events_base  = load(baseline_files[-1])
    events_tuned = load(tuned_files[-1])

    device_tag = "tpu" if args.tpu else "gpu"
    if args.label:
        device_tag = f"{device_tag}_{args.label}"
    os.makedirs("results", exist_ok=True)
    out_path = f"results/analysis_{device_tag}.txt"

    with open(out_path, "w") as f, redirect_stdout(f):
        s_base = analyze(events_base, "BASELINE (unoptimized)", cfg)
        s_tuned = analyze(events_tuned, "TUNED (optimized)",   cfg)

        section("THESIS SUMMARY")
        speedup = s_base["xla_avg_us"] / s_tuned["xla_avg_us"] if s_tuned["xla_avg_us"] else 0

        rows = [
            (f"{cfg['xla_label']} avg time",  f"{s_base['xla_avg_us']/1e3:.3f} ms",
                                               f"{s_tuned['xla_avg_us']/1e3:.3f} ms",
                                               f"{speedup:.2f}x faster"),
            ("Run-to-run std dev",             f"{s_base['xla_std_us']/1e3:.4f} ms",
                                               f"{s_tuned['xla_std_us']/1e3:.4f} ms", ""),
            ("Fused kernel invocations",       str(s_base["n_fused"]),
                                               str(s_tuned["n_fused"]), ""),
            ("While-loop body calls",          str(s_base["n_while"]),
                                               str(s_tuned["n_while"]), ""),
            ("Memory transfer ops",            str(s_base["n_memcpy"]),
                                               str(s_tuned["n_memcpy"]), ""),
            ("Dispatch overhead",              f"{s_base['dispatch_overhead_pct']:.1f}%",
                                               f"{s_tuned['dispatch_overhead_pct']:.1f}%", ""),
        ]

        print(f"\n {'Metric':<32} {'Baseline':>14} {'Tuned':>14} {'Note':>16}")
        for row in rows:
            print(f" {row[0]:<32} {row[1]:>14} {row[2]:>14} {row[3]:>16}")

    print(f"Results written to {out_path}")
