# mc_tuner — Automatic JAX Hyperparameter Tuner

A framework that automatically tunes JAX execution-level parameters (scan unrolling, buffer donation, numerical precision, activation checkpointing and more) for arbitrary user-defined JAX functions.

---

## What it does

JAX exposes low-level knobs that control how code is compiled and executed — things like `lax.scan` unroll depth, `jit` buffer donation, and matmul precision. The right values depend on the hardware, the algorithm and the batch size. Picking them by hand is tedious and non-obvious.

`mc_tuner` instruments your function at the source level (via LibCST), injects these knobs as tunable parameters and searches over the resulting space using bandit or Bayesian optimisation agents. It returns the fastest configuration that stays within a user-specified numerical precision.

---

## Tunable parameters

| Parameter | Values | Effect |
|---|---|---|
| `scan_unroll` | 1, 2, 4, 8, 16 | Loop unrolling depth inside `lax.scan` |
| `scan_reverse` | False, True | Traversal direction in `lax.scan` |
| `jit_donate_argnums` | (), (0,), (1,), (0,1) | Buffer donation (reuse input memory for output) |
| `dot_precision` | None, "high", "highest" | Floating-point precision for `jnp.dot` |
| `matmul_precision` | None, "high", "highest" | Floating-point precision for `jnp.matmul` |
| `autotune_level` | 0–4 | XLA kernel autotuning exhaustiveness |
| `map_chunk_size` | 1, 4, 8, 16, None | Chunk size for `lax.map` |
| `checkpoint_policy` | None + 3 JAX policies | Activation checkpointing strategy |

---

## Installation

**Requirements:** Python 3.10+, CUDA 12 (for GPU support)

```bash
git clone https://github.com/Kle1n19/diploma.git
cd diploma
conda create -n mc_tuner python=3.10
conda activate mc_tuner
pip install -r requirements.txt
```

For CPU-only (no CUDA):
```bash
pip install jax jaxlib libcst==1.8.6 optuna==4.8.0 numpy==1.26.4 scipy==1.15.3 matplotlib==3.10.8
```

---

## Quick start — AutoTuner

`AutoTuner` works on any JAX function defined in a `.py` file. It instruments the source automatically, finding and rewriting `lax.scan`, `lax.map`, and `jnp.dot` / `jnp.matmul` call sites.

### Example: Ornstein-Uhlenbeck path simulation

The Ornstein-Uhlenbeck (OU) process is a mean-reverting SDE used in interest rate models (Vasicek), RL exploration noise, and stochastic volatility. Its Euler-Maruyama discretisation maps directly onto a `lax.scan` loop:

$$x_{t+1} = x_t + \theta(\mu - x_t)\,dt + \sigma\sqrt{dt}\,\varepsilon_t, \quad \varepsilon_t \sim \mathcal{N}(0,1)$$

```python
import jax
import jax.numpy as jnp
from mc_tuner import AutoTuner

def ou_path(key, n_steps=2000, theta=0.7, mu=0.0, sigma=0.4, dt=0.005):
    noise = jax.random.normal(key, (n_steps,)) * jnp.sqrt(dt)

    def step(x, dW):                          # (carry, input) -> (carry, output)
        return x + theta * (mu - x) * dt + sigma * dW, x

    _, path = jax.lax.scan(step, jnp.float32(0.0), noise)
    return jnp.mean(path)

def generate_inputs(master_key, batch_size):
    return jax.random.split(master_key, batch_size)  # one key per sample

tuner = AutoTuner.from_fn(
    fn=ou_path,
    generate_inputs=generate_inputs,
    precision=0.005,   # max absolute drift in E[path] from baseline
    n_repeats=5,
)

best = tuner.run(method="grid", batch_size=256, verbose=True)
tuner.compare(batch_size=256)
print(best)
# e.g. {'scan_unroll': 8, 'scan_reverse': False, 'dot_precision': None, ...}
```

### The step function shape

The inner `step` follows `jax.lax.scan`'s required signature:

```
step(carry, x) -> (new_carry, output)
```

This is not an `mc_tuner` convention — it is imposed by JAX itself. `lax.scan` threads a `carry` state through successive inputs and accumulates per-step outputs into a stacked array. Any simulation expressible as "update a state and emit a value" fits this shape exactly. `AutoTuner.from_fn` detects `lax.scan` call sites in source and injects `unroll=` and `reverse=` automatically — you never touch the step signature.

### Search methods

| Method | Description |
|---|---|
| `"grid"` | Exhaustive search over all configs |
| `"random"` | Uniform random sampling (`n` configs) |
| `"rl"` | Bandit agent (see `agent=` parameter) |

### RL agents

```python
best = tuner.run(method="rl", agent="ucb", episodes=200)
# agent options: "ucb", "epsilon_greedy", "softmax", "thompson"
```

---

## Project structure

```
mc_tuner/
├── auto_tuner.py       # AutoTuner: end-to-end instrument → search → compare
├── instrumentor.py     # LibCST-based source instrumentation
├── fn_evaluator.py     # FnEvaluator: lower-level evaluator for plain callables
├── evaluator.py        # Legacy Evaluator (step_fn + init_carry_fn interface)
├── search_space.py     # JAX_INJECT_SPACE definitions and Cartesian product helper
├── scoring.py          # Precision modes, drift metrics, reward function
├── simulation.py       # JIT-compiled vmap simulation engine (legacy pipeline)
├── tuner.py            # Legacy Tuner (wraps Evaluator + Searcher)
├── io.py               # JSON serialisation for configs and results
├── hardware.py         # XLA flag space builder
├── agents/
│   ├── bandit.py       # Epsilon-greedy
│   ├── ucb.py          # UCB1
│   ├── thompson.py     # Thompson sampling
│   ├── softmax.py      # Boltzmann / softmax bandit
│   ├── sac.py          # Soft Actor-Critic (pure JAX neural net)
│   ├── grpo.py         # GRPO policy gradient agent
│   └── autosampler.py  # Optuna TPE wrapper
├── searchers/
│   ├── grid_search.py
│   ├── random_search.py
│   └── rl_search.py
└── tests/
    ├── test_unroll.py          # scan_unroll on GBM path simulation
    ├── test_reverse.py         # scan_reverse on discounted RL returns
    ├── test_precision.py       # dot_precision on GP posterior mean
    ├── test_chunk_size.py      # map_chunk_size
    ├── test_checkpoint.py      # checkpoint_policy
    ├── test_donate_argnums.py  # jit_donate_argnums on HMC
    └── bench_scan_reverse_cpu.py  # CPU memory-access benchmark

experiments/
├── domains/            # Reusable algorithm implementations
│   ├── american_option.py
│   ├── european_option.py
│   ├── basket_option.py
│   ├── runge_kutta.py
│   ├── kalman_smoother.py
│   └── kmc_random_walk.py
├── buffer_donation/    # Throughput vs donation sweep (state_dim × donate config)
├── convergence_speed/  # Agent convergence curves comparison
├── distributional_integrity/  # Wasserstein drift validation
├── precision_tradeoff/ # Pareto frontier: speedup vs numerical accuracy
├── speedup_magnitude/  # Wall-time speedup across 5 diverse domains
└── tuning_roi/         # Break-even analysis: when does tuning pay off?

trace_utils/
├── get_trace.py        # Re-runs the best config from convergence results and emits XLA traces
├── analyze_traces.py   # Compares baseline vs tuned trace profiles
└── sql/                # Perfetto SQL queries for manual trace analysis
    ├── dispatch.sql    # Kernel dispatch counts and average duration
    ├── fused.sql       # Fused op breakdown
    ├── memory.sql      # Memory transfer slices
    ├── overhead.sql    # pjit vs XLA GPU module overhead ratio
    ├── while.sql       # XLA while-loop slice timing
    ├── sq1.sql
    └── sq2.sql
```

The SQL queries in `trace_utils/sql/` are designed for [Perfetto UI](https://ui.perfetto.dev). Open a `.json.gz` trace file there, then paste any query into the **Query** tab to slice the trace by kernel type, memory ops or dispatch overhead.

---

## Running experiments

Each experiment is self-contained:

```bash
# Speedup across domains
python experiments/speedup_magnitude/run.py

# Agent convergence comparison
python experiments/convergence_speed/run.py

# Buffer donation sweep
python experiments/buffer_donation/run.py

# Precision-speedup Pareto frontier
python experiments/precision_tradeoff/run.py
```

Results are saved to `experiments/<name>/results/results.json` and figures to `experiments/<name>/figures/`.

---

## XLA trace profiling

After running an experiment, you can compare XLA traces for the baseline and best-found configuration:

```bash
python "trace_utils/get_trace.py"      # emits traces to ./traces/baseline and ./traces/tuned
python "trace_utils/analyze_traces.py" # diff and summarise the two profiles
```

`get_trace.py` reads the best result from `experiments/convergence_speed/results/results.json`, deserialises the parameter set, and calls `tuner.trace()` for both the default and tuned configs.

---

## Precision modes

The `precision` argument controls how much numerical drift from the baseline is acceptable. Set via `precision_mode` (default `"absolute"`):

| Mode | Meaning |
|---|---|
| `"absolute"` | `\|mean(candidate) - mean(baseline)\| ≤ threshold` |
| `"relative"` | Relative deviation as a fraction of baseline |
| `"zscore"` | Deviation in units of baseline standard deviation |
| `"wasserstein"` | Wasserstein-1 distance between output distributions |
| `"quantile"` | Max quantile drift across 7 probability levels |
| `"soft"` | Gaussian-penalised reward (no hard gate) |

---
