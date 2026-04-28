import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp

from mc_tuner import AutoTuner


def ais_importance_weight(key, n_temps=32, D=16, n_chains=32):
    chain_keys = jax.random.split(key, n_chains)
    betas = jnp.linspace(0.0, 1.0, n_temps + 1)

    def log_p0(x):
        return -0.5 * jnp.sum(x ** 2)

    def log_p1(x):
        return -0.5 * jnp.sum((x - jnp.ones(D) * 2.0) ** 2)

    def log_pt(x, beta):
        return (1.0 - beta) * log_p0(x) + beta * log_p1(x)

    def run_chain(chain_key):
        step_keys = jax.random.split(chain_key, n_temps + 1)
        x0 = jax.random.normal(step_keys[0], (D,))

        def anneal_step(carry, t):
            x, log_w = carry
            prev_beta = betas[t]
            next_beta = betas[t + 1]
            step_key = step_keys[t + 1]

            log_w = log_w + (next_beta - prev_beta) * (log_p1(x) - log_p0(x))

            x_prop = x + 0.3 * jax.random.normal(step_key, (D,))
            log_ratio = log_pt(x_prop, next_beta) - log_pt(x, next_beta)
            accept = jax.random.uniform(step_key) < jnp.exp(jnp.minimum(0.0, log_ratio))
            x = jnp.where(accept, x_prop, x)
            return (x, log_w), None

        (_, log_weight), _ = jax.lax.scan(anneal_step, (x0, 0.0), jnp.arange(n_temps))
        return log_weight

    log_weights = jax.lax.map(run_chain, chain_keys)
    return jnp.mean(log_weights)


def generate_inputs(master_key, batch_size):
    return jax.random.split(master_key, batch_size)


def test_chunk_size_ais(batch_size=128):
    print("Test: map_chunk_size | Annealed Importance Sampling"))

    tuner = AutoTuner.from_fn(
        fn=ais_importance_weight,
        generate_inputs=generate_inputs,
        fixed_kwargs={"n_temps": 32, "D": 16, "n_chains": 32},
        precision=0.5,
        n_repeats=5,
    )

    best = tuner.run(method="grid", batch_size=batch_size, verbose=True)
    tuner.compare(batch_size=batch_size)

    assert "map_chunk_size" in best, "map_chunk_size should be in best params"
    print(f"Best map_chunk_size = {best['map_chunk_size']}")
    return best


if __name__ == "__main__":
    test_chunk_size_ais()
