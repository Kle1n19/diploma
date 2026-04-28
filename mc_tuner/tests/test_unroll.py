"""
Algorithm: GBM path simulation for European call option pricing.
  dS_t = S_t * (mu*dt + sigma*sqrt(dt)*dW_t)
  European call payoff = E[max(S_T - K, 0)] * exp(-r*T)
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))
import jax
import jax.numpy as jnp
from mc_tuner import AutoTuner


def gbm_european_call(key, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2, n_steps=252):
    """
    key: JAX PRNGKey for path randomness
    S0: initial stock price
    K: strike price
    T: time to maturity (years)
    r: risk-free rate
    sigma: volatility
    n_steps: number of discretization steps
    """
    dt = T / n_steps
    keys = jax.random.split(key, n_steps)
    drift = (r - 0.5 * sigma ** 2) * dt
    vol = sigma * jnp.sqrt(dt)

    def gbm_step(S, step_key):
        dW = jax.random.normal(step_key)
        S_new = S * jnp.exp(drift + vol * dW)
        return S_new, S_new

    S_T, _ = jax.lax.scan(gbm_step, S0, keys)
    return jnp.maximum(S_T - K, 0.0) * jnp.exp(-r * T)

def generate_inputs(master_key, batch_size):
    return jax.random.split(master_key, batch_size)

def test_unroll_gbm(batch_size=256, n_search=12):
    print("Test: scan_unroll | Geometric Brownian Motion")

    tuner = AutoTuner.from_fn(
        fn=gbm_european_call,
        generate_inputs=generate_inputs,
        fixed_kwargs={"S0": 100.0, "K": 100.0, "T": 1.0, "r": 0.05, "sigma": 0.2, "n_steps": 252},
        precision=0.5,
        n_repeats=5
    )

    best = tuner.run(method="grid", batch_size=batch_size, verbose=True)
    tuner.compare(batch_size=batch_size)

    assert "scan_unroll" in best, "scan_unroll should be in best params"
    assert best["scan_unroll"] in [1, 2, 4, 8, 16]
    print(f"Best scan_unroll = {best['scan_unroll']}")
    return best


if __name__ == "__main__":
    test_unroll_gbm()
