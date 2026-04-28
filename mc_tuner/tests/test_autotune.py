import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import jax

import jax.numpy as jnp

from mc_tuner import AutoTuner


def qmc_matmul_integral(sample, projection=None, target_dim=64):
    D = sample.shape[0]
    if projection is None:
        projection = jnp.eye(target_dim, D)

    projected = jnp.matmul(projection, sample)

    return jnp.mean(jnp.sin(projected) + jnp.cos(projected))


def generate_inputs(master_key, batch_size):
    D = 256
    return jax.random.normal(master_key, (batch_size, D))


def test_autotune_qmc(batch_size=256):
    print("Test: autotune_level | Quasi-Monte Carlo Integration")

    D, target_dim = 256, 128
    proj_key = jax.random.PRNGKey(99)
    projection = jax.random.normal(proj_key, (target_dim, D)) / jnp.sqrt(D)

    tuner = AutoTuner.from_fn(
        fn=qmc_matmul_integral,
        generate_inputs=generate_inputs,
        fixed_kwargs={"projection": projection, "target_dim": target_dim},
        precision=0.001,
        n_repeats=5,
        include_autotune=True,
    )

    best = tuner.run(method="random", n=5, batch_size=batch_size, verbose=True)
    tuner.compare(batch_size=batch_size)

    assert "autotune_level" in best, "autotune_level should be in best params"
    assert best["autotune_level"] in [0, 1, 2, 3, 4]
    print(f"Best autotune_level = {best['autotune_level']}")
    return best


if __name__ == "__main__":
    test_autotune_qmc()
