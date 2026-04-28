import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
from mc_tuner import AutoTuner


def gp_posterior_mean(key, length_scale=1.0, noise=0.1):
    """
    Works on TPU

    key: PRNGKey used to sample x_train and y_train
    length_scale: RBF kernel length scale
    noise: observation noise std
    """
    N = 1024
    D = 8
    k1, k2 = jax.random.split(key)
    x_train = jax.random.normal(k1, (N, D))
    y_train = jax.random.normal(k2, (N,))

    def rbf(x1, x2):
        diff = x1 - x2
        return jnp.exp(-0.5 * jnp.sum(diff ** 2) / length_scale ** 2)

    K = jax.vmap(lambda xi: jax.vmap(lambda xj: rbf(xi, xj))(x_train))(x_train)
    K_noisy = K + noise ** 2 * jnp.eye(N)

    alpha = jnp.linalg.solve(K_noisy, y_train)

    posterior_means = jnp.dot(K, alpha)
    return jnp.mean(posterior_means)


def generate_inputs(master_key, batch_size):
    return jax.random.split(master_key, batch_size)


def test_precision_gp(batch_size=16):
    print("Test: dot_precision | Gaussian Process Posterior Mean")

    tuner = AutoTuner.from_fn(
        fn=gp_posterior_mean,
        generate_inputs=generate_inputs,
        fixed_kwargs={"length_scale": 1.0, "noise": 0.1},
        precision=0.01,
        n_repeats=5,
    )

    best = tuner.run(method="grid", batch_size=batch_size, verbose=True)
    tuner.compare(batch_size=batch_size)

    assert "dot_precision" in best, "dot_precision should be in best params"
    assert best["dot_precision"] in [None, "high", "highest"]
    print(f"Best dot_precision = {best['dot_precision']}")
    return best


if __name__ == "__main__":
    test_precision_gp()
