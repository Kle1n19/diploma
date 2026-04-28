import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import jax

import jax.numpy as jnp

from mc_tuner import AutoTuner


def hutchinson_trace_estimator(key, W=None, n_steps=200, D=512):
    if W is None:
        W = jnp.eye(D) * 0.9

    v = jnp.sign(jax.random.normal(key, (D,)))
    v = jnp.where(v == 0.0, 1.0, v)

    @jax.checkpoint
    def matvec_step(Wv, _):
        Wv_new = jnp.dot(W, Wv)
        return Wv_new, jnp.dot(v, Wv_new)

    _, vTW_kv = jax.lax.scan(matvec_step, v, None, length=n_steps)

    return jnp.mean(vTW_kv)


def generate_inputs(master_key, batch_size):
    return jax.random.split(master_key, batch_size)


def test_checkpoint_hutchinson(batch_size=64):
    print("Test: checkpoint_policy | Hutchinson Trace Estimator")

    D = 512
    W = jnp.eye(D) * 0.9

    tuner = AutoTuner.from_fn(
        fn=hutchinson_trace_estimator,
        generate_inputs=generate_inputs,
        fixed_kwargs={"W": W, "n_steps": 200, "D": D},
        precision=0.01,
        n_repeats=5,
    )

    best = tuner.run(method="grid", batch_size=batch_size, verbose=True)
    tuner.compare(batch_size=batch_size)

    assert "checkpoint_policy" in best, "checkpoint_policy should be in best params"
    print(f"Best checkpoint_policy = {best['checkpoint_policy']}")
    return best


if __name__ == "__main__":
    test_checkpoint_hutchinson()
