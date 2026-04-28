import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import time


def timeit(fn, *args, n_warm=5, n_runs=30):
    for _ in range(n_warm):
        jax.block_until_ready(fn(*args))
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        times.append(time.perf_counter() - t0)
    return min(times) * 1000  # ms

N = 50_000
D = 128
key = jax.random.PRNGKey(0)
xs = jax.random.normal(key, (N, D))

print(f"Backend: {jax.default_backend()}")
print(f"Devices: {jax.devices()}")

ALPHA = jnp.float32(0.01)
init = jnp.zeros(D)

def step(carry, x):
    carry = (1 - ALPHA) * carry + ALPHA * x
    return carry, jnp.sum(carry)

fwd = jax.jit(lambda xs: jax.lax.scan(step, init, xs, reverse=False)[1])
rev = jax.jit(lambda xs: jax.lax.scan(step, init, xs, reverse=True )[1])

jax.block_until_ready(fwd(xs))
jax.block_until_ready(rev(xs))

t_fwd = timeit(fwd, xs)
t_rev = timeit(rev, xs)

winner = "forward" if t_fwd < t_rev else "backward"
ratio  = max(t_fwd, t_rev) / min(t_fwd, t_rev)

print(f"reverse=False (forward):  {t_fwd:7.1f} ms")
print(f"reverse=True  (backward): {t_rev:7.1f} ms")
print(f"Winner: {winner}  ({ratio:.2f}x faster)")
