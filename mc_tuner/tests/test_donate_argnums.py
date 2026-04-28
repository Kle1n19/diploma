import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent))

import jax
import jax.numpy as jnp
import time


D = 4096
BATCH_SIZE = 64
N_LEAPFROG = 20
EPS = 0.1
N_WARM = 5
N_RUNS = 30

SEARCH_SPACE = [(), (0,), (1,), (0, 1)]

master_key = jax.random.PRNGKey(0)
q_batch = jax.random.normal(master_key, (BATCH_SIZE, D))
key_batch = jax.random.split(master_key, BATCH_SIZE)


def hmc_step(q, k):
    k, k_mom, k_acc = jax.random.split(k, 3)
    p = jax.random.normal(k_mom, (D,))

    def leapfrog(qp, _):
        q_, p_ = qp
        p_ = p_ - (EPS / 2) * q_
        q_ = q_ + EPS * p_
        p_ = p_ - (EPS / 2) * q_
        return (q_, p_), None

    (q_new, p_new), _ = jax.lax.scan(leapfrog, (q, p), None, length=N_LEAPFROG, unroll=1)

    H_old  = 0.5 * (jnp.sum(q ** 2)     + jnp.sum(p ** 2))
    H_new  = 0.5 * (jnp.sum(q_new ** 2) + jnp.sum(p_new ** 2))
    accept = jax.random.uniform(k_acc) < jnp.exp(jnp.minimum(0.0, H_old - H_new))
    return jnp.where(accept, q_new, q)


vmapped = jax.vmap(hmc_step)

print("Test: jit_donate_argnums | Hamiltonian Monte Carlo")
print(f"D={D}  batch={BATCH_SIZE}  q_buffer={D * BATCH_SIZE * 4 / 1024:.0f} KB")
print(f"Backend: {jax.default_backend()}")
print()

results = {}

for donate in SEARCH_SPACE:
    jitted = jax.jit(vmapped, donate_argnums=donate)
    for _ in range(N_WARM):
        jax.block_until_ready(jitted(jnp.array(q_batch), jnp.array(key_batch)))

    times = []
    for _ in range(N_RUNS):
        q0 = jnp.array(q_batch)
        k0 = jnp.array(key_batch)
        t0 = time.perf_counter()
        jax.block_until_ready(jitted(q0, k0))
        times.append(time.perf_counter() - t0)

    results[donate] = min(times) * 1000

baseline = results[()]
print(f"{'donate_argnums':<18} {'ms':>8} {'speedup':>10}")
for donate, t in results.items():
    marker = " *" if t == min(results.values()) else ""
    print(f"{str(donate):<18} {t:>8.3f} {baseline / t:>10.3f}x{marker}")

best = min(results, key=results.__getitem__)
print(f"\nBest donate_argnums = {best}")
