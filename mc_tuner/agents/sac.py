"""
SACAgent: Discrete Soft Actor-Critic in pure JAX (PureJaxRL style).
"""

import math
import jax
import jax.numpy as jnp
import numpy as np
from .base import TuningAgent

_S = 6
_H = 64
_GAMMA = 0.99
_TAU = 0.005
_LR = 3e-4
_B1, _B2, _EPS = 0.9, 0.999, 1e-8



def _init_net(key, dims: list[int]) -> dict:
    layers = []
    for k, (n_in, n_out) in zip(jax.random.split(key, len(dims) - 1), zip(dims[:-1], dims[1:])):
        std = math.sqrt(2.0 / n_in)
        layers.append({"W": jax.random.normal(k, (n_in, n_out)) * std, "b": jnp.zeros(n_out)})
    return {"layers": layers}


def _fwd(net: dict, x: jnp.ndarray) -> jnp.ndarray:
    ls = net["layers"]
    for i, l in enumerate(ls):
        x = x @ l["W"] + l["b"]
        if i < len(ls) - 1:
            x = jax.nn.relu(x)
    return x


def _zeros_like(net: dict) -> dict:
    return jax.tree_util.tree_map(jnp.zeros_like, net)


def _adam_step(params, grads, m, v, t: int) -> tuple:
    new_m = jax.tree_util.tree_map(lambda mi, g: _B1 * mi + (1 - _B1) * g, m, grads)
    new_v = jax.tree_util.tree_map(lambda vi, g: _B2 * vi + (1 - _B2) * g ** 2, v, grads)
    bc1 = 1.0 - _B1 ** t
    bc2 = 1.0 - _B2 ** t
    new_p = jax.tree_util.tree_map(lambda p, mi, vi: p - _LR * (mi / bc1) / (jnp.sqrt(vi / bc2) + _EPS), params, new_m, new_v)
    return new_p, new_m, new_v


def _soft_update(tgt, src) -> dict:
    return jax.tree_util.tree_map(lambda t, s: (1 - _TAU) * t + _TAU * s, tgt, src)


def _make_train_step(target_entropy: float):

    @jax.jit
    def step(actor_p, q1_p, q2_p, log_alpha,
             actor_m, actor_v, q1_m, q1_v, q2_m, q2_v,
             alpha_m, alpha_v,
             tgt_actor, tgt_q1, tgt_q2,
             t, states, actions, rewards, next_states):
        B = states.shape[0]
        alpha = jnp.exp(log_alpha)

        pi_ns = jax.nn.softmax(_fwd(tgt_actor, next_states))
        log_pi_ns = jnp.log(pi_ns + 1e-8)
        q_ns = jnp.minimum(_fwd(tgt_q1, next_states), _fwd(tgt_q2, next_states))
        v_ns = jnp.sum(pi_ns * (q_ns - alpha * log_pi_ns), axis=-1)
        target_q = jax.lax.stop_gradient(rewards + _GAMMA * v_ns)

        def critic_loss(q1p, q2p):
            q1_sa = _fwd(q1p, states)[jnp.arange(B), actions]
            q2_sa = _fwd(q2p, states)[jnp.arange(B), actions]
            return jnp.mean((q1_sa - target_q) ** 2 + (q2_sa - target_q) ** 2)

        c_loss, (gq1, gq2) = jax.value_and_grad(critic_loss, argnums=(0, 1))(q1_p, q2_p)

        def actor_loss(ap):
            pi = jax.nn.softmax(_fwd(ap, states))
            log_pi = jnp.log(pi + 1e-8)
            q_sg = jax.lax.stop_gradient(jnp.minimum(_fwd(q1_p, states), _fwd(q2_p, states)))
            return jnp.mean(jnp.sum(pi * (alpha * log_pi - q_sg), axis=-1))

        a_loss, g_actor = jax.value_and_grad(actor_loss)(actor_p)

        def alpha_loss(la):
            pi_sg = jax.lax.stop_gradient(jax.nn.softmax(_fwd(actor_p, states)))
            H = -jnp.sum(pi_sg * jnp.log(pi_sg + 1e-8), axis=-1)
            return jnp.mean(-jnp.exp(la) * (H - target_entropy))

        al_loss, g_alpha_scalar = jax.value_and_grad(alpha_loss)(log_alpha)

        new_q1, new_q1_m, new_q1_v = _adam_step(q1_p, gq1, q1_m, q1_v, t)
        new_q2, new_q2_m, new_q2_v = _adam_step(q2_p, gq2, q2_m, q2_v, t)
        new_actor, new_actor_m, new_actor_v = _adam_step(actor_p, g_actor, actor_m, actor_v, t)

        new_alpha_m = _B1 * alpha_m + (1 - _B1) * g_alpha_scalar
        new_alpha_v = _B2 * alpha_v + (1 - _B2) * g_alpha_scalar ** 2
        bc1 = 1.0 - _B1 ** t
        bc2 = 1.0 - _B2 ** t
        new_log_alpha = log_alpha - _LR * (new_alpha_m / bc1) / (jnp.sqrt(new_alpha_v / bc2) + _EPS)

        new_tgt_actor = _soft_update(tgt_actor, new_actor)
        new_tgt_q1 = _soft_update(tgt_q1,    new_q1)
        new_tgt_q2 = _soft_update(tgt_q2,    new_q2)

        return (new_actor,  new_q1, new_q2, new_log_alpha,
                new_actor_m, new_actor_v,
                new_q1_m, new_q1_v,
                new_q2_m, new_q2_v,
                new_alpha_m, new_alpha_v,
                new_tgt_actor, new_tgt_q1, new_tgt_q2,
                {"critic_loss": c_loss, "actor_loss": a_loss, "alpha_loss": al_loss, "alpha": jnp.exp(new_log_alpha)})

    return step

class SACAgent(TuningAgent):
    """
    Parameters
    ----------
    n_actions: total number of configs in the search space
    max_arms: max configs the network tracks explicitly (default 256)
    warmup_steps: random exploration before policy-guided selection
    batch_size: replay batch size per gradient step
    grad_steps: gradient updates per environment step
    buffer_capacity: replay buffer size
    min_buffer: minimum transitions before training starts
    seed: JAX + numpy RNG seed
    entropy_fraction: target entropy as fraction
    """

    def __init__(
        self,
        n_actions,
        max_arms = 256,
        warmup_steps = 10,
        batch_size = 32,
        grad_steps = 4,
        buffer_capacity = 10_000,
        min_buffer = 16,
        seed = 0,
        entropy_fraction = 0.98
    ):
        self.n_actions = n_actions
        self._n_arms = min(n_actions, max_arms)
        self._warmup = warmup_steps
        self._bs = batch_size
        self._grad_steps = grad_steps
        self._min_buf = min_buffer
        self._cap = buffer_capacity

        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)
        perm = rng.permutation(n_actions)
        self._arm_to_action = perm[:self._n_arms].tolist()
        self._action_to_arm = {a: i for i, a in enumerate(self._arm_to_action)}

        target_entropy = float(entropy_fraction * math.log(self._n_arms))
        self._train_step = _make_train_step(target_entropy)

        dims = [_S, _H, _H, self._n_arms]
        keys = jax.random.split(key, 3)
        self._actor_p = _init_net(keys[0], dims)
        self._q1_p = _init_net(keys[1], dims)
        self._q2_p = _init_net(keys[2], dims)
        self._log_alpha = jnp.array(0.0)

        self._actor_m = _zeros_like(self._actor_p)
        self._actor_v = _zeros_like(self._actor_p)
        self._q1_m = _zeros_like(self._q1_p)
        self._q1_v = _zeros_like(self._q1_p)
        self._q2_m = _zeros_like(self._q2_p)
        self._q2_v = _zeros_like(self._q2_p)
        self._alpha_m = jnp.array(0.0)
        self._alpha_v = jnp.array(0.0)

        self._tgt_actor = jax.tree_util.tree_map(jnp.copy, self._actor_p)
        self._tgt_q1 = jax.tree_util.tree_map(jnp.copy, self._q1_p)
        self._tgt_q2 = jax.tree_util.tree_map(jnp.copy, self._q2_p)

        self._buf_s = np.zeros((buffer_capacity, _S), dtype=np.float32)
        self._buf_a = np.zeros(buffer_capacity, dtype=np.int32)
        self._buf_r = np.zeros(buffer_capacity, dtype=np.float32)
        self._buf_ns = np.zeros((buffer_capacity, _S), dtype=np.float32)
        self._buf_ptr, self._buf_size = 0, 0

        self._step = 0
        self._reward_history = []
        self._last_reward = 0.0
        self._last_arm = 0
        self._opt_step = 0
        self._rng = rng

    def _state(self) -> np.ndarray:
        rs = np.array(self._reward_history) if self._reward_history else np.array([0.0])
        mean_r = float(np.mean(rs))
        std_r = float(np.std(rs))  if len(rs) > 1 else 1.0
        best_r = float(np.max(rs))
        scale = max(abs(mean_r), abs(best_r), std_r, 1.0)
        return np.array([
            min(self._step / max(self._warmup * 4, 1), 1.0),
            mean_r / scale,
            std_r  / scale,
            best_r / scale,
            self._last_reward / scale,
            self._last_arm / max(self._n_arms - 1, 1),
        ], dtype=np.float32)

    def _store(self, s, a_arm, r, ns):
        i = self._buf_ptr
        self._buf_s[i] = s
        self._buf_a[i] = a_arm
        self._buf_r[i] = r
        self._buf_ns[i] = ns
        self._buf_ptr = (i + 1) % self._cap
        self._buf_size = min(self._buf_size + 1, self._cap)

    def _sample_batch(self):
        idx = self._rng.integers(0, self._buf_size, size=self._bs)
        return (jnp.array(self._buf_s[idx]), jnp.array(self._buf_a[idx]), jnp.array(self._buf_r[idx]), jnp.array(self._buf_ns[idx]))

    def select_action(self) -> int:
        s = self._state()
        self._last_state = s

        if self._step < self._warmup:
            arm = self._step % self._n_arms
        else:
            logits = np.array(_fwd(self._actor_p, jnp.array(s[None]))[0])
            gumbel = -np.log(-np.log(self._rng.random(self._n_arms) + 1e-8) + 1e-8)
            arm = int(np.argmax(logits + gumbel))

        self._last_arm = arm
        return self._arm_to_action[arm]

    def update(self, action: int, reward: float) -> None:
        arm = self._action_to_arm.get(action, self._last_arm)
        s = self._last_state
        ns = self._state()

        self._reward_history.append(reward)
        self._last_reward = reward
        self._step += 1

        self._store(s, arm, reward, ns)

        if self._buf_size >= self._min_buf:
            for _ in range(self._grad_steps):
                self._opt_step += 1
                batch = self._sample_batch()
                (self._actor_p,  self._q1_p,    self._q2_p,   self._log_alpha,
                 self._actor_m,  self._actor_v,
                 self._q1_m,     self._q1_v,
                 self._q2_m,     self._q2_v,
                 self._alpha_m,  self._alpha_v,
                 self._tgt_actor, self._tgt_q1, self._tgt_q2,
                 _metrics) = self._train_step(
                    self._actor_p, self._q1_p, self._q2_p, self._log_alpha,
                    self._actor_m, self._actor_v,
                    self._q1_m,    self._q1_v,
                    self._q2_m,    self._q2_v,
                    self._alpha_m, self._alpha_v,
                    self._tgt_actor, self._tgt_q1, self._tgt_q2,
                    jnp.array(self._opt_step, dtype=jnp.int32),
                    *batch,
                )

    def best_action(self) -> int:
        s      = jnp.array(self._state()[None])
        q      = jnp.minimum(_fwd(self._q1_p, s), _fwd(self._q2_p, s))[0]
        best_arm = int(jnp.argmax(q))
        return self._arm_to_action[best_arm]

    def reset(self) -> None:
        key = jax.random.PRNGKey(0)
        dims = [_S, _H, _H, self._n_arms]
        ks = jax.random.split(key, 3)
        self._actor_p = _init_net(ks[0], dims)
        self._q1_p = _init_net(ks[1], dims)
        self._q2_p = _init_net(ks[2], dims)
        self._log_alpha = jnp.array(0.0)
        self._actor_m = _zeros_like(self._actor_p)
        self._actor_v = _zeros_like(self._actor_p)
        self._q1_m = _zeros_like(self._q1_p)
        self._q1_v = _zeros_like(self._q1_p)
        self._q2_m = _zeros_like(self._q2_p)
        self._q2_v = _zeros_like(self._q2_p)
        self._alpha_m = jnp.array(0.0)
        self._alpha_v = jnp.array(0.0)
        self._tgt_actor = jax.tree_util.tree_map(jnp.copy, self._actor_p)
        self._tgt_q1 = jax.tree_util.tree_map(jnp.copy, self._q1_p)
        self._tgt_q2 = jax.tree_util.tree_map(jnp.copy, self._q2_p)
        self._buf_ptr = 0
        self._buf_size = 0
        self._step = 0
        self._reward_history = []
        self._last_reward = 0.0
        self._last_arm = 0
        self._opt_step = 0
