import math
import jax
import jax.numpy as jnp
import numpy as np

from .base import TuningAgent

_S  = 6
_H  = 64
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
    new_m = jax.tree_util.tree_map(lambda mi, g: _B1 * mi + (1 - _B1) * g,       m, grads)
    new_v = jax.tree_util.tree_map(lambda vi, g: _B2 * vi + (1 - _B2) * g ** 2,  v, grads)
    bc1 = 1.0 - _B1 ** t
    bc2 = 1.0 - _B2 ** t
    new_p = jax.tree_util.tree_map(
        lambda p, mi, vi: p - _LR * (mi / bc1) / (jnp.sqrt(vi / bc2) + _EPS),
        params, new_m, new_v,
    )
    return new_p, new_m, new_v

@jax.jit
def _grpo_step(
    actor_p, pi_old_p, pi_ref_p,
    actor_m, actor_v,
    t,
    states, arms, advantages,
    clip_eps, beta,
):
    def loss_fn(ap):
        G = states.shape[0]

        logits = _fwd(ap, states)
        logits_old = _fwd(pi_old_p, states)
        logits_ref = _fwd(pi_ref_p, states)

        log_pi = jax.nn.log_softmax(logits)
        log_pi_old = jax.lax.stop_gradient(jax.nn.log_softmax(logits_old))
        log_pi_ref = jax.lax.stop_gradient(jax.nn.log_softmax(logits_ref))
        pi_cur = jax.nn.softmax(logits)
        pi_ref_ = jax.lax.stop_gradient(jax.nn.softmax(logits_ref))

        idx = jnp.arange(G)
        log_rho = log_pi[idx, arms] - log_pi_old[idx, arms]
        rho = jnp.exp(log_rho)

        surr1 = rho * advantages
        surr2 = jnp.clip(rho, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        policy_loss = -jnp.mean(jnp.minimum(surr1, surr2))

        kl = jnp.sum(pi_cur * (log_pi - log_pi_ref), axis=-1)
        kl_loss = beta * jnp.mean(kl)

        return policy_loss + kl_loss, {"policy_loss": policy_loss, "kl_loss": kl_loss}

    (total_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(actor_p)
    new_actor, new_m, new_v = _adam_step(actor_p, grads, actor_m, actor_v, t)
    return new_actor, new_m, new_v, total_loss, metrics

class GRPOAgent(TuningAgent):
    def __init__(
        self,
        n_actions: int,
        max_arms: int = 256,
        group_size: int = 4,
        clip_eps: float = 0.2,
        beta: float = 0.04,
        ref_interval: int = 5,
        warmup_steps: int = 8,
        seed: int = 0,
        entropy_fraction: float = 0.0,
    ):
        self.n_actions = n_actions
        self._n_arms = min(n_actions, max_arms)
        self._G = group_size
        self._clip_eps = jnp.array(clip_eps)
        self._beta = jnp.array(beta)
        self._ref_iv = ref_interval
        self._warmup = warmup_steps
        self._ent_frac = entropy_fraction

        rng = np.random.default_rng(seed)
        key = jax.random.PRNGKey(seed)

        perm = rng.permutation(n_actions)
        self._arm_to_action = perm[:self._n_arms].tolist()
        self._action_to_arm = {a: i for i, a in enumerate(self._arm_to_action)}

        dims = [_S, _H, _H, self._n_arms]
        k1, k2 = jax.random.split(key)
        self._actor_p = _init_net(k1, dims)
        self._pi_old = jax.tree_util.tree_map(jnp.copy, self._actor_p)
        self._pi_ref = jax.tree_util.tree_map(jnp.copy, self._actor_p)

        self._actor_m = _zeros_like(self._actor_p)
        self._actor_v = _zeros_like(self._actor_p)

        self._g_states: list = []
        self._g_arms: list = []
        self._g_rewards: list = []

        self._step = 0
        self._opt_step = 0
        self._groups_done = 0
        self._rng = rng
        self._reward_history: list[float] = []
        self._last_reward = 0.0
        self._last_arm = 0
        self._last_state = np.zeros(_S, dtype=np.float32)

    def _state(self) -> np.ndarray:
        rs = np.array(self._reward_history) if self._reward_history else np.array([0.0])
        mean_r = float(np.mean(rs))
        std_r = float(np.std(rs)) if len(rs) > 1 else 1.0
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

    def _flush_group(self) -> None:
        rewards = np.array(self._g_rewards, dtype=np.float32)
        mean_r = float(np.mean(rewards))
        std_r = float(np.std(rewards)) + 1e-8
        adv = (rewards - mean_r) / std_r

        states = jnp.array(np.stack(self._g_states))
        arms = jnp.array(self._g_arms, dtype=jnp.int32)
        advs = jnp.array(adv, dtype=jnp.float32)

        self._opt_step += 1
        (self._actor_p, self._actor_m, self._actor_v, _loss, _metrics) = _grpo_step(
            self._actor_p, self._pi_old, self._pi_ref,
            self._actor_m, self._actor_v,
            jnp.array(self._opt_step, dtype=jnp.int32),
            states, arms, advs,
            self._clip_eps, self._beta)

        self._groups_done += 1
        self._pi_old = jax.tree_util.tree_map(jnp.copy, self._actor_p)
        if self._groups_done % self._ref_iv == 0:
            self._pi_ref = jax.tree_util.tree_map(jnp.copy, self._actor_p)

        self._g_states.clear()
        self._g_arms.clear()
        self._g_rewards.clear()

    def select_action(self) -> int:
        s = self._state()
        self._last_state = s

        if self._step < self._warmup:
            arm = self._step % self._n_arms
        else:
            logits = np.array(_fwd(self._actor_p, jnp.array(s[None]))[0])
            probs  = np.exp(logits - logits.max())
            probs /= probs.sum()
            arm = int(self._rng.choice(self._n_arms, p=probs))

        self._last_arm = arm
        return self._arm_to_action[arm]

    def update(self, action: int, reward: float) -> None:
        arm = self._action_to_arm.get(action, self._last_arm)

        self._g_states.append(self._last_state.copy())
        self._g_arms.append(arm)
        self._g_rewards.append(reward)

        self._reward_history.append(reward)
        self._last_reward = reward
        self._step += 1

        if len(self._g_rewards) >= self._G:
            self._flush_group()

    def best_action(self) -> int:
        s = jnp.array(self._state()[None])
        q = jnp.minimum(_fwd(self._actor_p, s), _fwd(self._actor_p, s))[0]
        return self._arm_to_action[int(jnp.argmax(q))]

    def reset(self) -> None:
        key = jax.random.PRNGKey(0)
        dims = [_S, _H, _H, self._n_arms]
        self._actor_p = _init_net(key, dims)
        self._pi_old = jax.tree_util.tree_map(jnp.copy, self._actor_p)
        self._pi_ref = jax.tree_util.tree_map(jnp.copy, self._actor_p)
        self._actor_m = _zeros_like(self._actor_p)
        self._actor_v = _zeros_like(self._actor_p)
        self._g_states.clear()
        self._g_arms.clear()
        self._g_rewards.clear()
        self._step = 0
        self._opt_step = 0
        self._groups_done = 0
        self._reward_history = []
        self._last_reward = 0.0
        self._last_arm = 0
