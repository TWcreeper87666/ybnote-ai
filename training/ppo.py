"""PPO (clipped surrogate) trainer — RL_DESIGN.md §6/§9/§11. Chosen over
vanilla REINFORCE/plain actor-critic specifically because this project's
R-STDP attempts collapsed to permanent silence from unbounded update steps
(TRAIN_DIARY.md 2026-09-24) — PPO's clip bounds how far one update can move
the policy, the standard fix for exactly that failure mode.

Single-process rollout collection (§11), but from SEVERAL environments
(different charts) per update: train_rl.py keeps `--num-envs` episodes
alive and fills one RolloutBuffer per env, and update() computes GAE per
buffer then trains on their concatenation. With a single env, a 4096-step
update only ever saw one chart (full-chart episodes run 10k-40k steps, so
3-10 consecutive updates came from the same chart), which drifted the
policy toward whichever chart was current — see TRAIN_DIARY.md 2026-09-26
"v4 結果".
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from rl_env import TrailRLEnv
from rl_policy import ActorNet, CriticNet


class RunningNorm:
    """Running mean/std reward normalization (§8) — raw judgment rewards
    (-0.25 Wrong vs 0 Miss vs +1.0 Perfect vs small shaping deltas) are
    wildly different scales; PPO's advantage/value targets need this to
    keep gradient scale sane. Welford's online algorithm."""

    def __init__(self, eps: float = 1e-4):
        self.mean = 0.0
        self.var = 1.0
        self.count = eps

    def update(self, x: np.ndarray):
        batch_mean = float(np.mean(x))
        batch_var = float(np.var(x))
        batch_count = len(x)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / tot_count
        self.mean = new_mean
        self.var = m2 / tot_count
        self.count = tot_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RolloutBuffer:
    def __init__(self):
        self.obs: list[np.ndarray] = []
        self.raw_cursor: list[torch.Tensor] = []
        self.raw_action: list[torch.Tensor] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []

    def add(self, obs, raw_cursor, raw_action, log_prob, value, reward, done):
        self.obs.append(obs)
        self.raw_cursor.append(raw_cursor)
        self.raw_action.append(raw_action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def __len__(self):
        return len(self.obs)

    def clear(self):
        self.__init__()


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, last_value: float,
                 gamma: float = 0.995, lam: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    """§9: GAE(lambda). `values` has one entry per step (not T+1) — the
    bootstrap for the final step comes from `last_value` (0.0 at a true
    episode end, V(s_T) when a rollout was cut off mid-episode by the
    buffer size, standard PPO practice)."""
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    last_gae = 0.0
    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        next_nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_nonterminal - values[t]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        advantages[t] = last_gae
    returns = advantages + values
    return advantages, returns


class PPOTrainer:
    def __init__(
        self,
        actor: ActorNet,
        critic: CriticNet,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        gamma: float = 0.995,
        lam: float = 0.95,
        epochs: int = 4,
        minibatch_size: int = 256,
        max_grad_norm: float = 0.5,
    ):
        self.actor = actor
        self.critic = critic
        self.optimizer = torch.optim.Adam(
            list(actor.parameters()) + list(critic.parameters()), lr=lr
        )
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.gamma = gamma
        self.lam = lam
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_grad_norm = max_grad_norm
        self.reward_norm = RunningNorm()

    def collect_rollout(
        self,
        env: TrailRLEnv,
        n_steps: int,
        buffer: RolloutBuffer,
        obs: np.ndarray,
        progress_label: str = "",
    ) -> np.ndarray:
        """Runs up to n_steps, or until the episode ends (caller resets and
        keeps going for curriculum training across many short episodes —
        see train_rl.py). Returns the observation to resume from next
        call (a fresh env.reset() if the episode just ended)."""
        import time
        _t_start = time.time()
        label = f" {progress_label}" if progress_label else ""
        for i in range(n_steps):
            if i % 100 == 0:
                print(f"[boot{label} {time.time()-_t_start:6.2f}s in collect_rollout] step {i}/{n_steps} "
                      f"(env.step_idx={env.step_idx}/{env.length})", flush=True)
            obs_t = torch.from_numpy(obs).float()
            act = self.actor.act(obs_t)
            value = float(self.critic(obs_t.unsqueeze(0))[0])

            next_obs, reward, done, _info = env.step(act["cursor_delta"], act["action_type"])
            buffer.add(
                obs, act["raw_cursor"], act["raw_action"],
                act["log_prob"], value, reward, done,
            )
            obs = next_obs
            if done:
                return obs
        return obs

    def update(
        self,
        buffers: RolloutBuffer | list[RolloutBuffer],
        last_obs: np.ndarray | list[np.ndarray],
        last_done: bool | list[bool],
        train_actor: bool = True,
    ):
        """One PPO update over one or more per-env rollout buffers. Each
        buffer is its own contiguous trajectory (episode boundaries inside
        it are marked by `dones`), so GAE runs per buffer with that env's
        own bootstrap value, and only then are the buffers concatenated.

        train_actor=False fits only the critic (actor grads stay None, so
        Adam leaves it untouched) — a warmup after the observation/reward
        definition changed under a resumed critic, so the first actor steps
        aren't driven by advantages from a stale value function."""
        import time
        if isinstance(buffers, RolloutBuffer):
            buffers, last_obs, last_done = [buffers], [last_obs], [last_done]
        buffers = [b for b in buffers if len(b) > 0]
        total = sum(len(b) for b in buffers)
        print(f"[boot] update() starting with {total} buffered steps from {len(buffers)} env(s)", flush=True)
        _t_upd = time.time()

        all_rewards = np.concatenate([np.array(b.rewards, dtype=np.float32) for b in buffers])
        self.reward_norm.update(all_rewards)
        adv_parts, ret_parts, val_parts = [], [], []
        for b, obs_last, done_last in zip(buffers, last_obs, last_done):
            rewards_norm = self.reward_norm.normalize(np.array(b.rewards, dtype=np.float32))
            values = np.array(b.values, dtype=np.float32)
            dones = np.array(b.dones, dtype=bool)
            with torch.no_grad():
                last_value = 0.0 if done_last else float(
                    self.critic(torch.from_numpy(obs_last).float().unsqueeze(0))[0]
                )
            adv, ret = compute_gae(rewards_norm, values, dones, last_value, self.gamma, self.lam)
            adv_parts.append(adv)
            ret_parts.append(ret)
            val_parts.append(values)
        advantages = np.concatenate(adv_parts)
        returns = np.concatenate(ret_parts)
        values = np.concatenate(val_parts)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_t = torch.from_numpy(np.stack([o for b in buffers for o in b.obs])).float()
        raw_cursor_t = torch.stack([c for b in buffers for c in b.raw_cursor])
        raw_action_t = torch.stack([a for b in buffers for a in b.raw_action])
        old_log_probs_t = torch.tensor([lp for b in buffers for lp in b.log_probs], dtype=torch.float32)
        old_values_t = torch.from_numpy(values).float()
        advantages_t = torch.from_numpy(advantages).float()
        returns_t = torch.from_numpy(returns).float()

        N = total
        stats = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "n_updates": 0}
        for _ in range(self.epochs):
            perm = torch.randperm(N)
            for start in range(0, N, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]

                log_probs, entropy = self.actor.evaluate_actions(
                    obs_t[idx], raw_cursor_t[idx], raw_action_t[idx]
                )
                # Keep repeated minibatch epochs from turning a bounded
                # action's density change into an enormous ratio. PPO's
                # clipped surrogate assumes updates stay near the behavior
                # policy; +/-2 is already a roughly 7.4x trust-region guard.
                log_ratio = (log_probs - old_log_probs_t[idx]).clamp(-2.0, 2.0)
                ratio = log_ratio.exp()
                surr1 = ratio * advantages_t[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages_t[idx]
                policy_loss = -torch.min(surr1, surr2).mean()

                values_pred = self.critic(obs_t[idx])
                value_pred_clipped = old_values_t[idx] + (values_pred - old_values_t[idx]).clamp(
                    -self.clip_eps, self.clip_eps
                )
                value_loss_unclipped = (values_pred - returns_t[idx]).pow(2)
                value_loss_clipped = (value_pred_clipped - returns_t[idx]).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                entropy_loss = -entropy.mean()
                if train_actor:
                    loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                else:
                    loss = self.value_coef * value_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()), self.max_grad_norm
                )
                self.optimizer.step()

                stats["policy_loss"] += float(policy_loss)
                stats["value_loss"] += float(value_loss)
                stats["entropy"] += float(-entropy_loss)
                stats["n_updates"] += 1

        for k in ("policy_loss", "value_loss", "entropy"):
            stats[k] /= max(1, stats["n_updates"])
        print(f"[boot] update() done in {time.time()-_t_upd:.2f}s", flush=True)
        return stats
