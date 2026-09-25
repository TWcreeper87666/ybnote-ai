"""Actor-critic nets for the PPO agent — RL_DESIGN.md §4/§5.

Actor reuses ChartPolicyNet's proven shared-trunk MLP *shape* (see
dl_model.py) as the feature extractor, with new stochastic heads instead of
direct regression/BCE-classification outputs. Critic is a separate,
decoupled MLP (§5 — sharing early layers with the actor is common but
couples actor/critic instability, and this project's history (R-STDP
collapse) argues for the safer default)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Bernoulli, Normal

import config
from obstacles import MAX_OBSTACLES, OBSTACLE_FEATURE_DIM
from rl_env import HISTORY_STRIDES, OWN_STATE_DIM

# Bounds are relative to the action's own scale, not a generic RL default:
# cursor_mean is tanh-squashed to +/-config.CURSOR_MAX_SPEED_NORM_PER_STEP
# (0.05 in normalized 0..1 coordinate space — see rl_env.py). A LOG_STD_MAX
# of 1.0 (std up to e^1=2.72) was a 54x mismatch against that 0.05 mean
# range — TrailRLEnv.step() then hard-clamps the sampled delta to the same
# 0.05 speed cap regardless of what the mean said, so the policy could
# sample effectively pure noise and never learn to aim. Observed effect on
# a real run: entropy climbed monotonically for 300 iterations (3.7->4.9,
# near the heads' combined max) with holdout hits stuck at 0/634 the whole
# time — PPO's entropy bonus had nothing to push back against since the
# noise was getting clipped away rather than penalized by a failed hit.
# -2.0 (std up to ~0.135, a few times the max mean magnitude) keeps early
# exploration meaningfully directional instead of saturating the clamp.
LOG_STD_MIN = -5.0
LOG_STD_MAX = -2.0


class ActorNet(nn.Module):
    """object_dim/obstacle_dim below are for ONE stacked history slot; the
    net's actual input is `len(HISTORY_STRIDES)` copies of
    (object features + obstacle features + own-state) concatenated — see
    rl_env.obs_dim(). action_trunk (timing-only) still reads ONLY the
    object-feature slice of every history slot, not obstacle/own-state —
    same cross-talk reasoning as dl_model.py's ChartPolicyNet."""

    def __init__(self, max_objects: int, features_per_obj: int, hidden: int = 256):
        super().__init__()
        self.max_objects = max_objects
        self.features_per_obj = features_per_obj
        self.object_dim = max_objects * features_per_obj
        self.obstacle_dim = MAX_OBSTACLES * OBSTACLE_FEATURE_DIM
        self.per_step_dim = self.object_dim + self.obstacle_dim + OWN_STATE_DIM
        self.n_hist = len(HISTORY_STRIDES)
        self.input_dim = self.per_step_dim * self.n_hist

        self.trunk = nn.Sequential(
            nn.Linear(self.input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # (mean_dx, mean_dy, log_std_dx, log_std_dy)
        self.cursor_head = nn.Linear(hidden, 4)
        self.trail_head = nn.Linear(hidden, 1)

        action_input_dim = self.object_dim * self.n_hist
        self.action_trunk = nn.Sequential(
            nn.Linear(action_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.attack_head = nn.Linear(hidden, 1)

    def _object_slices(self, x: torch.Tensor) -> torch.Tensor:
        """Pull just the per-slot object-feature chunk out of the full
        stacked observation, for every history slot, and re-concatenate —
        mirrors how rl_env._raw_features_vec lays out one slot as
        [object_feats | obstacle_feats | own_state]."""
        chunks = []
        for i in range(self.n_hist):
            start = i * self.per_step_dim
            chunks.append(x[..., start : start + self.object_dim])
        return torch.cat(chunks, dim=-1)

    def forward(self, x: torch.Tensor):
        """x: [batch, input_dim]. Returns dict of distribution params."""
        h = self.trunk(x)
        cursor_out = self.cursor_head(h)
        mean = torch.tanh(cursor_out[..., :2]) * config.CURSOR_MAX_SPEED_NORM_PER_STEP
        log_std = torch.clamp(cursor_out[..., 2:], LOG_STD_MIN, LOG_STD_MAX)
        trail_logit = self.trail_head(h).squeeze(-1)

        object_features = self._object_slices(x)
        attack_logit = self.attack_head(self.action_trunk(object_features)).squeeze(-1)

        return {
            "cursor_mean": mean,
            "cursor_log_std": log_std,
            "trail_logit": trail_logit,
            "attack_logit": attack_logit,
        }

    def distributions(self, x: torch.Tensor):
        out = self.forward(x)
        cursor_dist = Normal(out["cursor_mean"], out["cursor_log_std"].exp())
        trail_dist = Bernoulli(logits=out["trail_logit"])
        attack_dist = Bernoulli(logits=out["attack_logit"])
        return cursor_dist, trail_dist, attack_dist

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False):
        """Returns (action dict of raw numpy/py values, log_prob sum,
        entropy sum) for ONE observation (unbatched, x: [input_dim])."""
        x = x.unsqueeze(0)
        cursor_dist, trail_dist, attack_dist = self.distributions(x)
        if deterministic:
            cursor = cursor_dist.mean
            trail = (torch.sigmoid(trail_dist.logits) > 0.5).float()
            attack = (torch.sigmoid(attack_dist.logits) > 0.5).float()
        else:
            cursor = cursor_dist.sample()
            trail = trail_dist.sample()
            attack = attack_dist.sample()

        log_prob = (
            cursor_dist.log_prob(cursor).sum(-1)
            + trail_dist.log_prob(trail)
            + attack_dist.log_prob(attack)
        )
        entropy = cursor_dist.entropy().sum(-1) + trail_dist.entropy() + attack_dist.entropy()

        return {
            "cursor_delta": (float(cursor[0, 0]), float(cursor[0, 1])),
            "trail_held": bool(trail[0].item() > 0.5),
            "attack_raw": bool(attack[0].item() > 0.5),
            "raw_cursor": cursor[0],
            "raw_trail": trail[0],
            "raw_attack": attack[0],
            "log_prob": float(log_prob[0]),
            "entropy": float(entropy[0]),
        }

    def evaluate_actions(self, x: torch.Tensor, raw_cursor: torch.Tensor, raw_trail: torch.Tensor, raw_attack: torch.Tensor):
        """Batched — for the PPO update. Returns (log_prob [B], entropy [B])."""
        cursor_dist, trail_dist, attack_dist = self.distributions(x)
        log_prob = (
            cursor_dist.log_prob(raw_cursor).sum(-1)
            + trail_dist.log_prob(raw_trail)
            + attack_dist.log_prob(raw_attack)
        )
        entropy = cursor_dist.entropy().sum(-1) + trail_dist.entropy() + attack_dist.entropy()
        return log_prob, entropy


class CriticNet(nn.Module):
    """Own trunk, not shared with ActorNet — see module docstring."""

    def __init__(self, max_objects: int, features_per_obj: int, hidden: int = 256):
        super().__init__()
        input_dim = (max_objects * features_per_obj + MAX_OBSTACLES * OBSTACLE_FEATURE_DIM + OWN_STATE_DIM) * len(HISTORY_STRIDES)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
