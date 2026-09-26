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
from torch.distributions import (
    AffineTransform,
    Bernoulli,
    Categorical,
    Normal,
    TanhTransform,
    TransformedDistribution,
)

import config
from obstacles import MAX_OBSTACLES, OBSTACLE_FEATURE_DIM
from rl_env import (
    EXTRA_OBJECT_FEATURES,
    HISTORY_STRIDES,
    NUM_PRESS,
    OWN_STATE_DIM,
    PRESS_CLICK,
    PRESS_KEY,
    PRESS_NONE,
)

# Checkpoint tag for the current action space; checkpoints without it
# predate the CLICK/KEY split and trail toggle (one 3-way categorical,
# 0=no-op 1=attack 2=trail-held-this-tick, key-vs-click hard-coded from the
# target's label).
ACTION_SPACE = "press_click_key+trail_toggle"

# own-state layout (rl_env._raw_features_vec): cursor_x, cursor_y,
# trail_held, ticks_since_attack, reach.
_OWN_STATE_TRAIL_HELD = 2

# Trail-toggle prior. A stroke is started with logit TRAIL_START_BIAS and,
# once held, released with logit TRAIL_START_BIAS + TRAIL_RELEASE_OFFSET
# (the offset is a learned scalar). -6 ~= 0.25%/tick, so a random start
# happens ~0.5x per second of chart instead of drowning notes in trail
# Wrongs; -3.5 while held ~= 3%/tick, a ~150ms average exploratory stroke.
TRAIL_START_BIAS = -6.0
TRAIL_RELEASE_OFFSET = 2.5

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
# (These now bound the LATENT Normal before tanh, so they are independent of
# the output scale set by CURSOR_COMPONENT_LIMIT below.)
LOG_STD_MIN = -5.0
LOG_STD_MAX = -2.0
# The cursor action is a fraction of the world speed ceiling (rl_env.step
# scales it by the chart's reach), so the per-component limit is unitless:
# a diagonal at full tanh saturation is exactly the ceiling.
CURSOR_COMPONENT_LIMIT = 1.0 / (2.0 ** 0.5)
CURSOR_ACTION_EPS = 1e-6


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
        # (latent_dx, latent_dy, log_std_dx, log_std_dy). The latent cursor
        # distribution is squashed to the environment's speed limit below.
        self.cursor_head = nn.Linear(hidden, 4)
        action_input_dim = self.object_dim * self.n_hist
        self.action_trunk = nn.Sequential(
            nn.Linear(action_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        own_state_input_dim = OWN_STATE_DIM * self.n_hist
        self.attack_state_trunk = nn.Sequential(
            nn.Linear(own_state_input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # rl_env's action = (press, trail toggle), three heads:
        # - action_head, WHEN to press: 0=no-op / 1=press.
        # - input_path_head, HOW a press is delivered: Bernoulli over mouse
        #   click (0) vs the target's bound key (1). Factored rather than a
        #   flat none/click/key head so a deterministic (argmax) policy keeps
        #   pressing whenever P(press) beats P(no-op); a flat head would
        #   split that mass and could drop both below no-op.
        # - trail_toggle_head: Bernoulli "flip the held state" — independent
        #   of the press, so a click/key can land mid-stroke. Reads the full
        #   trunk too (a stroke is a spatial decision: where the sweep will
        #   go and what it crosses), plus a learned offset applied while a
        #   stroke is held so start and release get their own priors.
        self.action_head = nn.Linear(hidden, 2)
        self.input_path_head = nn.Linear(hidden, 1)
        self.trail_toggle_head = nn.Linear(2 * hidden, 1)
        self.trail_release_offset = nn.Parameter(torch.tensor(TRAIL_RELEASE_OFFSET))
        # A 5ms control loop is mostly no-op. Starting from a uniform policy
        # would press on half of all ticks and drown the sparse positive
        # reward in Wrong judgments. Keep a small action prior for
        # exploration; PPO can move it.
        with torch.no_grad():
            self.action_head.bias.copy_(torch.tensor([3.0, -1.5]))
            self.input_path_head.weight.zero_()
            self.input_path_head.bias.zero_()
            self.trail_toggle_head.weight.zero_()
            self.trail_toggle_head.bias.fill_(TRAIL_START_BIAS)

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

    def _own_state_slices(self, x: torch.Tensor) -> torch.Tensor:
        chunks = []
        for i in range(self.n_hist):
            start = i * self.per_step_dim + self.object_dim + self.obstacle_dim
            chunks.append(x[..., start : start + OWN_STATE_DIM])
        return torch.cat(chunks, dim=-1)

    def forward(self, x: torch.Tensor):
        """x: [batch, input_dim]. Returns dict of distribution params."""
        h = self.trunk(x)
        cursor_out = self.cursor_head(h)
        log_std = torch.clamp(cursor_out[..., 2:], LOG_STD_MIN, LOG_STD_MAX)
        object_features = self._object_slices(x)
        own_state = self._own_state_slices(x)
        attack_features = self.action_trunk(object_features) + self.attack_state_trunk(own_state)
        action_logits = self.action_head(attack_features)
        input_path_logit = self.input_path_head(attack_features).squeeze(-1)
        trail_held_now = own_state[..., _OWN_STATE_TRAIL_HELD]  # history slot 0 = current tick
        trail_toggle_logit = (
            self.trail_toggle_head(torch.cat([h, attack_features], dim=-1)).squeeze(-1)
            + trail_held_now * self.trail_release_offset
        )

        return {
            "cursor_loc": cursor_out[..., :2],
            "cursor_log_std": log_std,
            "action_logits": action_logits,
            "input_path_logit": input_path_logit,
            "trail_toggle_logit": trail_toggle_logit,
        }

    def distributions(self, x: torch.Tensor):
        out = self.forward(x)
        cursor_dist = TransformedDistribution(
            Normal(out["cursor_loc"], out["cursor_log_std"].exp()),
            [
                TanhTransform(cache_size=1),
                AffineTransform(loc=0.0, scale=CURSOR_COMPONENT_LIMIT),
            ],
        )
        action_dists = {
            "when": Categorical(logits=out["action_logits"]),
            "path": Bernoulli(logits=out["input_path_logit"]),
            "toggle": Bernoulli(logits=out["trail_toggle_logit"]),
        }
        return cursor_dist, action_dists

    @staticmethod
    def _action_log_prob_entropy(dists: dict, env_action: torch.Tensor):
        """Joint log-prob / entropy of an env action (rl_env.encode_action)
        under the when/how/toggle heads. The how-head only contributes on a
        press."""
        press = env_action % NUM_PRESS
        toggle = (env_action // NUM_PRESS).float()
        is_press = (press != PRESS_NONE).long()
        is_key = (press == PRESS_KEY).float()
        log_prob = (
            dists["when"].log_prob(is_press)
            + is_press.float() * dists["path"].log_prob(is_key)
            + dists["toggle"].log_prob(toggle)
        )
        entropy = (
            dists["when"].entropy()
            + dists["when"].probs[..., 1] * dists["path"].entropy()
            + dists["toggle"].entropy()
        )
        return log_prob, entropy

    @torch.no_grad()
    def act(self, x: torch.Tensor, deterministic: bool = False):
        """Returns (action dict of raw numpy/py values, log_prob sum,
        entropy sum) for ONE observation (unbatched, x: [input_dim])."""
        x = x.unsqueeze(0)
        cursor_dist, dists = self.distributions(x)
        if deterministic:
            cursor = torch.tanh(cursor_dist.base_dist.loc) * CURSOR_COMPONENT_LIMIT
            when = dists["when"].probs.argmax(dim=-1)
            # Tie (a freshly migrated how-head, logit exactly 0) goes to the
            # key, which is what the pre-split ATTACK always did.
            use_key = dists["path"].logits >= 0
            toggle = dists["toggle"].logits > 0
        else:
            cursor = cursor_dist.sample()
            when = dists["when"].sample()
            use_key = dists["path"].sample() > 0.5
            toggle = dists["toggle"].sample() > 0.5
        press = torch.where(
            when == 0,
            torch.full_like(when, PRESS_NONE),
            torch.where(use_key, torch.full_like(when, PRESS_KEY), torch.full_like(when, PRESS_CLICK)),
        )
        action = press + NUM_PRESS * toggle.long()

        # Keep the stored rollout action strictly inside the inverse-tanh
        # domain as well; otherwise old_log_prob can become NaN before PPO
        # even starts its first update.
        cursor = cursor.clamp(
            -CURSOR_COMPONENT_LIMIT + CURSOR_ACTION_EPS,
            CURSOR_COMPONENT_LIMIT - CURSOR_ACTION_EPS,
        )
        action_log_prob, action_entropy = self._action_log_prob_entropy(dists, action)
        log_prob = cursor_dist.log_prob(cursor).sum(-1) + action_log_prob
        # TransformedDistribution does not expose entropy(); the latent
        # Normal entropy is a stable approximation for the PPO bonus.
        cursor_entropy = cursor_dist.base_dist.entropy().sum(-1)
        entropy = cursor_entropy + action_entropy

        return {
            "cursor_delta": (float(cursor[0, 0]), float(cursor[0, 1])),
            "action_type": int(action[0].item()),
            "raw_cursor": cursor[0],
            "raw_action": action[0],
            "log_prob": float(log_prob[0]),
            "entropy": float(entropy[0]),
        }

    def evaluate_actions(self, x: torch.Tensor, raw_cursor: torch.Tensor, raw_action: torch.Tensor):
        """Batched — for the PPO update. Returns (log_prob [B], entropy [B])."""
        cursor_dist, dists = self.distributions(x)
        # A float32 tanh sample can round exactly to +/-1. The inverse tanh
        # inside TransformedDistribution.log_prob is undefined at that
        # boundary, so keep PPO's replayed action strictly interior.
        raw_cursor = raw_cursor.clamp(
            -CURSOR_COMPONENT_LIMIT + CURSOR_ACTION_EPS,
            CURSOR_COMPONENT_LIMIT - CURSOR_ACTION_EPS,
        )
        raw_action = raw_action.long().reshape(-1)
        action_log_prob, action_entropy = self._action_log_prob_entropy(dists, raw_action)
        log_prob = cursor_dist.log_prob(raw_cursor).sum(-1) + action_log_prob
        cursor_entropy = cursor_dist.base_dist.entropy().sum(-1)
        entropy = cursor_entropy + action_entropy
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


def _relayout_columns(weight: torch.Tensor, n_hist: int, segments: list[tuple[int, int]]) -> torch.Tensor:
    """Re-lay an input weight matrix whose columns are n_hist copies of a
    slot made of consecutive segments, each growing from old_len to new_len
    columns (appended at the segment's end). Old columns keep their weights
    and every new column gets zero weight, so the layer's output is
    unchanged for any input."""
    old_slot = sum(old for old, _ in segments)
    new_slot = sum(new for _, new in segments)
    if weight.shape[1] != old_slot * n_hist:
        raise ValueError(f"expected {old_slot * n_hist} input columns, got {weight.shape[1]}")
    out = weight.new_zeros(weight.shape[0], new_slot * n_hist)
    for h in range(n_hist):
        src, dst = h * old_slot, h * new_slot
        for old, new in segments:
            out[:, dst : dst + old] = weight[:, src : src + old]
            src += old
            dst += new
    return out


# Encoded per-object features in frames.csv: proximity, x, y, keybind flag,
# then the key one-hot (config.KEY_VOCAB). The env appends
# EXTRA_OBJECT_FEATURES more.
CHART_FEATURES_PER_OBJ = 4 + len(config.KEY_VOCAB)


def _pad_object_features(ckpt: dict) -> dict:
    """A current-action-space checkpoint saved before some env-appended
    object column existed (e.g. v4-v7 predate hit_timing_at) gets
    zero-weight columns for the missing ones, appended after each object's
    existing columns like the env lays them out, so its outputs are
    unchanged until training uses them."""
    target = CHART_FEATURES_PER_OBJ + EXTRA_OBJECT_FEATURES
    old_fpo = ckpt["features_per_obj"]
    if old_fpo == target:
        return ckpt
    if old_fpo > target:
        raise ValueError(f"checkpoint has {old_fpo} features/object, more than the current {target}")
    max_objects = ckpt["max_objects"]
    n_hist = len(HISTORY_STRIDES)
    tail = MAX_OBSTACLES * OBSTACLE_FEATURE_DIM + OWN_STATE_DIM
    object_segments = [(old_fpo, target)] * max_objects
    actor = dict(ckpt["actor_state_dict"])
    actor["trunk.0.weight"] = _relayout_columns(actor["trunk.0.weight"], n_hist, object_segments + [(tail, tail)])
    actor["action_trunk.0.weight"] = _relayout_columns(actor["action_trunk.0.weight"], n_hist, object_segments)
    critic = dict(ckpt["critic_state_dict"])
    critic["net.0.weight"] = _relayout_columns(critic["net.0.weight"], n_hist, object_segments + [(tail, tail)])
    return {
        **ckpt,
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "features_per_obj": target,
        "padded_from_features_per_obj": old_fpo,
    }


def migrate_pre_split_checkpoint(ckpt: dict) -> dict:
    """Convert a checkpoint from before the CLICK/KEY split and trail
    toggle (one 3-way no-op/attack/trail head, no key-share observation
    column, 4 own-state fields, normalized cursor action) into the current
    shapes:

    - input layers get zero-weight columns for the new per-object feature
      and the new own-state `reach` field;
    - the cursor head is unchanged; its tanh output now means a fraction of
      the world speed ceiling instead of 0.05 normalized/tick, which is the
      same speed on a chart ~800 world units wide (see TRAIN_DIARY.md
      2026-09-26 "world-unit cursor");
    - the when-head keeps the old no-op and attack rows. The old per-tick
      trail row is dropped: deterministic play only changes on a tick
      where trail was the argmax (check with the v3 usage numbers in
      TRAIN_DIARY.md — it never was);
    - the new how-head (click vs key) starts at logit 0: deterministic play
      ties to the key, exactly the old ATTACK; sampled rollouts try click
      and key 50/50 on key-bound targets and PPO learns which pays off.
      (On an unbound target both paths are a click, as before.)
    - the new trail-toggle head starts at the fresh-init prior (never
      toggles deterministically).

    Bernoulli-era checkpoints (separate attack_head/trail_head) are routed
    through migrate_bernoulli_checkpoint() first, which lands them in the
    pre-split categorical layout this function then finishes.
    """
    if ckpt.get("action_space") == ACTION_SPACE:
        return _pad_object_features(ckpt)
    if is_bernoulli_checkpoint(ckpt):
        return migrate_bernoulli_checkpoint(ckpt)
    if ckpt.get("action_space") is not None:
        raise ValueError(f"no migration from action space {ckpt['action_space']!r}")
    max_objects = ckpt["max_objects"]
    old_fpo = ckpt["features_per_obj"]
    new_fpo = old_fpo + EXTRA_OBJECT_FEATURES
    n_hist = len(HISTORY_STRIDES)
    obstacle_dim = MAX_OBSTACLES * OBSTACLE_FEATURE_DIM
    old_own = 4
    object_segments = [(old_fpo, new_fpo)] * max_objects
    full_slot = object_segments + [(obstacle_dim, obstacle_dim), (old_own, OWN_STATE_DIM)]

    actor = dict(ckpt["actor_state_dict"])
    actor["trunk.0.weight"] = _relayout_columns(actor["trunk.0.weight"], n_hist, full_slot)
    actor["action_trunk.0.weight"] = _relayout_columns(actor["action_trunk.0.weight"], n_hist, object_segments)
    actor["attack_state_trunk.0.weight"] = _relayout_columns(
        actor["attack_state_trunk.0.weight"], n_hist, [(old_own, OWN_STATE_DIM)]
    )
    hidden = actor["action_head.weight"].shape[1]
    if actor["action_head.weight"].shape[0] != 3:
        raise ValueError(f"pre-split action head should have 3 rows, got {actor['action_head.weight'].shape[0]}")
    actor["action_head.weight"] = actor["action_head.weight"][:2].clone()
    actor["action_head.bias"] = actor["action_head.bias"][:2].clone()
    actor["input_path_head.weight"] = torch.zeros(1, hidden)
    actor["input_path_head.bias"] = torch.zeros(1)
    actor["trail_toggle_head.weight"] = torch.zeros(1, 2 * hidden)
    actor["trail_toggle_head.bias"] = torch.full((1,), TRAIL_START_BIAS)
    actor["trail_release_offset"] = torch.tensor(TRAIL_RELEASE_OFFSET)

    critic = dict(ckpt["critic_state_dict"])
    critic["net.0.weight"] = _relayout_columns(critic["net.0.weight"], n_hist, full_slot)
    return {
        **ckpt,
        "actor_state_dict": actor,
        "critic_state_dict": critic,
        "features_per_obj": new_fpo,
        "action_space": ACTION_SPACE,
        "migrated_from_action_space": "noop_attack_trail",
        "migrated_from_features_per_obj": old_fpo,
    }


# Bias for the placeholder trail row a Bernoulli checkpoint gets in the
# intermediate 3-way head. migrate_pre_split_checkpoint() drops that row, so
# the value only matters if someone inspects the intermediate dict; very
# negative keeps it "never chosen" there too.
_BERNOULLI_DROPPED_ROW_BIAS = -1e4


def is_bernoulli_checkpoint(ckpt: dict) -> bool:
    """First PPO generation (round1/precision/round4/rl_policy.pt): separate
    Bernoulli attack_head + trail_head, no attack_state_trunk, no
    action_space tag."""
    actor = ckpt.get("actor_state_dict", {})
    return ckpt.get("action_space") is None and "attack_head.weight" in actor and "action_head.weight" not in actor


def migrate_bernoulli_checkpoint(ckpt: dict) -> dict:
    """Convert a Bernoulli-era checkpoint (attack = sigmoid(attack_head(
    action_trunk(objects))), trail held = sigmoid(trail_head(trunk)) per
    tick) into the pre-split categorical layout, then chain into
    migrate_pre_split_checkpoint() for the 72->73 / own-state 4->5 relayout:

    - attack_state_trunk is added with every weight AND bias zero, so its
      output is exactly 0 (ReLU(0) = 0 through both layers) and the when-
      logits are still a function of action_trunk alone, as before;
    - action_head = [0 row, attack row]: softmax([0, a])[1] = sigmoid(a),
      so P(press) equals the old P(attack) exactly and the deterministic
      argmax presses iff sigmoid(a) > 0.5 (ties go to no-op, like the old
      strict > 0.5);
    - the per-tick trail_head is DROPPED. The current action space only has
      a stateful toggle, which a per-tick hold probability doesn't map onto;
      the migrated policy never holds trail. The returned checkpoint carries
      `trail_head_dropped: True` so callers can say so;
    - the cursor head is unchanged. The old net's deterministic cursor was
      tanh(loc) * 0.05 normalized/tick; the new env reads tanh(loc) as a
      fraction of the world speed ceiling — same caveat as the categorical
      migration (see migrate_pre_split_checkpoint)."""
    actor = dict(ckpt["actor_state_dict"])
    hidden = actor["attack_head.weight"].shape[1]
    own_in = 4 * len(HISTORY_STRIDES)  # pre-split own-state width, all history slots
    actor["attack_state_trunk.0.weight"] = torch.zeros(hidden, own_in)
    actor["attack_state_trunk.0.bias"] = torch.zeros(hidden)
    actor["attack_state_trunk.2.weight"] = torch.zeros(hidden, hidden)
    actor["attack_state_trunk.2.bias"] = torch.zeros(hidden)
    attack_w = actor.pop("attack_head.weight")
    attack_b = actor.pop("attack_head.bias")
    actor.pop("trail_head.weight")
    actor.pop("trail_head.bias")
    actor["action_head.weight"] = torch.cat([torch.zeros(1, hidden), attack_w, torch.zeros(1, hidden)])
    actor["action_head.bias"] = torch.cat(
        [torch.zeros(1), attack_b, torch.full((1,), _BERNOULLI_DROPPED_ROW_BIAS)]
    )
    out = migrate_pre_split_checkpoint({**ckpt, "actor_state_dict": actor})
    out["migrated_from_action_space"] = "bernoulli"
    out["trail_head_dropped"] = True
    return out
