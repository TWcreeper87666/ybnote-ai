"""Gym-like environment for the end-to-end RL agent — RL_DESIGN.md's actual
implementation. Wraps ChartData + Judge (reused as-is, §14) with the
observation/action space from §1/§2 and the shaping fix from §8.

Deliberately NOT a generic gym.Env subclass (no gym dependency in this
project) — same reset()/step() shape, used directly by ppo.py's rollout
collector.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np
import torch

import config
from data import ChartData
from obstacles import MAX_OBSTACLES, OBSTACLE_FEATURE_DIM, nearby_obstacle_features
from reward import Judge
from augment import INVERSE_MODES, augment_batch, transform_vector
from augment import MODES as AUGMENT_MODES

# §1 own-state fields: cursor_x, cursor_y, trail_held, ticks_since_attack
# (tanh-squashed), reach (this chart's max cursor move per tick in
# normalized units, x REACH_FEATURE_SCALE). Reach is needed because the
# cursor action is a fraction of a WORLD-unit speed ceiling (see
# config.RL_CURSOR_MAX_SPEED_WORLD_PER_S) while positions are normalized
# per chart; it tells the policy how far a full-speed tick gets it here.
OWN_STATE_DIM = 5
REACH_FEATURE_SCALE = 10.0

# Per-tick action (§2) = two independent parts, both allowed on one tick
# exactly like an AiReplayDriver entry carries attack/keybindsFired AND
# trailHeld together:
#
# - press: none / CLICK / KEY. CLICK and KEY are the two real input paths
#   (`attack` = a tap at the cursor, `keybindsFired` =
#   PixiApproachCircleManager.triggerBoundKey): a click scores only what the
#   cursor touches, key-bound or not, while a bound key scores EVERY object
#   sharing that key. Pressing while a trail is held is legal and leaves the
#   stroke alone (AimGestureController.discreteSecondaryHit / onKeyDown).
# - trail toggle: flip the held state (start a stroke if up, release it if
#   down). A stroke therefore stays held until the policy explicitly
#   releases it, instead of needing a fresh TRAIL choice on every 5ms tick
#   (which made any hold longer than a few ticks improbable under sampling
#   and cut the stroke whenever a click/key was chosen).
#
# Encoded as one int, press + NUM_PRESS * toggle, so rollout buffers keep a
# single action index.
PRESS_NONE = 0
PRESS_CLICK = 1
PRESS_KEY = 2
NUM_PRESS = 3
NUM_ACTIONS = NUM_PRESS * 2
PRESS_NAMES = ("none", "click", "key")


def encode_action(press: int, trail_toggle: int) -> int:
    return int(press) + NUM_PRESS * int(trail_toggle)


def decode_action(action: int) -> tuple[int, int]:
    """-> (press, trail_toggle)"""
    return int(action) % NUM_PRESS, int(action) // NUM_PRESS

# Per-object columns the env appends to ChartData's encoded features:
# key_share_at() (how many other objects this note's key also fires), then
# hit_timing_at() (signed time to the note, symmetric around it).
EXTRA_OBJECT_FEATURES = 2

# Ablation switch (train_rl.py --no-timing-feature): keep the
# hit_timing_at() column in the layout but always zero, so a checkpoint
# trained either way loads into the same shapes.
TIMING_FEATURE_ENABLED = True


def obs_features_per_obj(chart_features_per_obj: int) -> int:
    return chart_features_per_obj + EXTRA_OBJECT_FEATURES

# §1 "short history": strided stack, not consecutive ticks — RL_DESIGN.md's
# fix for the 20ms-window blind spot. Indices are OFFSETS BACK from the
# current tick, in ticks (each tick = config.DT_MS = 5ms), so this spans
# 0, 20, 40, 60ms back — ~60ms of context instead of 15-20ms.
HISTORY_STRIDES = (0, 4, 8, 12)

# Ticks-since-attack is squashed through tanh at this scale (in ticks) so a
# note that hasn't fired in a while doesn't blow up the observation's range —
# same rationale as obstacles.py's _SQUASH_SCALE.
_TICKS_SINCE_ATTACK_SQUASH = 40.0

# Reward-shaping gain (r_shape = SHAPING_GAIN * (dist_prev - dist_now)) —
# small relative to a Perfect's +1.0 judgment reward so the agent still
# ultimately optimizes for real hits, not merely approach.
SHAPING_GAIN = 2.0


def _per_step_obs_dim(max_objects: int, features_per_obj: int) -> int:
    return max_objects * features_per_obj + MAX_OBSTACLES * OBSTACLE_FEATURE_DIM + OWN_STATE_DIM


def obs_dim(max_objects: int, features_per_obj: int) -> int:
    return _per_step_obs_dim(max_objects, features_per_obj) * len(HISTORY_STRIDES)


class TrailRLEnv:
    """One chart's worth of episode. `window` (start_step, length) restricts
    an episode to a sub-range for Stage A curriculum (RL_DESIGN.md §10);
    None runs the full chart (Stage B / final eval)."""

    def __init__(
        self,
        chart: ChartData,
        window: tuple[int, int] | None = None,
        augmentation_mode: str = "identity",
    ):
        self.chart = chart
        if augmentation_mode not in AUGMENT_MODES:
            raise ValueError(f"unknown observation transform: {augmentation_mode}")
        self.augmentation_mode = augmentation_mode
        self.max_objects = chart.max_objects
        self.features_per_obj = obs_features_per_obj(chart.features_per_obj)
        self._per_step_dim = _per_step_obs_dim(self.max_objects, self.features_per_obj)
        # Max cursor move per tick, normalized: the world ceiling divided by
        # this chart's world span.
        self.reach = (
            config.RL_CURSOR_MAX_SPEED_WORLD_PER_S * config.DT_MS / 1000.0 / chart.world_span
        )

        if window is None:
            self.start_step = 0
            self.length = chart.num_steps
        else:
            self.start_step = window[0]
            self.length = window[1]
        self.max_history_back = max(HISTORY_STRIDES)

        self.judge: Judge | None = None
        self.step_idx = 0
        self.cursor = (0.5, 0.5)
        self.trail_held = False
        self.ticks_since_attack = 1_000_000
        self._history: deque[np.ndarray] = deque(maxlen=self.max_history_back + 1)
        # (uid, event) of the shaping target tracked last step, for §8's
        # same-target-only potential diff (RL_DESIGN.md §8 fix).
        self._shape_target: tuple[int, dict] | None = None
        # The game-level input this tick resolved to (attack/keybind/trail/
        # cursor) — what export_replay.py writes out for AiReplayDriver.
        self.last_action: dict | None = None

    def reset(self, cursor_start: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        self.judge = Judge(self.chart)
        self.step_idx = 0
        self.cursor = cursor_start
        self.trail_held = False
        self.ticks_since_attack = 1_000_000
        self._shape_target = None
        self.last_action = None
        self._history.clear()
        # Warm the history buffer with the first real frame repeated, so
        # step 0 already has a full (non-zero-padded) stack instead of a
        # cold start that looks structurally different from every later
        # step.
        first = self._raw_features_vec(self.start_step)
        for _ in range(self.max_history_back + 1):
            self._history.append(first)
        return self._stacked_obs()

    @property
    def done(self) -> bool:
        return self.step_idx >= self.length

    def _chart_step(self) -> int:
        return self.start_step + self.step_idx

    def _raw_features_vec(self, chart_step: int) -> np.ndarray:
        visible = self._visible_object_features_at(chart_step)
        key_share = torch.from_numpy(self.chart.key_share_at(chart_step)).unsqueeze(-1)
        # A resolved (hidden) slot carries no key label either.
        key_share = key_share * (visible.abs().sum(-1, keepdim=True) > 0)
        visible_mask = visible.abs().sum(-1, keepdim=True) > 0
        timing = torch.from_numpy(
            self.chart.hit_timing_at(chart_step, float(self.chart.t_ms[chart_step]))
        ).unsqueeze(-1) * visible_mask * float(TIMING_FEATURE_ENABLED)
        obj_feats = torch.cat([visible, key_share, timing], dim=-1).reshape(-1).numpy()
        obstacle_feats = nearby_obstacle_features(
            self.cursor, self.chart.collidable_centers, self.chart.collidable_halves
        ).reshape(-1).numpy()
        ticks_norm = float(np.tanh(self.ticks_since_attack / _TICKS_SINCE_ATTACK_SQUASH))
        own_state = np.array(
            [
                self.cursor[0],
                self.cursor[1],
                float(self.trail_held),
                ticks_norm,
                self.reach * REACH_FEATURE_SCALE,
            ],
            dtype=np.float32,
        )
        if self.augmentation_mode != "identity":
            obj_t, obstacle_t, cursor_t = augment_batch(
                torch.from_numpy(obj_feats).reshape(1, -1),
                torch.from_numpy(obstacle_feats).reshape(1, -1),
                torch.tensor([self.cursor], dtype=torch.float32),
                self.features_per_obj,
                self.augmentation_mode,
            )
            obj_feats = obj_t[0].numpy()
            obstacle_feats = obstacle_t[0].numpy()
            own_state[:2] = cursor_t[0].numpy()
        return np.concatenate([obj_feats, obstacle_feats, own_state]).astype(np.float32)

    def _stacked_obs(self) -> np.ndarray:
        # self._history[-1] is the most recent tick (stride 0); index back
        # from there for the other strides. deque is newest-appended-last.
        parts = []
        for stride in HISTORY_STRIDES:
            idx = len(self._history) - 1 - stride
            idx = max(0, idx)
            parts.append(self._history[idx])
        return np.concatenate(parts)

    def _current_shaping_target(self, t_ms: float) -> tuple[int, dict] | None:
        """Same "most urgent" rule as cursor_readout.target_xy() (highest
        proximity active object) but resolved against Judge.pending's real
        uid bookkeeping instead of a raw feature slot, so identity is
        actually trackable across steps (a feature slot's index is not a
        stable object id — see this module's docstring)."""
        if not self.judge.pending:
            return None
        best_uid, best_ev, best_dt = None, None, None
        for uid, rec in self.judge.pending.items():
            ev = rec["event"]
            dt = abs(t_ms - ev["time"])
            if best_dt is None or dt < best_dt:
                best_uid, best_ev, best_dt = uid, ev, dt
        return (best_uid, best_ev) if best_uid is not None else None

    def _shaping_reward(self, t_ms: float, prev_cursor: tuple[float, float], cursor: tuple[float, float]) -> float:
        target = self._current_shaping_target(t_ms)
        prev_target = self._shape_target
        self._shape_target = target
        if target is None:
            return 0.0
        uid, ev = target
        # Target identity changed since last step (RL_DESIGN.md §8 fix) —
        # freeze shaping this step rather than diffing distance to two
        # different objects.
        if prev_target is None or prev_target[0] != uid:
            return 0.0
        tx, ty = self.chart.normalized_xy(ev)
        dist_prev = ((prev_cursor[0] - tx) ** 2 + (prev_cursor[1] - ty) ** 2) ** 0.5
        dist_now = ((cursor[0] - tx) ** 2 + (cursor[1] - ty) ** 2) ** 0.5
        return SHAPING_GAIN * (dist_prev - dist_now)

    def step(self, cursor_delta: tuple[float, float], action: int) -> tuple[np.ndarray, float, bool, dict]:
        """cursor_delta: (dx, dy) as a FRACTION of the world speed ceiling
        (|delta| <= 1; the policy tanh-squashes to that), converted here to
        this chart's normalized units via self.reach. The ceiling is
        re-enforced as a hard clamp so a bug upstream can't exceed it, and
        each tick pays config.RL_CURSOR_EFFORT_COEF * |delta|^2 (§2/§8).
        action: encode_action(press,
        trail_toggle), sampled once per tick, so PPO's action probability
        maps directly to the environment action without edge-trigger
        ambiguity."""
        chart_step = self._chart_step()
        t_ms = float(self.chart.t_ms[chart_step])
        prev_cursor = self.cursor

        delta = torch.tensor(cursor_delta, dtype=torch.float32)
        if self.augmentation_mode != "identity":
            delta = transform_vector(delta, INVERSE_MODES[self.augmentation_mode])
        dx, dy = float(delta[0]), float(delta[1])
        speed_fraction = (dx * dx + dy * dy) ** 0.5
        if speed_fraction > 1.0:
            dx /= speed_fraction
            dy /= speed_fraction
            speed_fraction = 1.0
        effort_reward = -config.RL_CURSOR_EFFORT_COEF * speed_fraction * speed_fraction
        dx *= self.reach
        dy *= self.reach
        new_cursor = (
            float(np.clip(prev_cursor[0] + dx, 0.0, 1.0)),
            float(np.clip(prev_cursor[1] + dy, 0.0, 1.0)),
        )
        self.cursor = new_cursor

        press, trail_toggle = decode_action(action)
        if trail_toggle:
            self.trail_held = not self.trail_held
        attack_fired = press == PRESS_CLICK
        keybind_fired = set()
        if press == PRESS_KEY:
            # WHICH key is read off the currently targeted object's printed
            # label (observation-derived, §2). If that object has no key,
            # the press is of an unbound key, which AimGestureController's
            # onKeyDown turns into an attack at the cursor — i.e. a click.
            info = self._target_info(chart_step)
            if info is not None and info["key"] is not None:
                keybind_fired = {info["key"]}
            else:
                attack_fired = True
        if press != PRESS_NONE:
            self.ticks_since_attack = 0
        else:
            self.ticks_since_attack += 1

        game_action = {
            "attack_fired": attack_fired,
            "trail_held": self.trail_held,
            "keybind_fired": keybind_fired,
            "cursor": new_cursor,
            "output_spike_total": 0,  # no spiking energy term for the RL agent, see rl_env.py docstring
        }
        self.last_action = game_action
        judgment_reward = self.judge.step(t_ms, game_action)
        shape_reward = self._shaping_reward(t_ms, prev_cursor, new_cursor)
        reward = judgment_reward + shape_reward + effort_reward

        self.step_idx += 1
        info = {
            "judgment_reward": judgment_reward,
            "shape_reward": shape_reward,
            "effort_reward": effort_reward,
        }
        if self.done:
            if self.start_step + self.length >= self.chart.num_steps:
                # Episode reached the chart's end: notes whose window closes
                # after the last frame are Misses too (Judge.finalize).
                miss_reward = self.judge.finalize()
                reward += miss_reward
                info["judgment_reward"] += miss_reward
            return self._stacked_obs(), reward, True, info

        self._history.append(self._raw_features_vec(self._chart_step()))
        return self._stacked_obs(), reward, False, info

    def _target_info(self, chart_step: int) -> dict | None:
        from cursor_readout import target_info

        return target_info(self._visible_object_features_at(chart_step))

    def _visible_object_features_at(self, chart_step: int) -> torch.Tensor:
        """The pre-encoded chart is a schedule, not a live screen: hide a
        circle once Judge has resolved its unique note, as the real game
        removes that approach circle after hit or expiry."""
        features = self.chart.input_features_at(chart_step).clone()
        for slot, uid in enumerate(self.chart.event_uids_at(chart_step)):
            if uid in self.judge.resolved_uids:
                features[slot].zero_()
        return features


def sample_window(chart: ChartData, min_len: int = 500, max_len: int = 1500, rng: random.Random | None = None) -> tuple[int, int]:
    """Stage A curriculum window (RL_DESIGN.md §10): a random contiguous
    slice, short enough that one bad episode can't destabilize much
    accumulated policy."""
    rng = rng or random
    length = rng.randint(min_len, max_len)
    length = min(length, chart.num_steps)
    max_start = max(0, chart.num_steps - length)
    start = rng.randint(0, max_start) if max_start > 0 else 0
    return start, length
