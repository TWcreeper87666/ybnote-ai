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

# §1 own-state fields: cursor_x, cursor_y, trail_held, ticks_since_attack (tanh-squashed)
OWN_STATE_DIM = 4

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

    def __init__(self, chart: ChartData, window: tuple[int, int] | None = None):
        self.chart = chart
        self.max_objects = chart.max_objects
        self.features_per_obj = chart.features_per_obj
        self._per_step_dim = _per_step_obs_dim(self.max_objects, self.features_per_obj)

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
        self._prev_attack_raw = False
        self._history: deque[np.ndarray] = deque(maxlen=self.max_history_back + 1)
        # (uid, event) of the shaping target tracked last step, for §8's
        # same-target-only potential diff (RL_DESIGN.md §8 fix).
        self._shape_target: tuple[int, dict] | None = None

    def reset(self, cursor_start: tuple[float, float] = (0.5, 0.5)) -> np.ndarray:
        self.judge = Judge(self.chart)
        self.step_idx = 0
        self.cursor = cursor_start
        self.trail_held = False
        self.ticks_since_attack = 1_000_000
        self._prev_attack_raw = False
        self._shape_target = None
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
        obj_feats = self.chart.input_features_at(chart_step).reshape(-1).numpy()
        obstacle_feats = nearby_obstacle_features(
            self.cursor, self.chart.collidable_centers, self.chart.collidable_halves
        ).reshape(-1).numpy()
        ticks_norm = float(np.tanh(self.ticks_since_attack / _TICKS_SINCE_ATTACK_SQUASH))
        own_state = np.array(
            [self.cursor[0], self.cursor[1], float(self.trail_held), ticks_norm], dtype=np.float32
        )
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

    def step(self, cursor_delta: tuple[float, float], attack_raw: bool, trail_held: bool) -> tuple[np.ndarray, float, bool, dict]:
        """cursor_delta: (dx, dy) already tanh-squashed and scaled to
        config.CURSOR_MAX_SPEED_NORM_PER_STEP by the policy (§2 — the speed
        cap is part of the action definition, enforced here as a hard clamp
        so a bug upstream can't exceed it). attack_raw: the policy's raw
        per-tick Bernoulli sample — decoded edge-triggered (§2/§16 fix): an
        actual game tap only fires on attack_raw's 0->1 transition, so a
        multi-tick "held" sample doesn't multi-fire against the same
        target. trail_held is level-triggered, unchanged (matches the real
        game already)."""
        chart_step = self._chart_step()
        t_ms = float(self.chart.t_ms[chart_step])
        prev_cursor = self.cursor

        dx, dy = cursor_delta
        speed = (dx * dx + dy * dy) ** 0.5
        if speed > config.CURSOR_MAX_SPEED_NORM_PER_STEP:
            scale = config.CURSOR_MAX_SPEED_NORM_PER_STEP / speed
            dx *= scale
            dy *= scale
        new_cursor = (
            float(np.clip(prev_cursor[0] + dx, 0.0, 1.0)),
            float(np.clip(prev_cursor[1] + dy, 0.0, 1.0)),
        )
        self.cursor = new_cursor

        attack_fired = bool(attack_raw) and not self._prev_attack_raw
        self._prev_attack_raw = bool(attack_raw)
        self.trail_held = bool(trail_held)
        if attack_fired:
            self.ticks_since_attack = 0
        else:
            self.ticks_since_attack += 1

        # WHICH key (or plain mouse click) is read off whatever's currently
        # targeted, exactly like the supervised policy — see §2, this is
        # observation-derived, never a separate decision.
        keybind_fired = set()
        if attack_fired:
            info = self._target_info(chart_step)
            if info is not None and info["key"] is not None:
                keybind_fired = {info["key"]}
                attack_fired = False  # goes through keybind_fired instead, not a mouse click

        action = {
            "attack_fired": attack_fired,
            "trail_held": self.trail_held,
            "keybind_fired": keybind_fired,
            "cursor": new_cursor,
            "output_spike_total": 0,  # no spiking energy term for the RL agent, see rl_env.py docstring
        }
        judgment_reward = self.judge.step(t_ms, action)
        shape_reward = self._shaping_reward(t_ms, prev_cursor, new_cursor)
        reward = judgment_reward + shape_reward

        self.step_idx += 1
        info = {"judgment_reward": judgment_reward, "shape_reward": shape_reward}
        if self.done:
            return self._stacked_obs(), reward, True, info

        self._history.append(self._raw_features_vec(self._chart_step()))
        return self._stacked_obs(), reward, False, info

    def _target_info(self, chart_step: int) -> dict | None:
        from cursor_readout import target_info

        return target_info(self.chart.input_features_at(chart_step))


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
