"""Scripted demonstrator for behavior cloning (train_bc.py).

The engineered_policy.py rule (move to the most urgent object, press when
its proximity crosses a threshold), re-expressed INSIDE TrailRLEnv so every
action it takes is one the ActorNet can represent and is judged by the same
env/Judge: it reads only the policy's own observation (the current history
slot's object features and own state), outputs a cursor delta as a fraction
of the world speed ceiling, and presses with the note's own key when its
object has a key binding nobody else shares (no aiming needed), otherwise
with CLICK (a click scores only the touched object, so a shared key binding
can't add a Wrong).
Because it reads the (possibly D4-transformed) observation, its actions
live in the same frame the policy acts in, and the env maps them back.

The network trained from these demonstrations still makes every decision
itself at play time; the rule only supplies training targets.
"""

from __future__ import annotations

import numpy as np

import config
import torch

import rl_env
from augment import transform_vector
from rl_env import OWN_STATE_DIM, PRESS_CLICK, PRESS_KEY, PRESS_NONE, TrailRLEnv, encode_action


# A block is 60 world units and a full-speed tick moves 40, so its center
# half-width is 0.75 ticks; stay a bit inside it.
ON_TARGET_REACH = 0.6


# With nothing on screen, drift back toward the canvas center (fixed under
# every D4 augmentation) with this per-tick gain (~100 ticks = 0.5s time
# constant). A "stand still" label gives the student no restoring force:
# its small output bias integrated over long gaps (bc2 drifted toward the
# top-left, bc5 off the canvas edge on Rhythm Hell); a weak pull to a fixed
# point corrects it — TRAIN_DIARY.md 2026-09-27 "gap drift".
GAP_RETURN_GAIN = 0.01


class ScriptedExpert:
    def __init__(self, proximity_threshold: float = 0.998, refractory_ms: float = 0.0, use_keys: bool = True):
        # refractory_ms defaults to 0: the student no longer sees its own
        # click clock (rl_env.ATTACK_CLOCK_FEATURE_ENABLED), and a label
        # that depends on it is noise to the student. A hit note leaves the
        # observation, so the rule never double-clicks one anyway.
        self.proximity_threshold = proximity_threshold
        self.refractory_ticks = round(refractory_ms / config.DT_MS)
        # use_keys: a note whose object has a keyBinding that no other target
        # shares is pressed with its key (no aiming needed, no extra Wrong).
        # A shared key still fires every bound target and each one without a
        # due note scores a Wrong (FALL FROM THE SKY PT. 2), so those notes
        # keep the click.
        self.use_keys = use_keys

    def act(self, env: TrailRLEnv, obs: np.ndarray) -> tuple[tuple[float, float], int]:
        """-> (cursor_delta as speed-ceiling fraction, encoded action)."""
        guide = env.guide
        if guide is not None:
            # A planned trail stroke (trail_plan.StrokeGuide) is in play:
            # head for its waypoint and start/hold/release as it says.
            delta = (np.array(guide.waypoint) - np.array(env.cursor)) / env.reach
            norm = float(np.hypot(delta[0], delta[1]))
            if norm > 1.0:
                delta = delta / norm
            if env.augmentation_mode != "identity":
                delta = transform_vector(torch.from_numpy(delta).float(), env.augmentation_mode).numpy()
            return (float(delta[0]), float(delta[1])), encode_action(PRESS_NONE, int(guide.toggle))
        # No stroke planned now: never hold one.
        release = int(env.trail_held)
        fpo = env.features_per_obj
        objects = obs[: env.max_objects * fpo].reshape(env.max_objects, fpo)
        own = obs[env._per_step_dim - OWN_STATE_DIM : env._per_step_dim]  # history slot 0 own state
        cursor = own[0:2]
        proximity = objects[:, 0]
        if float(proximity.max()) <= 0.0:
            home = GAP_RETURN_GAIN * (0.5 - cursor) / env.reach
            norm = float(np.hypot(home[0], home[1]))
            if norm > 1.0:
                home = home / norm
            return (float(home[0]), float(home[1])), encode_action(PRESS_NONE, release)
        best = int(proximity.argmax())
        target = objects[best, 1:3].astype(np.float64)
        if not rl_env.SAFE_CLICK_HINT_ENABLED:
            # The observation shows a group rect at its center; the teacher
            # still aims at its safe click point (data.safe_click_offset).
            uids = env.chart.event_uids_at(env._chart_step())
            if best < len(uids):
                offset = torch.tensor(env.chart.safe_click_offset(uids[best]), dtype=torch.float32)
                if env.augmentation_mode != "identity":
                    offset = transform_vector(offset, env.augmentation_mode)
                target = target + offset.numpy()
        delta = (target - cursor) / env.reach
        norm = float(np.hypot(delta[0], delta[1]))
        if norm > 1.0:
            delta = delta / norm
        # Press only once the cursor is on the target (within ON_TARGET_REACH
        # full-speed ticks of its center, ~a half block): a click anywhere
        # else can land on some other block and score a Wrong. When the
        # student is still on its way, the label is "keep moving" — the
        # 200ms late window leaves time to arrive.
        on_target = norm <= ON_TARGET_REACH
        press = PRESS_NONE
        ready = proximity[best] >= self.proximity_threshold and env.ticks_since_attack >= self.refractory_ticks
        if ready and self.use_keys and self._keyable(env, objects[best]):
            # The key is printed on the approach circle; pressing it needs no
            # cursor on the target, so don't wait for on_target.
            press = PRESS_KEY
        elif ready and on_target:
            press = PRESS_CLICK
        return (float(delta[0]), float(delta[1])), encode_action(press, release)

    @staticmethod
    def _keyable(env: TrailRLEnv, obj: np.ndarray) -> bool:
        """The object has a key label (keybind flag, one-hot) and key_share
        is 0: its key is bound to it alone, so one press scores only it."""
        key_share_col = env.features_per_obj - rl_env.EXTRA_OBJECT_FEATURES
        return bool(obj[3] > 0.5 and obj[4:key_share_col].max() > 0.5 and obj[key_share_col] <= 0.0)
