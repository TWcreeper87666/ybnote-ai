"""Scripted demonstrator for behavior cloning (train_bc.py).

The engineered_policy.py rule (move to the most urgent object, press when
its proximity crosses a threshold), re-expressed INSIDE TrailRLEnv so every
action it takes is one the ActorNet can represent and is judged by the same
env/Judge: it reads only the policy's own observation (the current history
slot's object features and own state), outputs a cursor delta as a fraction
of the world speed ceiling, and presses with CLICK (never KEY: a click
scores only the touched object, so a shared key binding can't add a Wrong).
Because it reads the (possibly D4-transformed) observation, its actions
live in the same frame the policy acts in, and the env maps them back.

The network trained from these demonstrations still makes every decision
itself at play time; the rule only supplies training targets.
"""

from __future__ import annotations

import numpy as np
import torch

import config
from augment import transform_vector
from rl_env import PRESS_CLICK, PRESS_NONE, TrailRLEnv, encode_action


# A block is 60 world units and a full-speed tick moves 40, so its center
# half-width is 0.75 ticks; stay a bit inside it.
ON_TARGET_REACH = 0.6


# Candidate grid per axis when searching a group rect for a click point.
_SAFE_GRID = 9


def _rect_distance(px: float, py: float, r: dict) -> float:
    """Distance from a point to an axis-aligned rect (0 inside)."""
    dx = max(r["x"] - px, 0.0, px - (r["x"] + r["w"]))
    dy = max(r["y"] - py, 0.0, py - (r["y"] + r["h"]))
    return float(np.hypot(dx, dy))


class ScriptedExpert:
    def __init__(self, proximity_threshold: float = 0.998, refractory_ms: float = 100.0):
        self.proximity_threshold = proximity_threshold
        self.refractory_ticks = round(refractory_ms / config.DT_MS)
        # (chart name, uid) -> canonical-frame (dx, dy) from the note's
        # encoded position to where to click.
        self._aim_offsets: dict[tuple[str, int], tuple[float, float]] = {}

    def _aim_offset(self, env: TrailRLEnv, uid: int) -> tuple[float, float]:
        """A group-rect note is encoded at the rect's center, but a click
        that starts on any block inside the rect scores only that block
        (trailSweep.ts startedOnBlock), usually a Wrong — NIGHT DANCER's
        og14p2r has a block 10 world units from its center, and imitating
        "click the center" missed it by that much. Aim at the point of the
        live rect farthest from every block instead. Blocks and track
        handles are aimed at their center."""
        key = (env.chart.name, uid)
        if key in self._aim_offsets:
            return self._aim_offsets[key]
        ev = env.chart.events[uid]
        offset = (0.0, 0.0)
        if ev.get("type") == "groupRect":
            live = env.chart.live_collidables_at(float(ev["time"]))
            rect = next((c for c in live if c["id"] == ev["id"]), None)
            blocks = [c for c in live if c.get("type") == "block"]
            if rect is not None and blocks:
                cx, cy = env.chart.normalized_xy(ev)
                best, best_score = (cx, cy), -1.0
                for i in range(_SAFE_GRID):
                    for j in range(_SAFE_GRID):
                        px = rect["x"] + rect["w"] * (i + 0.5) / _SAFE_GRID
                        py = rect["y"] + rect["h"] * (j + 0.5) / _SAFE_GRID
                        clearance = min(_rect_distance(px, py, b) for b in blocks)
                        # Prefer clearance, then closeness to the center.
                        score = clearance - 1e-3 * np.hypot(px - cx, py - cy)
                        if score > best_score:
                            best, best_score = (px, py), score
                offset = (best[0] - cx, best[1] - cy)
        self._aim_offsets[key] = offset
        return offset

    def act(self, env: TrailRLEnv, obs: np.ndarray) -> tuple[tuple[float, float], int]:
        """-> (cursor_delta as speed-ceiling fraction, encoded action)."""
        fpo = env.features_per_obj
        objects = obs[: env.max_objects * fpo].reshape(env.max_objects, fpo)
        own = obs[env._per_step_dim - 5 : env._per_step_dim]  # history slot 0 own state
        cursor = own[0:2]
        proximity = objects[:, 0]
        if float(proximity.max()) <= 0.0:
            return (0.0, 0.0), encode_action(PRESS_NONE, 0)
        best = int(proximity.argmax())
        target = objects[best, 1:3].astype(np.float64)
        uid = int(env.chart._event_uid_slots[env._chart_step()][best])
        if uid >= 0:
            offset = torch.tensor(self._aim_offset(env, uid), dtype=torch.float32)
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
        if (
            on_target
            and proximity[best] >= self.proximity_threshold
            and env.ticks_since_attack >= self.refractory_ticks
        ):
            press = PRESS_CLICK
        return (float(delta[0]), float(delta[1])), encode_action(press, 0)
