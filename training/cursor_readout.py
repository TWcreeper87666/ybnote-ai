"""Cursor position: computed DIRECTLY from the chart's own encoded object
positions, not reconstructed by the neural network.

Tried the "obvious" fix first — a supervised linear readout from reservoir
spikes to (x, y), fit by ridge regression against this exact target (see
TRAIN_DIARY.md 2026-09-23 #10/#11). It didn't work: even the best-case fit
only landed 5.9% of predictions within the real hit radius, because
build_input_current() (reward.py) collapses each frame's x/y into a single
scalar current injected into a handful of neurons — the spatial information
needed to reconstruct position is simply gone by the time it reaches the
reservoir, for ANY readout to recover. It's also arguably the honest
biological framing anyway: LC4/LPLC2 -> Giant Fiber is a looming/escape
circuit, not a place-coding one — its real job is "is something closing in,
react or don't", which is exactly what stays reward-trained
(attack_gate/trail_gate/keybind in readout.py). Aiming is looked up from the
same ground-truth chart data the offline encoder already has, same as the
real game engine already knows exactly where its own objects are.
"""

import torch

import config


def target_xy(features: torch.Tensor) -> tuple[float, float] | None:
    """features: [max_objects, 4] (proximity, x, y, keybind) from
    ChartData.input_features_at() — the same array build_input_current()
    reads. Returns the position of whichever active object has the HIGHEST
    proximity (closest to its due time right now) — the one thing actually
    worth aiming at this instant — or None if nothing's on screen.

    NOT a proximity-weighted average across every active object (2026-09-23
    #12's bug): with notes only ~200ms apart but a ~1000ms active window,
    4-5 notes are routinely active at once. Averaging their positions
    together aims at empty space between them — timing was landing within
    the real judgment window ~98% of the time, but the averaged cursor
    still missed the specific note that was actually due, scoring Wrong
    instead of a hit almost every time despite good timing."""
    proximity, x, y = features[:, 0], features[:, 1], features[:, 2]
    if float(proximity.max()) <= 0:
        return None
    best = int(proximity.argmax())
    return float(x[best]), float(y[best])


class SmoothedCursor:
    """Rate-limits target_xy()'s raw (teleporting) output to a max speed —
    a real mouse (or the AiReplayDriver camera that follows it) moves
    continuously frame to frame, it doesn't jump. Without this, the cursor
    snaps instantly to whichever note has the highest proximity right now,
    which can be a different note (and a different part of the screen) from
    one 5ms step to the next whenever two notes' active windows overlap —
    reported symptom: the replay camera visibly teleports instead of
    panning. Applied identically during training (so the Judge scores what
    the rate-limited cursor can actually reach) and at replay export time —
    same object, not two independent implementations that could drift."""

    def __init__(self, max_step_norm: float = config.CURSOR_MAX_SPEED_NORM_PER_STEP):
        self.max_step = max_step_norm
        self.pos = (0.5, 0.5)

    def update(self, features: torch.Tensor) -> tuple[float, float]:
        target = target_xy(features)
        if target is None:
            return self.pos

        dx = target[0] - self.pos[0]
        dy = target[1] - self.pos[1]
        dist = (dx * dx + dy * dy) ** 0.5
        if dist > self.max_step:
            scale = self.max_step / dist
            dx *= scale
            dy *= scale
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        return self.pos
