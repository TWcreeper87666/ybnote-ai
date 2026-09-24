"""Pure hand-engineered policy — NO neural network anywhere in the decision
path. Answers a different question than train.py/readout.py: not "how well
can the real connectome's activity drive this", but "what's the ceiling if
we stop pretending and just write the obvious rule".

Fires attack whenever the closest-due object's proximity crosses a
threshold (proximity hits ~1.0 at a note's hit time, per encodeFrames.js's
approachProgress()), gated by a refractory period, cursor via the same
SmoothedCursor. That's it — no reservoir, no readout, no R-STDP, no
training loop.

Defaults (threshold=0.998, refractory=208ms) are the best of a grid search
on this chart's accuracy score — see TRAIN_DIARY.md 2026-09-24's later
entries. There's a real hits-vs-accuracy trade-off here: refractory=208ms
happens to match this chart's note spacing almost exactly, so the policy
effectively "attacks at most once per note slot" — Wrong drops to ~18 and
Perfect jumps to ~170, but some notes get missed outright when timing
drifts (Miss ~52, hits ~253/305 = 83%, down from ~279/305 = 91.5% at the
more trigger-happy refractory=140ms). Re-run the grid search per-chart if
you want to retune — note spacing varies, and that's exactly what these two
numbers trade off against.

The neural path (snn_model.py/readout.py/train.py) is untouched and still
fully wired up — this is an alternative decision source, not a replacement.
"""

import torch

import config
from cursor_readout import SmoothedCursor, target_xy


class EngineeredPolicy:
    def __init__(self, proximity_threshold: float = 0.998, refractory_ms: float = 208.0):
        self.cursor_source = SmoothedCursor()
        self.proximity_threshold = proximity_threshold
        self.refractory_steps = round(refractory_ms / config.DT_MS)
        self.refractory_steps_left = 0

    def decide(self, features: torch.Tensor) -> dict:
        cursor = self.cursor_source.step(target_xy(features))
        proximity_max = float(features[:, 0].max())

        attack_fired = False
        if self.refractory_steps_left > 0:
            self.refractory_steps_left -= 1
        elif proximity_max >= self.proximity_threshold:
            attack_fired = True
            self.refractory_steps_left = self.refractory_steps

        return {
            "attack_fired": attack_fired,
            "trail_held": False,
            "keybind_fired": set(),
            "cursor": cursor,
            "output_spike_total": 0,
        }
