"""Loads test.frames.csv (network input) and test.events.json (judgment
ground truth) produced by scripts/encodeFrames.js."""

import json
import bisect

import numpy as np
import pandas as pd
import torch


class ChartData:
    def __init__(self, frames_csv_path: str, events_json_path: str):
        self.t_ms, self.frame_tensor, self.max_objects, self.features_per_obj = _load_frames_csv(frames_csv_path)

        with open(events_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.bounds = payload["bounds"]
        self.events = payload["events"]  # sorted by time, each has time/id/type/x/y/hasKeyBinding/keyBinding
        # `id` is the TARGET OBJECT's id (a block/groupRect), not a unique
        # per-note id — the same object gets hit by many different notes
        # (e.g. a repeated drum hit), so `id` alone can't key a judge's
        # pending-note bookkeeping (see TRAIN_DIARY.md 2026-09-23 #7). `_uid`
        # is the note's actual unique identity: its position in this array.
        for i, ev in enumerate(self.events):
            ev["_uid"] = i
        self._event_times = [e["time"] for e in self.events]

    @property
    def num_steps(self) -> int:
        return len(self.t_ms)

    def input_features_at(self, step: int) -> torch.Tensor:
        """[max_objects, features_per_obj] tensor: proximity, x, y, keybind,
        then a one-hot over config.KEY_VOCAB (0 vector if unbound or vocab
        miss) — already 0..1 normalized by encodeFrames.js, ready to scale
        into LIF input current or feed the DL policy directly."""
        return self.frame_tensor[step]

    def active_events_at(self, t: float, window_before_ms: float, window_after_ms: float):
        """Events whose hit time is within [t - window_after_ms, t + window_before_ms]
        i.e. spawned but not yet past their Bad grace window. Mirrors the
        window used to build frames in encodeFrames.js."""
        lo = t - window_after_ms
        hi = t + window_before_ms
        lo_idx = bisect.bisect_left(self._event_times, lo)
        result = []
        for i in range(lo_idx, len(self.events)):
            ev = self.events[i]
            if ev["time"] > hi:
                break
            result.append(ev)
        return result

    def normalized_xy(self, ev) -> tuple[float, float]:
        bx0, bx1 = self.bounds["minX"], self.bounds["maxX"]
        by0, by1 = self.bounds["minY"], self.bounds["maxY"]
        return (ev["x"] - bx0) / (bx1 - bx0), (ev["y"] - by0) / (by1 - by0)


def _load_frames_csv(path: str):
    # header: t, obj0_proximity, obj0_x, obj0_y, obj0_keybind, obj0_key0, obj0_key1, ..., obj1_..., ...
    # features_per_obj is inferred from how many obj0_* columns exist, so
    # this doesn't hardcode the old fixed width of 4 — see TRAIN_DIARY.md
    # 2026-09-24 "keybind support" for why a one-hot key block got appended.
    with open(path, "r", encoding="utf-8") as f:
        header = f.readline().strip().split(",")
    obj_cols = header[1:]
    features_per_obj = sum(1 for c in obj_cols if c.startswith("obj0_"))
    max_objects = len(obj_cols) // features_per_obj

    # pandas' C parser instead of np.loadtxt — with the keybind one-hot
    # block, columns went 32->328 (max_objects * features_per_obj), and
    # np.loadtxt's pure-Python row-by-row parsing made loading the 26-chart
    # training corpus slow enough to thrash system memory instead of
    # finishing. See TRAIN_DIARY.md 2026-09-24 "keybind support".
    data = pd.read_csv(path, skiprows=1, header=None, dtype=np.float32).to_numpy()
    if data.ndim == 1:  # single-row file
        data = data[None, :]

    t_ms = data[:, 0]
    feature_cols = data[:, 1:].reshape(-1, max_objects, features_per_obj)
    return t_ms, torch.from_numpy(feature_cols).float(), max_objects, features_per_obj
