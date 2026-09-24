"""Loads test.frames.csv (network input) and test.events.json (judgment
ground truth) produced by scripts/encodeFrames.js."""

import json
import bisect
import os

import numpy as np
import pandas as pd
import torch

from track_eval import resolve_live_geometry, resolve_start_time


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

        # Every enabled Block/GroupRect in the level (not just ones a chart
        # note ever targets) — already normalized 0..1 by encodeFrames.js's
        # collectCollidables() using the SAME bounds as `events` above.
        # {"id", "type", "x", "y", "w", "h"} (x/y = top-left, axis-aligned —
        # see that function's docstring for the rotation-ignoring
        # simplification). Used by Judge for the real game's trail-vs-
        # anything-enabled Wrong rule, and by nearby-obstacle features for
        # model input. See TRAIN_DIARY.md 2026-09-24 "trail obstacle Wrong".
        collidables_path = frames_csv_path[: -len(".frames.csv")] + ".collidables.json"
        self.collidables = []
        # {"minX", "minY", "tileUnit", "resX", "resY"} — WORLD-space (not
        # normalized) integer tile grid metadata computed directly from raw
        # collidable coordinates (collectCollidables()'s computeCollidableGrid
        # in encodeFrames.js), so cell boundaries land exactly on wall
        # boundaries regardless of this chart's normalization bounds/padding
        # (see pathing.py's docstring / TRAIN_DIARY.md 2026-09-24 "trail path
        # label" for why that distinction mattered). None if the chart has no
        # collidables at all.
        self.collidable_grid = None
        if os.path.exists(collidables_path):
            with open(collidables_path, "r", encoding="utf-8") as f:
                payload2 = json.load(f)
            self.collidables = payload2["collidables"]
            self.collidable_grid = payload2.get("grid")
        # Precomputed once per chart for obstacles.nearby_obstacle_features()
        # (centers[M,2], half_sizes[M,2]) — see that module's docstring.
        # REST geometry only — a carried collidable's LIVE geometry is
        # resolved per-query by live_collidables_at() below, since it
        # changes every instant a track is actively carrying it.
        from obstacles import build_collidable_arrays

        self.collidable_centers, self.collidable_halves = build_collidable_arrays(self.collidables)

        # {"trackId": {"id", "channels", "loop", "bpm", "autoplay"}} and
        # {"trackId": [{"start","end"}, ...]} (seconds) — only for tracks
        # that actually carry an exported collidable (exportTrackData in
        # encodeFrames.js). Used by live_collidables_at() to resolve a
        # carried collidable's real position/scale at any instant, instead
        # of always using its rest rect — see track_eval.py's docstring
        # for why this gap mattered (14/32 real charts have a carried
        # collidable, one has 361).
        tracks_path = frames_csv_path[: -len(".frames.csv")] + ".tracks.json"
        self._tracks_by_id: dict = {}
        self._track_segments: dict = {}
        if os.path.exists(tracks_path):
            with open(tracks_path, "r", encoding="utf-8") as f:
                tpayload = json.load(f)
            self._tracks_by_id = {t["id"]: t for t in tpayload["tracks"]}
            self._track_segments = tpayload["segments"]

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

    def world_xy(self, nx: float, ny: float) -> tuple[float, float]:
        """Inverse of normalized_xy — this chart's 0..1 normalized space
        back to world units (what collidable_grid's minX/minY/tileUnit are
        expressed in)."""
        bx0, bx1 = self.bounds["minX"], self.bounds["maxX"]
        by0, by1 = self.bounds["minY"], self.bounds["maxY"]
        return nx * (bx1 - bx0) + bx0, ny * (by1 - by0) + by0

    def live_collidables_at(self, t_ms: float) -> list[dict]:
        """self.collidables, but any track-carried entry has its
        x/y/w/h replaced with LIVE geometry at this instant (position,
        uniform scale, AND rotation — confirmed real and load-bearing:
        CHROMANCE – Wrap Me In Plastic carries an actual scored noteblock
        90°→450° via its track, 4 other charts also rotate a real note
        object, not just decoration. Every entry also gets a
        `rotation_deg` field now (0 for anything not currently rotating,
        including every static collidable) so callers doing rect-overlap
        tests know when they must switch from the cheap axis-aligned test
        to a real OBB one — see reward.py's `_segment_intersects_obb`).
        A carried collidable whose track isn't currently running (an
        untriggered non-autoplay track) falls back to its rest rect
        (rotation_deg=0), same as encodeFrames.js's own fallback. Static
        (non-carried) collidables pass through unchanged. This is the
        function every per-step collision test (Judge, obstacle features)
        should call instead of reading self.collidables directly whenever
        the chart might have carried objects."""
        if not self._tracks_by_id:
            return [{**c, "rotation_deg": 0.0} for c in self.collidables]
        t_sec = t_ms / 1000.0
        out = []
        for c in self.collidables:
            track_id = c.get("carriedByTrackId")
            track = self._tracks_by_id.get(track_id) if track_id else None
            if track is None:
                out.append({**c, "rotation_deg": 0.0})
                continue
            elapsed = resolve_start_time(track, self._track_segments.get(track_id), t_sec)
            if elapsed is None:
                out.append({**c, "rotation_deg": 0.0})
                continue
            wx, wy, rotation_deg, scale = resolve_live_geometry(track, elapsed)
            nx, ny = self.normalized_xy({"x": wx, "y": wy})
            w, h = c["w"] * scale, c["h"] * scale
            entry = {**c, "x": nx - w / 2, "y": ny - h / 2, "w": w, "h": h, "rotation_deg": rotation_deg}
            if rotation_deg != 0.0:
                # World-space (isotropic) center/half-extents for the OBB
                # test — normalized space divides x by spanX and y by
                # spanY separately, so a rotation applied there would come
                # out SHEARED, not rotated (see this method's docstring /
                # RL_DESIGN.md §16). world_xy()'s inverse of the SAME
                # per-axis spans, applied consistently, is what keeps this
                # correct: wx/wy are already the live WORLD center (straight
                # from track_eval), and spanX/spanY convert the rest
                # (normalized) half-size back to world units.
                bx0, bx1 = self.bounds["minX"], self.bounds["maxX"]
                by0, by1 = self.bounds["minY"], self.bounds["maxY"]
                span_x, span_y = bx1 - bx0, by1 - by0
                entry["world_cx"], entry["world_cy"] = wx, wy
                entry["world_hw"] = (c["w"] * span_x) * scale / 2
                entry["world_hh"] = (c["h"] * span_y) * scale / 2
            out.append(entry)
        return out


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
