"""Loads test.frames.csv (network input) and test.events.json (judgment
ground truth) produced by scripts/encodeFrames.js."""

import json
import bisect
import os

import numpy as np
import pandas as pd
import torch

import config
from track_eval import resolve_live_geometry, resolve_start_time

# A group-rect note's click point keeps this many world units from every
# block and from the rect's own edge (one full-speed tick), searched on a
# SAFE_CLICK_GRID x SAFE_CLICK_GRID grid over the rect.
SAFE_CLICK_MARGIN_WORLD = 40.0
SAFE_CLICK_GRID = 21


def rect_distance(px: float, py: float, r: dict) -> float:
    """Distance from a point to an axis-aligned rect (0 inside)."""
    dx = max(r["x"] - px, 0.0, px - (r["x"] + r["w"]))
    dy = max(r["y"] - py, 0.0, py - (r["y"] + r["h"]))
    return float(np.hypot(dx, dy))

def approach_progress(t_ms: float, event_time: float) -> float:
    """Same as encodeFrames.js approachProgress(): 0 -> 1 over the approach,
    then 1 -> 2 across the +HIT_WINDOW_MS late window. Must match it exactly
    so _build_event_uid_slots orders slots like the encoded frames do."""
    if t_ms <= event_time:
        return max(0.0, min(1.0, (t_ms - (event_time - config.APPROACH_TIME_MS)) / config.APPROACH_TIME_MS))
    return 1.0 + min(1.0, (t_ms - event_time) / config.HIT_WINDOW_MS)


# key_share_at() saturates at this many extra same-key objects.
KEY_SHARE_SATURATION = 4


class ChartData:
    def __init__(self, frames_csv_path: str, events_json_path: str):
        self.name = os.path.basename(frames_csv_path)[: -len(".frames.csv")]
        self.events_path = events_json_path
        self.t_ms, self.frame_tensor, self.max_objects, self.features_per_obj = _load_frames_csv(frames_csv_path)

        with open(events_json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.bounds = payload["bounds"]
        self.match_by_pitch_instrument = bool(
            payload.get("constants", {}).get("matchByPitchInstrument", False)
        )
        self.events = payload["events"]  # sorted by time, each has time/id/type/x/y/hasKeyBinding/keyBinding
        # `id` is the TARGET OBJECT's id (a block/groupRect), not a unique
        # per-note id — the same object gets hit by many different notes
        # (e.g. a repeated drum hit), so `id` alone can't key a judge's
        # pending-note bookkeeping (see TRAIN_DIARY.md 2026-09-23 #7). `_uid`
        # is the note's actual unique identity: its position in this array.
        for i, ev in enumerate(self.events):
            ev["_uid"] = i
        self._event_times = [e["time"] for e in self.events]
        self._event_times_arr = np.array(self._event_times, dtype=np.float64)
        self._event_uid_slots = self._build_event_uid_slots()

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
        self.track_handles = []
        for track in self._tracks_by_id.values():
            handle = track.get("controlHandle")
            if handle is None:
                continue
            self.track_handles.append({
                "id": track["id"],
                "type": "track",
                "x": (handle["x"] - self.bounds["minX"]) / (self.bounds["maxX"] - self.bounds["minX"]),
                "y": (handle["y"] - self.bounds["minY"]) / (self.bounds["maxY"] - self.bounds["minY"]),
                "w": handle["w"] / (self.bounds["maxX"] - self.bounds["minX"]),
                "h": handle["h"] / (self.bounds["maxY"] - self.bounds["minY"]),
                "rotation_deg": 0.0,
                "enabled": track.get("enabled", True),
                "controlHandleHidden": track.get("controlHandleHidden", False),
                "keyBinding": track.get("keyBinding"),
                "pitch": None,
                "instrument": None,
            })
        self._key_share_slots = self._build_key_share_slots()

    @property
    def world_span(self) -> float:
        """World units spanned by the 0..1 normalized space (bounds are
        square, see encodeFrames.js computeBounds)."""
        return max(self.bounds["maxX"] - self.bounds["minX"], self.bounds["maxY"] - self.bounds["minY"])

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

    def first_event_after(self, t_ms: float) -> int:
        """Index into self.events (time-sorted) of the first note due after t_ms."""
        times = self.__dict__.get("_event_times_sorted")
        if times is None:
            times = self._event_times_sorted = [float(ev["time"]) for ev in self.events]
        return bisect.bisect_right(times, t_ms)

    def event_uids_at(self, step: int) -> list[int]:
        """Unique event ids in the same proximity-sorted slots as the
        encoded frame at `step`. This lets the RL environment remove notes
        that have already been resolved from its live observation."""
        return [int(uid) for uid in self._event_uid_slots[step] if uid >= 0]

    def key_bound_targets(self, key: str) -> list[dict]:
        """Every target one physical press of `key` scores, in the order
        PixiApproachCircleManager.triggerBoundKey() visits them: enabled
        blocks/groupRects and tracks whose keyBinding matches."""
        lower = key.lower()
        return [
            c for c in [*self.collidables, *self.track_handles]
            if c.get("enabled", True)
            if c.get("keyBinding")
            and c["keyBinding"].lower() == lower
        ]

    def hit_timing_at(self, step: int, t_ms: float) -> np.ndarray:
        """[max_objects] per-slot (t - note time) / HIT_WINDOW_MS clipped to
        [-1, 1]: -1 at 200ms early, 0 on the note, +1 at 200ms late, equally
        steep on both sides. proximity alone ramps 0->1 over the 800ms
        approach but 1->2 over the 200ms after the note, 4x steeper there,
        and the policy learned to fire on the steep late side (median hit
        +53ms, 59% of hits >50ms late — TRAIN_DIARY.md 2026-09-26 "timing
        feature"). 0 for empty slots."""
        uids = self._event_uid_slots[step]
        out = np.zeros(self.max_objects, dtype=np.float32)
        valid = uids >= 0
        if valid.any():
            times = self._event_times_arr[uids[valid]]
            out[valid] = np.clip((t_ms - times) / config.HIT_WINDOW_MS, -1.0, 1.0)
        return out

    def safe_click_offset(self, uid: int) -> tuple[float, float]:
        """Normalized (dx, dy) from a note's encoded position to where to
        click it. A group rect is encoded at its center, but a click that
        starts on any block inside it scores only that block
        (trailSweep.ts startedOnBlock), usually a Wrong; NIGHT DANCER's rects
        hold blocks 1-10 world units from their centers. So a group rect's
        click point is the point of the live rect nearest its center that
        keeps SAFE_CLICK_MARGIN_WORLD from every block and from the rect's
        edge (or, if none does, the one with the most room). The env shows
        the policy this point instead of the center (bc6b, taught to click
        the offset point from the center encoding, still clicked 7-8 world
        units from the blocks: 149 rects among thousands of notes were too
        few to learn the offset from obstacle features). (0, 0) for
        everything else."""
        cache = self.__dict__.setdefault("_safe_click_offsets", {})
        if uid in cache:
            return cache[uid]
        ev = self.events[uid]
        offset = (0.0, 0.0)
        if ev.get("type") == "groupRect":
            live = self.live_collidables_at(float(ev["time"]))
            rect = next((c for c in live if c["id"] == ev["id"]), None)
            blocks = [c for c in live if c.get("type") == "block"]
            if rect is not None and blocks:
                cx, cy = self.normalized_xy(ev)
                margin = SAFE_CLICK_MARGIN_WORLD / self.world_span
                safe, roomy, roomiest = None, (cx, cy), -1.0
                for i in range(SAFE_CLICK_GRID):
                    for j in range(SAFE_CLICK_GRID):
                        px = rect["x"] + rect["w"] * (i + 0.5) / SAFE_CLICK_GRID
                        py = rect["y"] + rect["h"] * (j + 0.5) / SAFE_CLICK_GRID
                        inset = min(px - rect["x"], rect["x"] + rect["w"] - px,
                                    py - rect["y"], rect["y"] + rect["h"] - py)
                        room = min(inset, min(rect_distance(px, py, b) for b in blocks))
                        dist = float(np.hypot(px - cx, py - cy))
                        if room >= margin and (safe is None or dist < safe[0]):
                            safe = (dist, px, py)
                        if room > roomiest:
                            roomy, roomiest = (px, py), room
                best = safe[1:] if safe is not None else roomy
                offset = (best[0] - cx, best[1] - cy)
        cache[uid] = offset
        return offset

    def key_share_at(self, step: int) -> np.ndarray:
        """[max_objects] per-slot "other objects this note's key would also
        fire", squashed to 0..1 (0 = its key is bound to it alone, or it has
        no key). A player reads every block's key label on screen; without
        this an observation only shows objects that have a note due, so a
        second same-key block with nothing due (FALL FROM THE SKY PT. 2:
        two `f` tom blocks, one of them never scored) is invisible even
        though pressing `f` scores it a Wrong. See TRAIN_DIARY.md
        2026-09-26 "CLICK / KEY split"."""
        return self._key_share_slots[step]

    def _build_key_share_slots(self) -> np.ndarray:
        extra_by_key: dict[str, float] = {}
        per_event = np.zeros(len(self.events), dtype=np.float32)
        for i, ev in enumerate(self.events):
            key = ev.get("keyBinding") if ev.get("hasKeyBinding") else None
            if not key:
                continue
            lower = key.lower()
            if lower not in extra_by_key:
                extra = max(0, len(self.key_bound_targets(lower)) - 1)
                extra_by_key[lower] = min(extra, KEY_SHARE_SATURATION) / KEY_SHARE_SATURATION
            per_event[i] = extra_by_key[lower]
        slots = np.zeros((self.num_steps, self.max_objects), dtype=np.float32)
        valid = self._event_uid_slots >= 0
        slots[valid] = per_event[self._event_uid_slots[valid]]
        return slots

    def _build_event_uid_slots(self) -> np.ndarray:
        slots = np.full((self.num_steps, self.max_objects), -1, dtype=np.int32)
        for step, t_value in enumerate(self.t_ms):
            t_ms = float(t_value)
            active = self.active_events_at(
                t_ms,
                window_before_ms=config.APPROACH_TIME_MS,
                window_after_ms=config.HIT_WINDOW_MS,
            )
            active.sort(key=lambda ev: approach_progress(t_ms, ev["time"]), reverse=True)
            count = min(self.max_objects, len(active))
            slots[step, :count] = [ev["_uid"] for ev in active[:count]]
        return slots

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

    def live_judge_targets_at(self, t_ms: float) -> list[dict]:
        """Visible collision targets used by game input: blocks, group rects,
        plus enabled track control handles that are not hidden."""
        handles = [
            handle for handle in self.track_handles
            if handle["enabled"] and not handle["controlHandleHidden"]
        ]
        return self.live_collidables_at(t_ms) + handles


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
