"""Trail stroke planner: which notes need a held stroke, and where the
cursor goes while one is held.

Why strokes exist at all (Judge / trailSweep.ts semantics): a click scores
EVERY object under the cursor, so a note whose object is covered by
something with nothing due (only因為你那渴望自由的心臟: a track carries the
scaled-up D5 block onto each drum right at its beat; 迷宮: ten silent
1350x1350 group rects lie under the goal) can't be clicked without Wrongs.
A held stroke only scores FRESH entries: start it on a note's block
(startedOnBlock keeps the enclosing group rects from firing, but marks them
intersected), keep the cursor inside what it is already in, and the covering
objects never fire; stepping into the target at its beat scores only it.

The plan is computed from the chart's live geometry (privileged: the full
level, including where tracks will carry things) and drives
bc_expert.ScriptedExpert. The policy sees its output through rl_env's
own-state stroke columns (stroke active + next waypoint), the same way it
sees a group rect's safe click point: the maze route and a track's future
path are level knowledge a player reads off the screen or learns by
replaying, not something a 60ms observation window can infer.
TRAIN_DIARY.md 2026-09-27 "trail".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import csgraph

import config
from reward import _collidable_obb

# Distance-field / candidate grid resolution, in world units (a block is
# 60, a maze wall 30, a full-speed tick 40).
GRID_WORLD = 5.0
# Objects are padded by this much in the navigator's containment test, so a
# path it calls clean stays clean under the Judge's exact segment test.
CLEARANCE_WORLD = 2.0
# Points sampled along each candidate move (a wall is thinner than a tick).
SEGMENT_SAMPLES = 8
# The navigator's path keeps this far from anything it must not touch where
# there is room (half a 30-unit maze corridor), paying CENTER_PENALTY per
# cell of shortfall per cell travelled.
CENTER_CLEARANCE_WORLD = 15.0
CENTER_PENALTY = 10.0
# Pacing and smoothness of stroke moves (the expert raced the maze at 4000
# units/s and then waited 55s by the goal, and picked each tick's move from
# a discrete candidate set, which jittered around the carried block's
# center — bc9 amplified that to 50% direction reversals).
#   - speed = path left / time left, arriving ARRIVE_EARLY_MS before the
#     beat, clamped to [MIN, MAX]_STROKE_SPEED world units/s;
#   - wait STANDOFF_WORLD short of the target (path length), then glide in;
#   - plan the centerline route once and advance along it by arc length;
#   - inside a moving cover, keep the cover-local position (ride it).
ARRIVE_EARLY_MS = 300.0
# Re-plan the route when the cursor is this far off it (the student drove
# elsewhere during DAgger); smooth the cell path with this many passes.
TRACK_REPLAN_WORLD = 6.0
TRACK_SMOOTHING_PASSES = 4
# Exponential smoothing of the cursor's move relative to a moving cover.
RIDE_SMOOTHING = 0.8
STANDOFF_WORLD = 10.0
MIN_STROKE_SPEED = 30.0
MAX_STROKE_SPEED = 1500.0
SETTLE_SPEED = 150.0
# Stroke moves use at most this share of the speed ceiling (stepping into a
# target on its beat may use all of it): a 30-unit corridor at 40 units per
# tick left the greedy path trading centering for progress, and there is
# no hurry — 迷宮 gives the route 57 seconds.
STROKE_SPEED_FRACTION = 0.5
# Start a stroke this early (still Perfect, |offset| < 50ms). On the beat
# itself, the start competed with the press head, which fires ~20ms after
# it: bc8 kept clicking the start note (losing the stroke) because the
# toggle label sat on too few ticks to win that race.
STROKE_START_LEAD_MS = 25.0
# Candidate click points per axis inside a note's object.
CLICK_GRID = 9


def judge_id(c: dict) -> str:
    """The id Judge._inside_collidables uses for a live target."""
    return f"track:{c['id']}" if c.get("type") == "track" else c["id"]


class LiveGeometry:
    """Every live judge target at one instant, as world-space OBBs."""

    def __init__(self, chart, t_ms: float):
        targets = chart.live_judge_targets_at(t_ms)
        self.ids = [judge_id(c) for c in targets]
        self.index = {i: k for k, i in enumerate(self.ids)}
        self.is_block = np.array([c.get("type") == "block" for c in targets], dtype=bool)
        self.is_rect = np.array([c.get("type") == "groupRect" for c in targets], dtype=bool)
        obb = np.array([_collidable_obb(chart, c) for c in targets], dtype=np.float64).reshape(-1, 5)
        self.cx, self.cy, self.hw, self.hh = obb[:, 0], obb[:, 1], obb[:, 2], obb[:, 3]
        rad = np.radians(obb[:, 4])
        self.cos, self.sin = np.cos(rad), np.sin(rad)
        # Only moving things change the distance field.
        self.key = tuple(np.round(obb[:, :2] / (GRID_WORLD / 2)).astype(int).ravel()) + tuple(
            np.round(obb[:, 2:4]).astype(int).ravel()
        ) + tuple(np.round(obb[:, 4]).astype(int))

    def near(self, x: float, y: float, radius: float) -> np.ndarray:
        """Indices of objects that could touch a disk of `radius` around (x, y)."""
        reach = np.hypot(self.hw, self.hh) + radius
        return np.flatnonzero(np.hypot(self.cx - x, self.cy - y) <= reach)

    def subset(self, idx: np.ndarray) -> "LiveGeometry":
        sub = object.__new__(LiveGeometry)
        sub.ids = [self.ids[i] for i in idx]
        sub.index = {i: k for k, i in enumerate(sub.ids)}
        for name in ("is_block", "is_rect", "cx", "cy", "hw", "hh", "cos", "sin"):
            setattr(sub, name, getattr(self, name)[idx])
        sub.key = self.key
        return sub

    def inside(self, px: np.ndarray, py: np.ndarray, pad: float = 0.0) -> np.ndarray:
        """bool [P, N]: point p lies in object n (edges inclusive)."""
        dx = np.asarray(px)[:, None] - self.cx
        dy = np.asarray(py)[:, None] - self.cy
        lx = dx * self.cos + dy * self.sin
        ly = -dx * self.sin + dy * self.cos
        return (np.abs(lx) <= self.hw + pad) & (np.abs(ly) <= self.hh + pad)

    def clearance(self, px: np.ndarray, py: np.ndarray, avoid: np.ndarray, required: list[int]) -> np.ndarray:
        """Per point: distance to the nearest `avoid` object, or to the edge
        of a `required` object it is inside (world units; inf if none)."""
        dx = np.asarray(px)[:, None] - self.cx
        dy = np.asarray(py)[:, None] - self.cy
        lx = np.abs(dx * self.cos + dy * self.sin)
        ly = np.abs(-dx * self.sin + dy * self.cos)
        out = np.full(len(px), np.inf)
        if avoid.any():
            gap = np.hypot(np.clip(lx - self.hw, 0, None), np.clip(ly - self.hh, 0, None))
            out = np.minimum(out, gap[:, avoid].min(axis=1))
        if required:
            room = np.minimum(self.hw - lx, self.hh - ly)
            out = np.minimum(out, room[:, required].min(axis=1))
        return out

    def local_to_world(self, k: int, lx: float, ly: float) -> tuple[float, float]:
        return (
            self.cx[k] + lx * self.cos[k] - ly * self.sin[k],
            self.cy[k] + lx * self.sin[k] + ly * self.cos[k],
        )

    def points_in(self, k: int, n: int = CLICK_GRID, inset: float = 0.8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """n x n grid of points inside object k -> (world x, world y, local (lx, ly))."""
        u = (np.arange(n) + 0.5) / n * 2 - 1
        lx, ly = np.meshgrid(u * self.hw[k] * inset, u * self.hh[k] * inset, indexing="ij")
        lx, ly = lx.ravel(), ly.ravel()
        wx = self.cx[k] + lx * self.cos[k] - ly * self.sin[k]
        wy = self.cy[k] + lx * self.sin[k] + ly * self.cos[k]
        return wx, wy, np.stack([lx, ly], axis=1)


def first_point_scored(geom: LiveGeometry, row: np.ndarray) -> np.ndarray:
    """trailSweep startedOnBlock: a first point on a block doesn't fire the
    group rects it also touches."""
    if (row & geom.is_block).any():
        return row & ~geom.is_rect
    return row


@dataclass
class Stroke:
    start_uid: int
    start_id: str
    start_local: tuple[float, float]  # point in the start object's own frame
    hit_uids: list[int]
    # Per hit: the objects covering it (a click would score them). The
    # cursor must stay inside all of them until the hit — leaving one and
    # letting it come back over the cursor (a carried block) is a fresh
    # entry, i.e. a Wrong.
    covers: dict[int, frozenset[str]]
    t_start: float
    t_end: float  # last hit's note time
    t_take_over: float  # the guide leads from here (start note visible, earlier notes done)


@dataclass
class Guide:
    """What the expert should do this tick, and what the policy is shown."""

    waypoint: tuple[float, float]  # normalized cursor target for this tick
    toggle: bool  # flip trail held this tick
    active: bool = True


@dataclass
class TrailPlan:
    strokes: list[Stroke] = field(default_factory=list)


def _click_offenders(chart, ev: dict) -> tuple[frozenset[str], LiveGeometry] | None:
    """Smallest set of other objects a click on this note's object must
    also score (frozenset() = cleanly clickable). None if the object isn't
    live at its note time."""
    geom = LiveGeometry(chart, float(ev["time"]))
    k = geom.index.get(ev["id"])
    if k is None:
        return None
    wx, wy, _ = geom.points_in(k)
    rows = geom.inside(wx, wy)
    best = None
    for row in rows:
        scored = first_point_scored(geom, row)
        others = frozenset(geom.ids[j] for j in np.flatnonzero(scored) if j != k)
        if best is None or len(others) < len(best):
            best = others
            if not best:
                break
    return best, geom


def click_offender_counts(chart) -> dict[int, int]:
    """uid -> how many other objects the cleanest click on that note's
    object at its beat would also score (0 = cleanly clickable). A level
    fact for the observation (rl_env lookahead), cached on the chart."""
    cached = chart.__dict__.get("_click_offender_counts")
    if cached is None:
        cached = {uid: len(ids) for uid, ids in click_offender_sets(chart).items()}
        chart._click_offender_counts = cached
    return cached


def click_offender_sets(chart) -> dict[int, frozenset[str]]:
    """uid -> the other objects the cleanest click on that note's object at
    its beat would also score. Cached on the chart."""
    cached = chart.__dict__.get("_click_offender_sets")
    if cached is None:
        cached = {}
        for ev in chart.events:
            off = _click_offenders(chart, ev)
            cached[ev["_uid"]] = frozenset() if off is None else off[0]
        chart._click_offender_sets = cached
    return cached


def cover_relation(chart, uid_first: int, uid_later: int) -> float:
    """Share of the objects covering note `uid_later` (click_offender_sets)
    that also cover the object of note `uid_first` at that note's beat —
    e.g. 只因為你那渴望自由的心臟🫀: the block of the note before the drums
    is the one that will lie over each drum (1.0); 迷宮🗣️🔥: the rects over
    the goal also enclose the start block (1.0). A geometric fact about two
    notes of the level (rl_env lookahead), cached on the chart; 0 when
    `uid_later` is cleanly clickable."""
    cache = chart.__dict__.setdefault("_cover_relation", {})
    key = (uid_first, uid_later)
    if key in cache:
        return cache[key]
    covers = click_offender_sets(chart)[uid_later]
    value = 0.0
    if covers:
        first = chart.events[uid_first]
        geom = LiveGeometry(chart, float(first["time"]))
        k = geom.index.get(first["id"])
        if k is not None:
            row = geom.inside(np.array([geom.cx[k]]), np.array([geom.cy[k]]))[0]
            inside = {geom.ids[j] for j in np.flatnonzero(row)}
            value = len(covers & inside) / len(covers)
    cache[key] = value
    return value


def build_plan(chart) -> TrailPlan:
    """A stroke for every run of consecutive notes a click can't take
    cleanly, started on the note right before the run — provided a first
    point on that note's object (scoring it alone) is already inside every
    object that covers the run's first note. Candidates only: the expert's
    simulation (validate_plan) keeps the strokes that actually work."""
    events = sorted(chart.events, key=lambda e: (e["time"], e["_uid"]))
    offenders = [_click_offenders(chart, ev) for ev in events]
    plan = TrailPlan()
    i = 1
    while i < len(events):
        off = offenders[i]
        if off is None or not off[0]:
            i += 1
            continue
        start_ev = events[i - 1]
        if start_ev["time"] >= events[i]["time"] or any(s.start_uid == start_ev["_uid"] for s in plan.strokes):
            i += 1
            continue
        geom = LiveGeometry(chart, float(start_ev["time"]))
        k = geom.index.get(start_ev["id"])
        if k is None:
            i += 1
            continue
        wx, wy, local = geom.points_in(k)
        rows = geom.inside(wx, wy)
        chosen = None
        for p, row in enumerate(rows):
            scored = first_point_scored(geom, row)
            if frozenset(geom.ids[j] for j in np.flatnonzero(scored)) != {geom.ids[k]}:
                continue
            touched = frozenset(geom.ids[j] for j in np.flatnonzero(row))
            if off[0] <= touched:
                # Prefer the point nearest the object's center.
                d = float(np.hypot(*local[p]))
                if chosen is None or d < chosen[0]:
                    chosen = (d, p, touched)
        if chosen is None:
            i += 1
            continue
        allowed = set(chosen[2])
        hits, covers = [], {}
        j = i
        while j < len(events) and offenders[j] is not None and offenders[j][0] and offenders[j][0] <= allowed:
            hits.append(events[j])
            covers[events[j]["_uid"]] = offenders[j][0]
            allowed.add(events[j]["id"])
            j += 1
        earlier = [e["time"] for e in events[: i - 1]]
        take_over = float(start_ev["time"]) - config.APPROACH_TIME_MS
        if earlier:
            take_over = max(take_over, max(earlier) + config.HIT_WINDOW_MS)
        plan.strokes.append(
            Stroke(
                start_uid=start_ev["_uid"],
                start_id=start_ev["id"],
                start_local=tuple(local[chosen[1]]),
                hit_uids=[e["_uid"] for e in hits],
                covers=covers,
                t_start=float(start_ev["time"]),
                t_end=float(hits[-1]["time"]),
                t_take_over=min(take_over, float(start_ev["time"])),
            )
        )
        i = j
    return plan


class StrokeGuide:
    """Per-episode stroke driver: reads the Judge's live state and returns
    this tick's Guide, or None when no stroke is (or can still be) in play
    and the ordinary click rule applies."""

    def __init__(self, chart, plan: TrailPlan, reach_norm: float):
        self.chart = chart
        self.plan = plan
        self.span = chart.world_span
        self.reach_world = reach_norm * self.span
        b = chart.bounds
        self.x0, self.y0 = b["minX"], b["minY"]
        self.nx = int(np.ceil((b["maxX"] - b["minX"]) / GRID_WORLD))
        self.ny = int(np.ceil((b["maxY"] - b["minY"]) / GRID_WORLD))
        gx = self.x0 + (np.arange(self.nx) + 0.5) * GRID_WORLD
        gy = self.y0 + (np.arange(self.ny) + 0.5) * GRID_WORLD
        mx, my = np.meshgrid(gx, gy, indexing="ij")
        self.grid_x, self.grid_y = mx.ravel(), my.ravel()
        self._fields: dict = {}
        self._event_by_uid = {e["_uid"]: e for e in chart.events}
        # Candidate moves: a disk of radius one full-speed tick.
        # Small steps (0.5-3 world units) too, so a fallback move can be as
        # gentle as the paced one it replaces (5-unit minimum steps made the
        # cursor twitch when a drum came into reach).
        r = np.concatenate([np.array([0.5, 1.0, 2.0, 3.0]) / self.reach_world, np.linspace(0.0, 1.0, 9)[1:]])
        a = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
        rr, aa = np.meshgrid(r, a, indexing="ij")
        self._moves = np.concatenate(
            [np.zeros((1, 2)), np.stack([(rr * np.cos(aa)).ravel(), (rr * np.sin(aa)).ravel()], axis=1)]
        ) * self.reach_world * 0.999
        self._slow = np.hypot(self._moves[:, 0], self._moves[:, 1]) <= self.reach_world * STROKE_SPEED_FRACTION + 1e-6
        self.lost: set[int] = set()

    def _to_world(self, p):
        return self.chart.world_xy(*p)

    def _to_norm(self, wx, wy):
        return self.chart.normalized_xy({"x": wx, "y": wy})

    def _stroke_at(self, t_ms: float, judge) -> Stroke | None:
        for idx, s in enumerate(self.plan.strokes):
            if idx in self.lost:
                continue
            if t_ms < s.t_take_over:
                return None
            if all(u in judge.resolved_uids for u in s.hit_uids):
                continue
            return s
        return None

    def _field(self, geom: LiveGeometry, allowed: frozenset[str], cover: frozenset[str],
               target_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        key = (geom.key, allowed, cover, target_id)
        cached = self._fields.get(key)
        if cached is not None:
            return cached
        k = geom.index.get(target_id)
        if k is None:
            return None
        n = self.nx * self.ny
        free = np.ones(n, dtype=bool)
        for j, oid in enumerate(geom.ids):
            if oid not in allowed and oid != target_id:
                free[self._object_cells(geom, j, CLEARANCE_WORLD)] = False
        for c in cover:
            if c in geom.index:
                keep = np.zeros(n, dtype=bool)
                keep[self._object_cells(geom, geom.index[c], -CLEARANCE_WORLD)] = True
                free &= keep
        goal = np.zeros(n, dtype=bool)
        goal[self._object_cells(geom, k, CLEARANCE_WORLD)] = True
        goal &= free
        field = self._centerline_distance(free, goal)  # (cost, length)
        # Each field is two float arrays over the whole grid (1.6 MB on
        # 迷宮's); only the current few are ever reused.
        if len(self._fields) > 8:
            self._fields.clear()
        self._fields[key] = field
        return field

    def _object_cells(self, geom: LiveGeometry, k: int, pad: float) -> np.ndarray:
        """Flat indices of the grid cells inside object k (padded), tested
        only over its bounding box. Cached for objects no track carries
        (their pose never changes); a carried object's pose changes every
        tick, and caching those — as full-grid masks, up to 20000 of them —
        filled memory until the OS killed training (bc11)."""
        carried = self.__dict__.get("_carried")
        if carried is None:
            carried = self._carried = {c["id"] for c in self.chart.collidables if c.get("carriedByTrackId")}
        oid = geom.ids[k]
        key = (oid, pad)
        cache = self.__dict__.setdefault("_cells", {})
        if oid not in carried:
            cells = cache.get(key)
            if cells is not None:
                return cells
        r = float(np.hypot(geom.hw[k], geom.hh[k])) + abs(pad) + GRID_WORLD
        i0 = max(0, int((geom.cx[k] - r - self.x0) / GRID_WORLD))
        i1 = min(self.nx, int((geom.cx[k] + r - self.x0) / GRID_WORLD) + 1)
        j0 = max(0, int((geom.cy[k] - r - self.y0) / GRID_WORLD))
        j1 = min(self.ny, int((geom.cy[k] + r - self.y0) / GRID_WORLD) + 1)
        cells = np.zeros(0, dtype=np.int64)
        if i1 > i0 and j1 > j0:
            ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            box = (ii * self.ny + jj).ravel()
            sub = geom.subset(np.array([k]))
            cells = box[sub.inside(self.grid_x[box], self.grid_y[box], pad=pad)[:, 0]]
        if oid not in carried:
            cache[key] = cells
        return cells
        if len(cache) > 20000:
            cache.clear()
        r = float(np.hypot(geom.hw[k], geom.hh[k])) + abs(pad) + GRID_WORLD
        i0 = max(0, int((geom.cx[k] - r - self.x0) / GRID_WORLD))
        i1 = min(self.nx, int((geom.cx[k] + r - self.x0) / GRID_WORLD) + 1)
        j0 = max(0, int((geom.cy[k] - r - self.y0) / GRID_WORLD))
        j1 = min(self.ny, int((geom.cy[k] + r - self.y0) / GRID_WORLD) + 1)
        mask = np.zeros(self.nx * self.ny, dtype=bool)
        if i1 > i0 and j1 > j0:
            ii, jj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing="ij")
            cells = (ii * self.ny + jj).ravel()
            sub = geom.subset(np.array([k]))
            mask[cells] = sub.inside(self.grid_x[cells], self.grid_y[cells], pad=pad)[:, 0]
        cache[key] = mask
        return mask

    def _centerline_distance(self, free: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """Path length (in cells) from every free cell to the goal, where a
        cell closer than CENTER_CLEARANCE_WORLD to anything the stroke must
        not touch costs extra — so the path runs down the middle of a
        corridor. 迷宮's corridors are 30 world units wide; the shortest
        path hugged the walls at ~6, and bc8's first stroke attempts
        brushed them 9 times in the first second."""
        free2 = free.reshape(self.nx, self.ny)
        clear = ndimage.distance_transform_edt(free2) * GRID_WORLD
        cell_cost = 1.0 + CENTER_PENALTY * np.clip(CENTER_CLEARANCE_WORLD - clear, 0.0, None) / GRID_WORLD
        cell_cost = cell_cost.reshape(-1)
        n = self.nx * self.ny
        ix, iy = np.divmod(np.arange(n), self.ny)
        rows, cols, weights, lengths = [], [], [], []
        for sx, sy in ((1, 0), (0, 1), (1, 1), (1, -1)):
            valid = (ix + sx >= 0) & (ix + sx < self.nx) & (iy + sy >= 0) & (iy + sy < self.ny)
            a = np.flatnonzero(valid)
            b = (ix[a] + sx) * self.ny + (iy[a] + sy)
            ok = free[a] & free[b]
            if sx and sy:
                # No cutting a blocked corner.
                ok &= free2[ix[a] + sx, iy[a]] & free2[ix[a], iy[a] + sy]
            a, b = a[ok], b[ok]
            step = 1.414 if sx and sy else 1.0
            w = 0.5 * (cell_cost[a] + cell_cost[b]) * step
            rows += [a, b]
            cols += [b, a]
            weights += [w, w]
            lengths += [np.full(len(a), step), np.full(len(a), step)]
        sources = np.flatnonzero(goal)
        if len(sources) == 0:
            return np.full(n, np.inf), np.full(n, np.inf)
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        graph = sparse.csr_matrix((np.concatenate(weights), (rows, cols)), shape=(n, n))
        # One super-source: dijkstra with min_only over every goal cell.
        cost = csgraph.dijkstra(graph, directed=True, indices=sources, min_only=True)
        # Plain path length (cells) to the goal, for pacing.
        steps = sparse.csr_matrix((np.concatenate(lengths), (rows, cols)), shape=(n, n))
        length = csgraph.dijkstra(steps, directed=True, indices=sources, min_only=True)
        return cost, length

    def _field_at(self, field: np.ndarray, wx: np.ndarray, wy: np.ndarray) -> np.ndarray:
        ix = np.clip(((wx - self.x0) / GRID_WORLD).astype(int), 0, self.nx - 1)
        iy = np.clip(((wy - self.y0) / GRID_WORLD).astype(int), 0, self.ny - 1)
        return field[ix * self.ny + iy]

    def _geometry(self, t_ms: float) -> LiveGeometry:
        cache = self.__dict__.setdefault("_geoms", {})
        geom = cache.get(t_ms)
        if geom is None:
            if len(cache) > 4:
                cache.clear()
            geom = cache[t_ms] = LiveGeometry(self.chart, t_ms)
        return geom

    def _ride(self, t_ms: float, cx: float, cy: float, cover: frozenset[str]) -> tuple[float, float]:
        """Where the cursor lands if it keeps its position in the cover's
        own frame from the previous tick to this one (identity when the
        covers don't move)."""
        prev, now = self._geometry(t_ms - config.DT_MS), self._geometry(t_ms)
        best, moved = (cx, cy), 0.0
        for c in cover:
            if c not in prev.index or c not in now.index:
                continue
            i, j = prev.index[c], now.index[c]
            dx, dy = cx - prev.cx[i], cy - prev.cy[i]
            lx = (dx * prev.cos[i] + dy * prev.sin[i]) * now.hw[j] / max(prev.hw[i], 1e-9)
            ly = (-dx * prev.sin[i] + dy * prev.cos[i]) * now.hh[j] / max(prev.hh[i], 1e-9)
            x, y = now.local_to_world(j, lx, ly)
            d = float(np.hypot(x - cx, y - cy))
            if d > moved:
                best, moved = (float(x), float(y)), d
        return best

    def _check(self, geom: LiveGeometry, cx, cy, ex, ey, forbidden, k, required, step_in):
        """Per candidate end: (safe, ends inside the target)."""
        s = np.linspace(0.0, 1.0, SEGMENT_SAMPLES + 1)[1:]
        px = cx + (ex[:, None] - cx) * s[None, :]
        py = cy + (ey[:, None] - cy) * s[None, :]
        inside = geom.inside(px.ravel(), py.ravel(), pad=CLEARANCE_WORLD).reshape(len(ex), len(s), -1)
        # An object the cursor is already within the safety pad of (it just
        # left it) is tested exactly: otherwise every move away from it
        # counts as touching it, and the fallback search jumped ~15 units.
        near = geom.inside(np.array([cx]), np.array([cy]), pad=CLEARANCE_WORLD)[0]
        if near.any():
            exact = geom.inside(px.ravel(), py.ravel()).reshape(len(ex), len(s), -1)
            inside[:, :, near] = exact[:, :, near]
        touched = inside.any(axis=1)
        new_other = touched & forbidden
        new_other[:, k] = False
        ok = ~new_other.any(axis=1)
        if required:
            ok &= geom.inside(px.ravel(), py.ravel(), pad=-CLEARANCE_WORLD).reshape(len(ex), len(s), -1)[
                :, :, required
            ].all(axis=(1, 2))
        if not step_in:
            ok &= ~(touched[:, k] & forbidden[k])
        in_target = geom.inside(ex, ey, pad=-3.0)[:, k]
        return ok, in_target

    def _navigate(self, t_ms: float, cursor_norm, allowed: frozenset[str], cover: frozenset[str],
                  target_uid: int) -> tuple[float, float] | None:
        """Next cursor point (normalized) inside the stroke: never enter
        anything new, never leave the objects covering the target, ride a
        moving cover, and follow the centerline path at the pace that
        reaches the target on its beat; glide into it on the beat."""
        ev = self._event_by_uid[target_uid]
        full = self._geometry(t_ms)
        if ev["id"] not in full.index:
            return None
        cx, cy = self._to_world(cursor_norm)
        bx, by = self._ride(t_ms, cx, cy, cover)
        idx = full.near(bx, by, self.reach_world + CLEARANCE_WORLD + float(np.hypot(bx - cx, by - cy)))
        idx = np.union1d(idx, [full.index[ev["id"]]] + [full.index[c] for c in cover if c in full.index])
        geom = full.subset(idx.astype(int))
        k = geom.index[ev["id"]]
        required = [geom.index[c] for c in cover if c in geom.index]
        forbidden = np.array([i not in allowed for i in geom.ids], dtype=bool)
        note_t = float(ev["time"])
        tick = config.DT_MS / 1000.0
        # Start gliding in early enough to cover the standoff at the top
        # stroke pace, landing a few ms before the beat (still Perfect).
        glide_ms = STANDOFF_WORLD / MAX_STROKE_SPEED * 1000.0 + config.DT_MS
        step_in = t_ms >= note_t - glide_ms

        field = self._field(full, allowed, cover, ev["id"])
        desired = None
        if step_in:
            # Glide straight at the target's center.
            tx, ty = geom.cx[k], geom.cy[k]
            d = float(np.hypot(tx - bx, ty - by))
            step = min(d, MAX_STROKE_SPEED * tick)
            desired = (bx + (tx - bx) / max(d, 1e-9) * step, by + (ty - by) / max(d, 1e-9) * step)
        elif field is not None and np.isfinite(field[0][self._cell(bx, by)]):
            track = self._track_for(field, bx, by)
            pos = self._track_progress(track, bx, by)
            need = track["cum"][-1] - pos
            if need > 0.5:
                t_left = note_t - ARRIVE_EARLY_MS - t_ms
                speed = need / (t_left / 1000.0) if t_left > 0 else MAX_STROKE_SPEED
                speed = float(np.clip(speed, MIN_STROKE_SPEED, MAX_STROKE_SPEED))
                pos = min(pos + speed * tick, track["cum"][-1])
            track["s"] = pos
            desired = self._track_point(track, pos)
        elif required:
            # Not reachable inside the covers yet (a track still carrying the
            # cover toward it): settle toward the covers' center and ride.
            mx, my = geom.cx[required].mean(), geom.cy[required].mean()
            d = float(np.hypot(mx - bx, my - by))
            step = min(d, SETTLE_SPEED * tick)
            desired = (bx + (mx - bx) / max(d, 1e-9) * step, by + (my - by) / max(d, 1e-9) * step)

        moving_cover = float(np.hypot(bx - cx, by - cy)) > 1e-3
        if desired is not None and moving_cover and not step_in:
            # Riding a moving cover, the route is replanned whenever the
            # geometry changes, each time from a slightly different cell:
            # smooth the move relative to the cover instead.
            rel = np.array([desired[0] - bx, desired[1] - by])
            prev = self.__dict__.get("_rel_move")
            if prev is not None and prev[0] == target_uid:
                rel = RIDE_SMOOTHING * prev[1] + (1.0 - RIDE_SMOOTHING) * rel
            self._rel_move = (target_uid, rel)
            desired = (bx + float(rel[0]), by + float(rel[1]))
        if desired is not None:
            ex, ey = np.array([desired[0]]), np.array([desired[1]])
            ok, in_target = self._check(geom, cx, cy, ex, ey, forbidden, k, required, step_in)
            if ok[0]:
                return self._to_norm(desired[0], desired[1])

        # The direct move isn't safe: search moves around the ride point,
        # preferring the ones closest to the intended move.
        ends = np.array([bx, by]) + self._moves
        ex = np.clip(ends[:, 0], self.x0, self.x0 + self.span)
        ey = np.clip(ends[:, 1], self.y0, self.y0 + self.span)
        ok, in_target = self._check(geom, cx, cy, ex, ey, forbidden, k, required, step_in)
        if step_in and (ok & in_target).any():
            hit = np.flatnonzero(ok & in_target)
            pick = int(hit[np.argmin(np.hypot(ex[hit] - bx, ey[hit] - by))])
            return self._to_norm(ex[pick], ey[pick])
        if not step_in and (ok & self._slow).any():
            ok &= self._slow
        if not ok.any():
            return None
        clear = geom.clearance(ex, ey, forbidden, required)
        shortfall = np.clip(CENTER_CLEARANCE_WORLD - clear, 0.0, None) / GRID_WORLD
        if desired is not None:
            # Closest safe move to the intended one, but never standing still
            # when the path goes on (field progress); clearance only breaks
            # near-ties (weighting it like the path search made the cursor
            # jump ~15 units off the path and back, tick after tick).
            cost = np.hypot(ex - desired[0], ey - desired[1]) / GRID_WORLD + 0.1 * shortfall
        else:
            cost = CENTER_PENALTY * shortfall
        if field is not None and not step_in:
            progress = self._field_at(field[0], ex, ey)
            if np.isfinite(progress[ok]).any():
                cost = cost + progress
        cost[~ok] = np.inf
        pick = int(np.argmin(cost))
        if not np.isfinite(cost[pick]):
            return None
        return self._to_norm(ex[pick], ey[pick])

    def _cell(self, wx: float, wy: float) -> int:
        ix = int(np.clip((wx - self.x0) / GRID_WORLD, 0, self.nx - 1))
        iy = int(np.clip((wy - self.y0) / GRID_WORLD, 0, self.ny - 1))
        return ix * self.ny + iy

    def _descend(self, cost: np.ndarray, cur: int) -> int:
        """The lowest-cost 8-neighbor of `cur`, or `cur` at the bottom."""
        ix, iy = divmod(cur, self.ny)
        best, best_cost = cur, cost[cur]
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                jx, jy = ix + sx, iy + sy
                if (sx or sy) and 0 <= jx < self.nx and 0 <= jy < self.ny:
                    j = jx * self.ny + jy
                    if cost[j] < best_cost:
                        best, best_cost = j, cost[j]
        return best

    def _track_for(self, field, wx: float, wy: float) -> dict:
        """The route being followed: the least-cost path from the cursor's
        cell down to the standoff (STANDOFF_WORLD of path left), smoothed,
        as a polyline with cumulative arc length. Planned once and kept
        while the field is the same and the cursor stays near it — per-tick
        replanning (cell boundaries, carrot lengths, safety retries) made
        the cursor step back and forth."""
        track = self.__dict__.get("_track")
        if track is not None and track["field"] is field:
            px, py = self._track_point(track, track["s"])
            if np.hypot(px - wx, py - wy) <= TRACK_REPLAN_WORLD:
                return track
        cost, length = field
        cell = self._cell(wx, wy)
        pts = [(wx, wy)]
        for _ in range(4 * (self.nx + self.ny)):
            if length[cell] * GRID_WORLD <= STANDOFF_WORLD or cost[cell] == 0.0:
                break
            nxt = self._descend(cost, cell)
            if nxt == cell:
                break
            cell = nxt
            ix, iy = divmod(cell, self.ny)
            pts.append((self.x0 + (ix + 0.5) * GRID_WORLD, self.y0 + (iy + 0.5) * GRID_WORLD))
        pts = np.array(pts, dtype=np.float64)
        for _ in range(TRACK_SMOOTHING_PASSES):
            if len(pts) > 2:
                pts[1:-1] = 0.25 * pts[:-2] + 0.5 * pts[1:-1] + 0.25 * pts[2:]
        seg = np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1]))
        track = {"field": field, "pts": pts, "cum": np.concatenate([[0.0], np.cumsum(seg)]), "s": 0.0}
        self._track = track
        return track

    @staticmethod
    def _track_point(track: dict, s: float) -> tuple[float, float]:
        pts, cum = track["pts"], track["cum"]
        if len(pts) == 1:
            return float(pts[0, 0]), float(pts[0, 1])
        i = int(np.clip(np.searchsorted(cum, s) - 1, 0, len(pts) - 2))
        f = 0.0 if cum[i + 1] <= cum[i] else (s - cum[i]) / (cum[i + 1] - cum[i])
        f = float(np.clip(f, 0.0, 1.0))
        p = pts[i] + f * (pts[i + 1] - pts[i])
        return float(p[0]), float(p[1])

    def _track_progress(self, track: dict, wx: float, wy: float) -> float:
        """Arc length of the cursor's projection onto the track near where it
        was — never behind the last position (no stepping back)."""
        pts, cum, s0 = track["pts"], track["cum"], track["s"]
        if len(pts) == 1:
            return 0.0
        lo = int(np.clip(np.searchsorted(cum, s0 - TRACK_REPLAN_WORLD) - 1, 0, len(pts) - 2))
        hi = int(np.clip(np.searchsorted(cum, s0 + 2 * TRACK_REPLAN_WORLD), lo + 1, len(pts) - 1))
        a, b = pts[lo:hi], pts[lo + 1 : hi + 1]
        ab = b - a
        denom = np.maximum((ab ** 2).sum(1), 1e-12)
        f = np.clip(((np.array([wx, wy]) - a) * ab).sum(1) / denom, 0.0, 1.0)
        proj = a + f[:, None] * ab
        j = int(np.argmin(np.hypot(proj[:, 0] - wx, proj[:, 1] - wy)))
        s = cum[lo + j] + f[j] * (cum[lo + j + 1] - cum[lo + j])
        return float(max(s, s0))

    def guide(self, t_ms: float, cursor_norm, trail_held: bool, judge) -> Guide | None:
        while True:
            stroke = self._stroke_at(t_ms, judge)
            if stroke is None:
                return None
            idx = self.plan.strokes.index(stroke)
            started = stroke.start_uid in judge.resolved_uids
            if not started:
                if trail_held or t_ms > stroke.t_start + config.HIT_WINDOW_MS:
                    self.lost.add(idx)
                    continue
                geom = LiveGeometry(self.chart, t_ms)
                k = geom.index.get(stroke.start_id)
                if k is None:
                    self.lost.add(idx)
                    continue
                wx, wy = geom.local_to_world(k, *stroke.start_local)
                point = self._to_norm(wx, wy)
                cx, cy = self._to_world(cursor_norm)
                close = np.hypot(wx - cx, wy - cy) <= self.reach_world * 0.999
                on_beat = t_ms >= stroke.t_start - STROKE_START_LEAD_MS
                return Guide(waypoint=point, toggle=bool(close and on_beat))
            if not trail_held:
                # The start note went some other way (clicked, or the stroke
                # was released): the covering objects are no longer held.
                self.lost.add(idx)
                continue
            pending = [u for u in stroke.hit_uids if u not in judge.resolved_uids]
            allowed = frozenset(judge._inside_collidables)
            point = self._navigate(t_ms, cursor_norm, allowed, stroke.covers[pending[0]] & allowed, pending[0])
            if point is None:
                self.lost.add(idx)
                return Guide(waypoint=cursor_norm, toggle=True, active=False)
            return Guide(waypoint=point, toggle=False)


def validate_plan(chart, plan: TrailPlan, run_episode) -> TrailPlan:
    """Keep only strokes the expert actually lands: each stroke's notes all
    hit, and no Wrong between its start and its last note. run_episode(plan)
    -> Judge after a full expert episode."""
    judge = run_episode(plan)
    grade = {e["eventId"]: e["judgment"] for e in judge.log if "eventId" in e}
    wrong_times = [e["time"] for e in judge.log if e["judgment"] == "Wrong"]
    kept = TrailPlan()
    for s in plan.strokes:
        uids = [s.start_uid] + s.hit_uids
        hits_ok = all(grade.get(u) in ("Perfect", "Good", "Bad") for u in uids)
        clean = not any(s.t_start - config.HIT_WINDOW_MS <= t <= s.t_end + config.HIT_WINDOW_MS for t in wrong_times)
        if hits_ok and clean:
            kept.strokes.append(s)
    return kept


def plan_cache_path(chart) -> str | None:
    """Sidecar next to the chart's events.json (precompute_trail_plans.py)."""
    path = getattr(chart, "events_path", None)
    return None if path is None else path[: -len(".events.json")] + ".trailplan.json"


def _load_cached_plan(path: str, chart) -> TrailPlan | None:
    import json
    import os

    if not os.path.exists(path) or os.path.getmtime(path) < os.path.getmtime(chart.events_path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except (OSError, ValueError):
        return None  # e.g. a write cut short by a killed job: recompute
    return TrailPlan(strokes=[
        Stroke(**{**s, "start_local": tuple(s["start_local"]),
                  "covers": {int(k): frozenset(v) for k, v in s["covers"].items()}})
        for s in raw["strokes"]
    ])


def save_plan(plan: TrailPlan, path: str) -> None:
    import json
    from dataclasses import asdict

    strokes = []
    for s in plan.strokes:
        d = asdict(s)
        d["covers"] = {str(k): sorted(v) for k, v in s.covers.items()}
        strokes.append(d)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"strokes": strokes}, f)


def get_trail_plan(chart) -> TrailPlan:
    """build_plan + validate_plan, cached on the chart and in a sidecar
    file (validation plays a whole expert episode: seconds per chart, which
    adds up over hundreds of generated levels)."""
    cached = chart.__dict__.get("_trail_plan")
    if cached is not None:
        return cached
    path = plan_cache_path(chart)
    if path is not None:
        loaded = _load_cached_plan(path, chart)
        if loaded is not None:
            chart._trail_plan = loaded
            return loaded
    chart._trail_plan = _compute_trail_plan(chart)
    if path is not None:
        save_plan(chart._trail_plan, path)
    return chart._trail_plan


def _compute_trail_plan(chart) -> TrailPlan:
    """Validation plays one identity-view expert episode with the
    candidate strokes."""
    candidates = build_plan(chart)
    if not candidates.strokes:
        return candidates

    from bc_expert import ScriptedExpert
    from rl_env import TrailRLEnv

    def run_episode(plan: TrailPlan):
        chart._trail_plan = plan
        env = TrailRLEnv(chart)
        obs = env.reset()
        expert = ScriptedExpert()
        while not env.done:
            obs, *_ = env.step(*expert.act(env, obs))
        return env.judge

    try:
        return validate_plan(chart, candidates, run_episode)
    finally:
        del chart._trail_plan
