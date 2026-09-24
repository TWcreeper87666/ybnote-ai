"""Obstacle-aware cursor TRAINING LABELS only — never used at inference.

target_xy() (cursor_readout.py) points straight at the current target,
which is exactly wrong inside a maze-style chart: the straight line crosses
walls the real game would judge Wrong for touching (see reward.py's
_resolve_trail_collisions / TRAIN_DIARY.md 2026-09-24 "trail obstacle
Wrong"). This module walks a virtual pursuer at the SAME speed cap
SmoothedCursor enforces at inference, replanning a BFS grid path around
collidables whenever the target changes, so the supervised y_cursor label
itself already routes around walls — the model then LEARNS that routing via
ordinary MSE regression, the same way it already learns "where to aim" from
target_xy(). This is training-time label construction, not output patching:
at inference the model's own cursor_head prediction is used as-is, nothing
here runs live (see TRAIN_DIARY.md 2026-09-24 "no output patching").

For every chart with no collidables a straight line could ever cross (every
existing chart except the maze-style ones), this degenerates to exactly
target_xy()'s old straight-line behavior — nothing changes for those.

The occupancy grid uses EXACT integer cell indices computed by encodeFrames.
js's computeCollidableGrid, straight from each collidable's raw world x/y/w/
h — not re-derived here from this chart's normalized (0..1) coordinates.
That distinction mattered: a maze's wall segments tile edge-to-edge with
zero gap, and re-deriving cell bounds by dividing normalized coordinates by
an arbitrary resolution reintroduces float drift + a resolution/tile-unit
mismatch that can spuriously merge two edge-to-edge walls and seal a
doorway that's genuinely open in the source geometry (see TRAIN_DIARY.md
2026-09-24 "trail path label" for the full debugging trail — cost a lot of
trial and error to track down)."""

import bisect
from collections import deque

import numpy as np
import torch

import config
from cursor_readout import target_info
from obstacles import MAX_OBSTACLES, OBSTACLE_FEATURE_DIM, build_collidable_arrays, nearby_obstacle_features
from reward import _segment_intersects_rect


class Grid:
    """World-space integer tile grid for one chart, built from
    chart.collidable_grid + each collidable's gridCol0/Row0/Col1/Row1 (see
    encodeFrames.js's computeCollidableGrid). `occupied` is None (no
    collidables at all, or too few to imply a shared tile unit) when the
    chart has nothing to route around."""

    def __init__(self, chart):
        meta = chart.collidable_grid
        self.chart = chart
        self.meta = meta
        self.occupied = None
        if meta is None:
            return
        res_y, res_x = meta["resY"], meta["resX"]
        occ = np.zeros((res_y, res_x), dtype=bool)
        for c in chart.collidables:
            if "gridCol0" not in c:
                continue
            c0, c1 = max(0, c["gridCol0"]), min(res_x, c["gridCol1"])
            r0, r1 = max(0, c["gridRow0"]), min(res_y, c["gridRow1"])
            if c1 > c0 and r1 > r0:
                occ[r0:r1, c0:c1] = True
        self.occupied = occ

    @property
    def has_obstacles(self) -> bool:
        return self.occupied is not None and bool(self.occupied.any())

    def cell(self, norm_xy: tuple[float, float]) -> tuple[int, int]:
        wx, wy = self.chart.world_xy(*norm_xy)
        m = self.meta
        col = int(np.clip((wx - m["minX"]) / m["tileUnit"], 0, m["resX"] - 1))
        row = int(np.clip((wy - m["minY"]) / m["tileUnit"], 0, m["resY"] - 1))
        return (row, col)

    def cell_to_norm_xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        row, col = cell
        m = self.meta
        wx = m["minX"] + (col + 0.5) * m["tileUnit"]
        wy = m["minY"] + (row + 0.5) * m["tileUnit"]
        bx0, bx1 = self.chart.bounds["minX"], self.chart.bounds["maxX"]
        by0, by1 = self.chart.bounds["minY"], self.chart.bounds["maxY"]
        return (wx - bx0) / (bx1 - bx0), (wy - by0) / (by1 - by0)


def _nearest_open_cell(occupied: np.ndarray, cell: tuple[int, int], max_radius: int = 15):
    """BFS outward from `cell` (over ALL cells, blocked or not) for the
    closest open one — a more robust "escape the wall I'm standing in" than
    a fixed clear-radius: works regardless of how big the object sitting on
    `cell` actually is, at the cost of a short local search. Returns None
    if nothing opens up within `max_radius` cells (a truly sealed pocket —
    caller falls back to a straight line)."""
    res_y, res_x = occupied.shape
    if not occupied[cell]:
        return cell
    seen = {cell}
    q = deque([cell])
    while q:
        cur = q.popleft()
        if max(abs(cur[0] - cell[0]), abs(cur[1] - cell[1])) >= max_radius:
            continue
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = cur[0] + dr, cur[1] + dc
                if not (0 <= nr < res_y and 0 <= nc < res_x) or (nr, nc) in seen:
                    continue
                seen.add((nr, nc))
                if not occupied[nr, nc]:
                    return (nr, nc)
                q.append((nr, nc))
    return None


_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def shortest_path_cells(occupied: np.ndarray, start_xy_cell: tuple[int, int], goal_xy_cell: tuple[int, int]):
    """8-connected BFS from start to goal (grid cells). Returns a list of
    (row, col) cells start..goal, or None if no open path connects them.

    The TARGET note's own object is itself a collidable (every enabled
    Block/GroupRect is — see collectCollidables()), so its cell is "walled"
    in the raw occupancy grid; naively bailing out whenever start/goal sit
    inside a wall would treat every single target as unreachable. Instead,
    each endpoint that starts inside a wall is first walked out to the
    nearest genuinely open cell (_nearest_open_cell) — you don't need to
    path INTO your own target's or your own current position's body, just
    up to it."""
    res_y, res_x = occupied.shape
    start = _nearest_open_cell(occupied, start_xy_cell)
    goal = _nearest_open_cell(occupied, goal_xy_cell)
    if start is None or goal is None:
        return None
    if start == goal:
        return [start]
    prev = {start: None}
    q = deque([start])
    while q:
        cur = q.popleft()
        if cur == goal:
            break
        for dr, dc in _NEIGHBORS:
            nr, nc = cur[0] + dr, cur[1] + dc
            if not (0 <= nr < res_y and 0 <= nc < res_x) or occupied[nr, nc] or (nr, nc) in prev:
                continue
            if dr != 0 and dc != 0:
                # Diagonal step — disallow cutting across a wall CORNER: a
                # naive 8-connected grid lets you squeeze between two
                # orthogonally-blocked flanking cells even though nothing
                # occupying actual continuous space could fit through that
                # gap. Require at least one flanking orthogonal cell open.
                if occupied[cur[0] + dr, cur[1]] and occupied[cur[0], cur[1] + dc]:
                    continue
            prev[(nr, nc)] = cur
            q.append((nr, nc))
    if goal not in prev:
        return None
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = prev[node]
    path.reverse()
    return path


def _lookahead_target(chart, step: int, extend: bool):
    """(xy, is_mouse) for whichever note the cursor should be heading
    toward THIS step, or (None, False) if there's nothing to do right now.
    target_info() (proximity-argmax) wins whenever it finds something
    actively due — it's already correct for disambiguating several
    near-simultaneous notes, and is the WHOLE answer when `extend` is
    False (this is the original target_xy()-only behavior, byte-for-byte —
    see below for why it stays the default).

    `extend=True` additionally falls back to the nearest FUTURE event
    regardless of how far off (or the last event, once the chart's final
    note has passed), so the cursor has a target to walk toward across the
    ENTIRE gap, not just each note's own ~1s active window. Needed for a
    maze-style chart, where the two notes can be tens of seconds apart and
    the whole point is walking the gap between them — but applying it to
    every chart would make trail_held (see y_trail below) true almost the
    entire song instead of only near actual notes, and the new
    trail-vs-collidable Wrong check only fires while trailing (see
    reward.py's _resolve_trail_collisions): that's new Wrong-risk exposure
    for the ~30 ordinary charts with no reason to take it. Caller passes
    `extend = _needs_routing(...)`, so this only ever activates for a chart
    that actually has something to route around."""
    info = target_info(chart.input_features_at(step))
    if info is not None:
        return info["xy"], info["key"] is None
    if not extend:
        return None, False
    if not chart.events:
        return None, False
    idx = bisect.bisect_left(chart._event_times, float(chart.t_ms[step]))
    if idx >= len(chart.events):
        idx = len(chart.events) - 1
    ev = chart.events[idx]
    return chart.normalized_xy(ev), not ev["hasKeyBinding"]


# A chart needs the extended full-gap routing treatment only when a note
# is due FAR enough in the future that reaching it can't just be "the next
# attack click" — walking there while avoiding obstacles along the way is
# the entire point. 2026-09-24 survey of the 26-chart training corpus: the
# largest gap between consecutive notes on any ordinary chart was 9915ms
# (Never Gonna Give You Up); 迷宮🗣️🔥's is 57398ms. A first attempt at this
# check instead tested every consecutive note pair's straight line against
# every collidable, but almost every chart has SOME collidable with no note
# of its own (decorative objects unrelated to gameplay — found on more than
# half the corpus, 1 stray extra up to 201 on one chart) that some other
# unrelated pair of notes' straight line incidentally grazes despite the
# player never being expected to sweep a continuous trail through that
# specific stretch — that check came back true for 21 of 26 charts, clearly
# the wrong signal. Time gap is what actually distinguishes "normal chart,
# scattered decoration nearby" from "maze, walk the whole way there."
MAX_ROUTING_GAP_MS = 15000.0


def _needs_routing(chart, grid: Grid) -> bool:
    """True iff this chart has any obstacle AND asks the player to travel
    across a gap long enough that only sustained navigation (not a
    discrete click) could cover it — see MAX_ROUTING_GAP_MS above.
    Scoping `extend` (see _lookahead_target) and the BFS re-route to charts
    that pass this check keeps every other chart's y_trail/y_cursor labels
    byte-identical to the pre-obstacle-feature behavior (module
    docstring)."""
    if not grid.has_obstacles:
        return False
    times = chart._event_times
    return any(times[i + 1] - times[i] > MAX_ROUTING_GAP_MS for i in range(len(times) - 1))


# The label is built from an IDEAL (ground-truth) pursuer path, but the
# model's own real cursor_head prediction at inference is a noisy
# regression, not a perfect line — real in-game testing showed 274 Wrong
# even after gating trail to per-step "safe" segments measured against
# the ideal path, because the ideal path almost never grazes anything
# (rarely a reason to say "unsafe") while the model's ACTUAL trajectory
# wanders enough to clip nearby objects the ideal path cleared easily. This
# margin inflates every collidable's rect before the safety test, so
# "safe" means "safe with real room to spare," not "the perfect path
# threads the needle" — approximates the cursor regression's typical
# error band. See TRAIN_DIARY.md 2026-09-24 "trail path label".
_TRAIL_SAFETY_MARGIN = 0.05


def _segment_safe_to_trail(chart, step: int, prev_pos, pos) -> bool:
    """True iff moving the cursor from prev_pos to pos THIS STEP wouldn't
    touch any collidable that isn't currently due (inflated by
    _TRAIL_SAFETY_MARGIN for realistic clearance) — the real per-situation
    trail/attack trade-off (see reward.py's _resolve_trail_collisions):
    trailing through a not-due object is a free Wrong, trailing through
    only due ones (or empty space) never is. Chart-level heuristics (does
    this SONG need routing, is this note a mouse note) both turned out to
    be the wrong granularity — this is the actual local safety check."""
    if prev_pos == pos:
        return True
    t_ms = float(chart.t_ms[step])
    due_ids = {
        ev["id"]
        for ev in chart.active_events_at(t_ms, window_before_ms=config.APPROACH_TIME_MS, window_after_ms=config.HIT_WINDOW_MS)
    }
    m = _TRAIL_SAFETY_MARGIN
    for c in chart.collidables:
        if c["id"] in due_ids:
            continue
        if _segment_intersects_rect(
            prev_pos[0], prev_pos[1], pos[0], pos[1], c["x"] - m, c["y"] - m, c["w"] + 2 * m, c["h"] + 2 * m
        ):
            return False
    return True


def build_cursor_and_obstacle_labels(chart, max_step_norm: float):
    """Returns (y_cursor [T,2], cursor_mask [T], obstacle_feats [T,
    MAX_OBSTACLES*OBSTACLE_FEATURE_DIM], y_trail [T]) for one chart.

    y_cursor: obstacle-routed pursuer position (see module docstring),
    identical to plain target_xy() wherever the chart has no collidables.
    obstacle_feats: nearby_obstacle_features() relative to the pursuer's
    position AT THE START of each step (i.e. "what's around me right now,
    before I move") — same relationship inference will use (computed from
    SmoothedCursor's current position each step, see train_dl_multi.py's
    evaluate_chart). y_trail: 1 only when THIS STEP's actual movement segment
    is geometrically safe to trail — doesn't cross any collidable that isn't
    currently due (see the inline comment below for why this replaced two
    earlier, cruder attempts: "trail whenever heading toward any mouse
    note" and "trail only on chart-level needs_routing charts")."""
    T = chart.num_steps
    y_cursor = torch.full((T, 2), 0.5)
    cursor_mask = torch.zeros(T, dtype=torch.bool)
    y_trail = torch.zeros(T)
    obstacle_feats = torch.zeros(T, MAX_OBSTACLES * OBSTACLE_FEATURE_DIM)

    centers, halves = build_collidable_arrays(chart.collidables)
    grid = Grid(chart)
    needs_routing = _needs_routing(chart, grid)

    pos = (0.5, 0.5)
    path_cells = None
    path_idx = 0
    last_target = None

    for step in range(T):
        obstacle_feats[step] = nearby_obstacle_features(pos, centers, halves).reshape(-1)

        target, is_mouse = _lookahead_target(chart, step, extend=needs_routing)
        if target is None:
            continue
        cursor_mask[step] = True
        prev_pos = pos

        if last_target is None:
            # Very first target of the chart — nothing to route FROM yet
            # (the (0.5, 0.5) starting default isn't necessarily even an
            # open, reachable point in a maze-style chart's geometry — see
            # TRAIN_DIARY.md 2026-09-24 "trail path label"). Snap straight
            # there instead of pathing from an arbitrary cold-start point;
            # every subsequent leg routes from wherever the previous target
            # actually left the pursuer, which is always a real position.
            pos = target
            last_target = target
        elif not needs_routing:
            pos = target
        else:
            if target != last_target:
                path_cells = shortest_path_cells(grid.occupied, grid.cell(pos), grid.cell(target))
                path_idx = 0
                last_target = target

            if path_cells is None:
                waypoint = target
            else:
                while path_idx < len(path_cells) - 1:
                    wp = grid.cell_to_norm_xy(path_cells[path_idx])
                    if (wp[0] - pos[0]) ** 2 + (wp[1] - pos[1]) ** 2 < max_step_norm ** 2:
                        path_idx += 1
                    else:
                        break
                waypoint = grid.cell_to_norm_xy(path_cells[path_idx])

            dx, dy = waypoint[0] - pos[0], waypoint[1] - pos[1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist > max_step_norm:
                scale = max_step_norm / dist
                dx *= scale
                dy *= scale
            pos = (pos[0] + dx, pos[1] + dy)

        y_cursor[step] = torch.tensor(pos)

        # Trail ONLY when THIS STEP's move is actually safe to trail — a
        # first attempt set this to "1 whenever heading toward any mouse
        # note" (trail/attack score identically, so more trail seemed
        # free); real in-game test on a normal chart proved that wrong:
        # trail stayed held almost the whole song, and every pass near some
        # OTHER not-yet-due object on the way to the real target picked up
        # a trail-vs-collidable Wrong (see reward.py's
        # _resolve_trail_collisions) a discrete attack click would never
        # have risked — 274 Wrong, 65.7% in-game accuracy. A second attempt
        # gated trail on the whole CHART needing routing (_needs_routing) —
        # better, but that's still a rule *I* picked per-song, not the
        # model deciding per-situation. This is the actual per-situation
        # rule: safe to trail on this exact segment (no not-due collidable
        # in the way) — the model (obstacle_feats are already its input)
        # can in principle learn this same judgment call itself; the label
        # just needs to reward the outcome that's actually safe. See
        # TRAIN_DIARY.md 2026-09-24.
        if is_mouse and _segment_safe_to_trail(chart, step, prev_pos, pos):
            y_trail[step] = 1.0

    return y_cursor, cursor_mask, obstacle_feats, y_trail
