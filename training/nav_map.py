"""Whole-level map input and a learned value-propagation module, so the
policy plans a stroke's route itself (TRAIN_DIARY.md 2026-09-28 "map").

The map is what a player sees on screen plus where the next note will be
(the lookahead a practised player has), MAP_SIZE x MAP_SIZE over the
chart's square normalized canvas:

  0 blocked — objects the cursor is NOT inside: while a stroke is held,
              entering one is a fresh entry (a Wrong unless it's due);
  1 inside  — objects the cursor is inside (the stroke keeps them; a
              carried cover to stay in);
  2 goal    — the next unjudged note's object, where it will be on its beat.

ValueIteration is a Value Iteration Network (Tamar et al. 2016): one learned
3x3 convolution applied over and over, each pass taking a max over its
channels, so value spreads from the goal across the map one cell per pass
— the structure of planning, with every weight learned. It is supervised
per cell with bfs_distance() (a plain geometric flood fill from the goal on
the same map, computed on the fly on the GPU) as a TRAINING target only;
at play time the policy reads its own learned value map, so a new maze is
solved by what it learned, and can fail.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from augment import MODES, _transform_point
from trail_plan import LiveGeometry

# 256: at 128 the real 迷宮 (12.3 world units a cell) lost its narrowest
# corridors to the wall padding and the flood fill never reached the start;
# at 256 (6.2 units a cell) its route is 1101 cells (TRAIN_DIARY.md
# 2026-09-30 "map resolution").
MAP_SIZE = 256
MAP_CHANNELS = 3
PACKED_MAP_BYTES = MAP_CHANNELS * MAP_SIZE * MAP_SIZE // 8
# Value readout around the cursor: a PATCH x PATCH window of the value map.
PATCH = 7
# Learned value = VALUE_TOP - distance in cells at a reachable cell.
VALUE_TOP = 8.0 * MAP_SIZE
VALUE_FLOOR = -1.0


def _mode_permutations(size: int) -> dict[str, np.ndarray]:
    """mode -> flat source index for each flat output cell, so that
    out.flat[i] = src.flat[perm[i]] applies the same D4 map to the image as
    augment._transform_point does to positions."""
    c = (np.arange(size) + 0.5) / size
    gx, gy = np.meshgrid(c, c, indexing="ij")  # [x, y] indexing
    src = torch.tensor(np.stack([gx.ravel(), gy.ravel()], axis=1), dtype=torch.float64)
    perms = {}
    for mode in MODES:
        dst = _transform_point(src, mode).numpy()
        di = np.clip((dst[:, 0] * size).astype(int), 0, size - 1)
        dj = np.clip((dst[:, 1] * size).astype(int), 0, size - 1)
        perm = np.empty(size * size, dtype=np.int64)
        perm[di * size + dj] = np.arange(size * size)
        perms[mode] = perm
    return perms


_PERMS = _mode_permutations(MAP_SIZE)


class MapRenderer:
    """Per chart. Object footprints are rasterized once per pose and cached,
    so a tick costs an OR over a few hundred cached masks."""

    def __init__(self, chart):
        self.chart = chart
        c = (np.arange(MAP_SIZE) + 0.5) / MAP_SIZE
        gx, gy = np.meshgrid(c, c, indexing="ij")
        wx, wy = chart.world_xy(gx.ravel(), gy.ravel())
        self.wx, self.wy = np.asarray(wx), np.asarray(wy)
        self.cell_world = chart.world_span / MAP_SIZE
        self._masks: dict = {}
        self._goal_masks: dict[int, np.ndarray] = {}
        self._moving: set[str] | None = None

    def moving_ids(self) -> set[str]:
        """Objects whose pose ever changes (carried blocks, track handles),
        from a scan every 250ms. The map leaves them out: it is for the
        route through what stays put, so it keeps one value map per stroke
        instead of a new one every tick (the local view, whiskers and
        per-object features show what moves)."""
        if self._moving is None:
            first: dict[str, tuple] = {}
            moving: set[str] = set()
            t_end = float(self.chart.t_ms[-1])
            for t in np.arange(0.0, t_end + 250.0, 250.0):
                g = LiveGeometry(self.chart, float(t))
                for k, oid in enumerate(g.ids):
                    pose = (round(g.cx[k], 1), round(g.cy[k], 1), round(g.hw[k], 1), round(g.hh[k], 1),
                            round(float(g.cos[k]), 3))
                    if first.setdefault(oid, pose) != pose:
                        moving.add(oid)
            self._moving = moving
        return self._moving

    def _mask(self, geom: LiveGeometry, k: int) -> np.ndarray:
        key = (geom.ids[k], round(geom.cx[k], 1), round(geom.cy[k], 1), round(geom.hw[k], 1),
               round(geom.hh[k], 1), round(float(np.degrees(np.arctan2(geom.sin[k], geom.cos[k]))), 1))
        mask = self._masks.get(key)
        if mask is None:
            if len(self._masks) > 2000:  # full-grid masks: keep memory bounded
                self._masks.clear()
            # Any cell the object touches (half a cell of slack), so a thin
            # wall is never lost between cell centers.
            sub = geom.subset(np.array([k]))
            mask = sub.inside(self.wx, self.wy, pad=self.cell_world * 0.5)[:, 0]
            self._masks[key] = mask
        return mask

    def goal_mask(self, uid: int) -> np.ndarray:
        mask = self._goal_masks.get(uid)
        if mask is None:
            ev = self.chart.events[uid]
            geom = LiveGeometry(self.chart, float(ev["time"]))
            k = geom.index.get(ev["id"])
            if k is None:
                nx, ny = self.chart.normalized_xy(ev)
                i = min(int(nx * MAP_SIZE), MAP_SIZE - 1)
                j = min(int(ny * MAP_SIZE), MAP_SIZE - 1)
                mask = np.zeros(MAP_SIZE * MAP_SIZE, dtype=bool)
                mask[max(i, 0) * MAP_SIZE + max(j, 0)] = True
            else:
                mask = self._mask(geom, k)
            self._goal_masks[uid] = mask
        return mask

    def render(self, t_ms: float, inside_ids: set[str], goal_uid: int | None, mode: str) -> np.ndarray:
        """-> packed bits (PACKED_MAP_BYTES,) of the [3, S, S] map in the
        policy's (augmented) frame."""
        geom = LiveGeometry(self.chart, t_ms)
        blocked = np.zeros(MAP_SIZE * MAP_SIZE, dtype=bool)
        inside = np.zeros(MAP_SIZE * MAP_SIZE, dtype=bool)
        moving = self.moving_ids()
        for k, oid in enumerate(geom.ids):
            if oid in moving:
                continue
            if oid in inside_ids:
                inside |= self._mask(geom, k)
            else:
                blocked |= self._mask(geom, k)
        goal = self.goal_mask(goal_uid) if goal_uid is not None else np.zeros_like(blocked)
        stack = np.stack([blocked, inside, goal])
        if mode != "identity":
            stack = stack[:, _PERMS[mode]]
        return np.packbits(stack.ravel())


def unpack_maps(packed: torch.Tensor) -> torch.Tensor:
    """[B, PACKED_MAP_BYTES] uint8 -> [B, 3, S, S] float (on packed's device)."""
    bits = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    unpacked = (packed.unsqueeze(-1) >> bits) & 1
    return unpacked.reshape(packed.shape[0], MAP_CHANNELS, MAP_SIZE, MAP_SIZE).float()


@torch.no_grad()
def bfs_distance(maps: torch.Tensor, max_iters: int = 8 * MAP_SIZE) -> torch.Tensor:
    """Training target: 8-connected flood fill from the goal cells through
    cells not blocked, in cells; inf where unreachable. [B, S, S]."""
    blocked, goal = maps[:, 0] > 0.5, maps[:, 2] > 0.5
    passable = ~blocked | goal
    inf = float("inf")
    dist = torch.where(goal, torch.zeros_like(maps[:, 0]), torch.full_like(maps[:, 0], inf))
    for _ in range(max_iters):
        neighbor = -F.max_pool2d(-dist.unsqueeze(1), 3, stride=1, padding=1).squeeze(1)
        new = torch.where(passable, torch.minimum(dist, neighbor + 1.0), dist)
        if torch.equal(new, dist):
            break
        dist = new
    return dist


class ValueIteration(nn.Module):
    """VIN: v <- max_a (W_in * x + W_v * v)_a, K times. x = map + coord
    channels. Only the last `grad_iters` passes carry gradients (truncated
    backprop through the recurrence), so K can cover a long route on a
    4GB GPU."""

    def __init__(self, actions: int = 10):  # 8 neighbors + goal + stay
        super().__init__()
        self.inp = nn.Conv2d(MAP_CHANNELS + 2, actions, 3, padding=1)
        # Per action, a 3x3 transition distribution (softmax over the 9
        # taps), as in the VIN paper's transition model. Unconstrained
        # kernels over 1280 passes blew up geometrically (vin1's loss went
        # 55 -> 416 in 100 steps); a convex mix of neighbor values can't.
        # Initialized as moves (8 neighbor shifts, then "stay"), still all
        # learnable: from a random init the kernels stayed near-uniform
        # averages (max tap weight 0.13-0.29), which can't carry a value
        # 1100 cells without drifting, and vin1 saturated every cell at
        # VALUE_TOP (TRAIN_DIARY.md 2026-09-30).
        init = torch.zeros(actions, 1, 3, 3)
        taps = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]
        for a in range(actions):
            i, j = taps[a] if a < len(taps) else (1, 1)
            init[a, 0, i, j] = 10.0
        self.rec_logits = nn.Parameter(init)

    def _step(self, q_in: torch.Tensor, v: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
        x = (q_in + F.conv2d(v, kernel, padding=1)).amax(dim=1, keepdim=True).clamp(max=VALUE_TOP)
        # Leaky floor: a hard clamp has no gradient below it, so vin4's goal
        # cells (input ~-2100) sat at the floor for good and the training
        # that should teach the goal's value could never raise them.
        return torch.where(x < VALUE_FLOOR, VALUE_FLOOR + 1e-3 * (x - VALUE_FLOOR), x)

    def q_in(self, maps: torch.Tensor) -> torch.Tensor:
        """Per-action input from the map. The two coordinate channels are fed
        zeros: with real x/y the input learned position-specific values (a
        third of free cells got some action > 0, many on the border), which
        became false goals on the 8-16 cell mazes. Moving is the same rule
        everywhere. x64: the goal must reach VALUE_TOP (2048) while a step
        costs ~1, from 0/1 map inputs."""
        b, _, s, _ = maps.shape
        zeros = maps.new_zeros(b, 2, s, s)
        return 64.0 * self.inp(torch.cat([maps, zeros], dim=1))

    def kernel(self) -> torch.Tensor:
        return torch.softmax(self.rec_logits.view(self.rec_logits.shape[0], -1), dim=-1).view_as(self.rec_logits)

    def forward(self, maps: torch.Tensor, iters: int, grad_iters: int = 16) -> torch.Tensor:
        b, _, s, _ = maps.shape
        q_in = self.q_in(maps)
        kernel = self.kernel()
        v = torch.full((b, 1, s, s), VALUE_FLOOR, device=maps.device)
        free = max(0, iters - grad_iters)
        with torch.no_grad():
            for _ in range(free):
                v = self._step(q_in, v, kernel)
        for _ in range(iters - free):
            v = self._step(q_in, v, kernel)
        return v.squeeze(1)


def value_target(dist: torch.Tensor) -> torch.Tensor:
    """The value a VIN can represent exactly for the flood fill: VALUE_TOP at
    the goal, one less per cell, VALUE_FLOOR where unreachable."""
    return torch.where(torch.isfinite(dist), VALUE_TOP - dist, torch.full_like(dist, VALUE_FLOOR))


def cursor_patch(value: torch.Tensor, cursor_xy: torch.Tensor) -> torch.Tensor:
    """[B, S, S] value map, [B, 2] cursor in 0..1 (policy frame) -> [B,
    PATCH*PATCH + 1]: the window around the cursor relative to its own cell
    (scaled), plus the cell's own value as cells-to-go (scaled)."""
    b, s, _ = value.shape
    r = PATCH // 2
    padded = F.pad(value.unsqueeze(1), (r, r, r, r), value=VALUE_FLOOR).squeeze(1)
    ci = (cursor_xy[:, 0] * s).long().clamp(0, s - 1)
    cj = (cursor_xy[:, 1] * s).long().clamp(0, s - 1)
    offs = torch.arange(PATCH, device=value.device)
    rows = (ci.view(b, 1) + offs.view(1, PATCH)).view(b, PATCH, 1).expand(b, PATCH, PATCH)
    cols = (cj.view(b, 1) + offs.view(1, PATCH)).view(b, 1, PATCH).expand(b, PATCH, PATCH)
    patch = padded[torch.arange(b, device=value.device).view(b, 1, 1), rows, cols]
    center = value[torch.arange(b, device=value.device), ci, cj]
    rel = (patch - center.view(b, 1, 1)).clamp(-8.0, 8.0) / 4.0
    togo = ((VALUE_TOP - center) / MAP_SIZE).clamp(0.0, 8.0).view(b, 1) / 2.0
    return torch.cat([rel.reshape(b, -1), togo], dim=1)


class MapPlanner:
    """Runs the policy's ValueIteration: map features for acting (cached per
    distinct map, since a maze's map stays the same for seconds) and the
    per-cell distance loss that trains it."""

    def __init__(self, vin: ValueIteration, device: str, iters: int = 5 * MAP_SIZE):
        self.vin = vin
        self.device = device
        self.iters = iters
        self._cache: dict[bytes, torch.Tensor] = {}

    @torch.no_grad()
    def value_maps(self, packed: np.ndarray) -> torch.Tensor:
        """[B, PACKED_MAP_BYTES] uint8 -> [B, S, S] learned value maps (no grad),
        computed once per distinct map in the batch."""
        keys = [row.tobytes() for row in packed]
        missing = sorted({k for k in keys if k not in self._cache})
        if missing:
            maps = unpack_maps(torch.from_numpy(np.frombuffer(b"".join(missing), dtype=np.uint8)
                                                .reshape(len(missing), -1)).to(self.device))
            values = []
            for i in range(0, len(missing), 64):
                values.append(self.vin(maps[i : i + 64], self.iters, grad_iters=0))
            values = torch.cat(values)
            if len(self._cache) + len(missing) > 2048:  # 256KB each on the GPU at 256x256
                self._cache.clear()
            for k, v in zip(missing, values):
                self._cache[k] = v
        return torch.stack([self._cache[k] for k in keys])

    def clear(self):
        """Call after the VIN's weights change."""
        self._cache.clear()

    @torch.no_grad()
    def features(self, packed: np.ndarray, cursor_xy: np.ndarray) -> torch.Tensor:
        """[B, bytes], [B, 2] (policy frame) -> [B, PATCH*PATCH+1] on device."""
        value = self.value_maps(packed)
        return cursor_patch(value, torch.as_tensor(cursor_xy, dtype=torch.float32, device=self.device))

    @torch.no_grad()
    def features_or_zero(self, maps: list, cursor_xy: np.ndarray) -> torch.Tensor:
        """maps: per sample a packed map or None (no stroke held) -> [B,
        PATCH*PATCH+1], zeros where there is no map."""
        out = torch.zeros(len(maps), PATCH * PATCH + 1, device=self.device)
        have = [i for i, m in enumerate(maps) if m is not None]
        if have:
            out[have] = self.features(np.stack([maps[i] for i in have]), np.asarray(cursor_xy)[have])
        return out

    def step_loss(self, packed: np.ndarray, unroll: int = 16) -> tuple[torch.Tensor, float]:
        """One learned propagation pass, supervised directly: from the flood
        fill after k passes (cells within k of the goal hold their value,
        the rest VALUE_FLOOR) it must produce the flood fill after k + 1,
        for a random k per map. Backprop through only the last 16 of 1280
        passes (distance_loss) stalled at ~200 cells of error; this teaches
        the rule the whole recurrence repeats (Deep Thinking's
        incremental-progress idea). Returns (loss, mean abs error in cells
        on the cells that change)."""
        maps = unpack_maps(torch.from_numpy(packed).to(self.device))
        dist = bfs_distance(maps)
        b = maps.shape[0]
        finite = torch.where(torch.isfinite(dist), dist, torch.zeros_like(dist))
        top = finite.flatten(1).amax(1)
        # k from -unroll: some states start before the goal is seeded (the
        # rollout starts all at VALUE_FLOOR), so the step learns to create
        # the goal's value too — vin3's goal input was ~-17, the goal never
        # rose, and stray sources took over.
        k = (torch.rand(b, device=self.device) * (top + 2 + unroll)).floor().view(b, 1, 1) - unroll

        def upto(n):
            return torch.where(dist <= n, VALUE_TOP - dist, torch.full_like(dist, VALUE_FLOOR))

        # `unroll` passes, not one: a single-pass loss left a small upward
        # drift per pass that 1280 passes turned into every cell at
        # VALUE_TOP (test error stuck at the mean route length, ~180 cells).
        v_k, v_next = upto(k), upto(k + unroll)
        q_in = self.vin.q_in(maps)
        kernel = self.vin.kernel()
        pred = v_k.unsqueeze(1)
        for _ in range(unroll):
            pred = self.vin._step(q_in, pred, kernel)
        pred = pred.squeeze(1)
        # In cells, Huber: squared error / MAP_SIZE^2 made a cell drifting
        # 4 a pass cost ~2e-4, and those drifting cells became false goals
        # over a full rollout on the big mazes.
        err = F.smooth_l1_loss(pred, v_next, reduction="none", beta=1.0)
        changed = v_k != v_next
        # The frontier is a few hundred of 65k cells: weigh it on its own or
        # "leave everything as it was" already scores well.
        loss = err.mean() + (err[changed].mean() if changed.any() else 0.0)
        # Walls must stay at the floor however high their neighbors get:
        # without this term they rose to ~1200 over a full rollout and the
        # value leaked through them (test error stuck at ~160 cells).
        blocked = (maps[:, 0] > 0.5) & (maps[:, 2] < 0.5)
        if blocked.any():
            loss = loss + err[blocked].mean()
        mae = float((pred - v_next).abs()[changed].mean()) if changed.any() else 0.0
        return loss, mae

    def distance_loss(self, packed: np.ndarray, grad_iters: int = 16) -> tuple[torch.Tensor, float]:
        """VIN value vs the flood-fill value at every cell (scaled to cells /
        MAP_SIZE). Returns (loss, mean abs error in cells on reachable cells)."""
        maps = unpack_maps(torch.from_numpy(packed).to(self.device))
        target = value_target(bfs_distance(maps))
        value = self.vin(maps, self.iters, grad_iters=grad_iters)
        err = (value - target) / MAP_SIZE
        loss = (err ** 2).mean()
        reach = target > VALUE_FLOOR
        mae = float((value - target).abs()[reach].mean()) if reach.any() else 0.0
        return loss, mae


# Local view: the screen right around the cursor at a fixed WORLD scale
# (LOCAL_VIEW x LOCAL_VIEW cells of LOCAL_VIEW_CELL_WORLD units, +-64), with
# the game's interaction semantics per cell rather than any chosen point
# (TRAIN_DIARY.md 2026-09-28 "interaction semantics"):
#
#   0 block       a block or track handle covers the cell
#   1 group rect  a group rect covers the cell
#   2 tap         how many objects a click here would score, by the game's
#                 first-point rule (on a block: the block(s), not the rects
#                 under it; on rect space: the rect(s)) — 0..3+
#   3 sweep       how many objects a held stroke moving INTO this cell
#                 would newly trigger (those not already intersected;
#                 not holding = all of them) — 0..3+
#   4 next        the next unjudged note's object, where it is now
#   5 next tap    a click here would score that object (same first-point
#                 rule: on a block inside a group rect, only the block
#                 fires, so the rect's note isn't hit there)
#
# A clean click on the next note is where `next tap` is set and `tap` is 1;
# nothing marks a point — the policy finds it. The counts use two
# bit-planes each, so a view packs to 8 bit-planes.
LOCAL_VIEW = 32
LOCAL_VIEW_CELL_WORLD = 4.0
LOCAL_VIEW_CHANNELS = 6
_VIEW_PLANES = 8
PACKED_VIEW_BYTES = _VIEW_PLANES * LOCAL_VIEW * LOCAL_VIEW // 8
_VIEW_OFFSETS = None


def _view_offsets() -> np.ndarray:
    global _VIEW_OFFSETS
    if _VIEW_OFFSETS is None:
        c = (np.arange(LOCAL_VIEW) + 0.5 - LOCAL_VIEW / 2) * LOCAL_VIEW_CELL_WORLD
        gx, gy = np.meshgrid(c, c, indexing="ij")
        _VIEW_OFFSETS = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return _VIEW_OFFSETS


def interaction_counts(geom: LiveGeometry, inside: np.ndarray, intersected: set[str] | None
                       ) -> tuple[np.ndarray, np.ndarray]:
    """inside: [P, N] point-in-object. -> (tap, sweep) object counts per
    point. tap follows Judge._scored_on_first_point (a first point on a
    block drops the group rects); sweep counts objects not in `intersected`
    (None = not holding: every touched object is a fresh entry)."""
    on_block = inside[:, geom.is_block].any(axis=1)
    tap = np.where(on_block, inside[:, ~geom.is_rect].sum(axis=1), inside.sum(axis=1))
    if intersected:
        fresh = np.array([oid not in intersected for oid in geom.ids], dtype=bool)
        sweep = inside[:, fresh].sum(axis=1)
    else:
        sweep = inside.sum(axis=1)
    return tap, sweep


def render_local_view(chart, t_ms: float, cursor_norm, mode: str, intersected: set[str] | None,
                      next_id: str | None) -> np.ndarray:
    """-> packed [_VIEW_PLANES * V * V] bits in the policy's frame: cell
    (i, j) is the policy-frame offset (i, j) from the cursor, mapped back to
    the canvas through the inverse D4 transform."""
    from augment import INVERSE_MODES, transform_vector

    offsets = _view_offsets()
    if mode != "identity":
        offsets = transform_vector(torch.from_numpy(offsets), INVERSE_MODES[mode]).numpy()
    cx, cy = chart.world_xy(*cursor_norm)
    geom = LiveGeometry(chart, t_ms)
    near = geom.near(cx, cy, LOCAL_VIEW * LOCAL_VIEW_CELL_WORLD)
    planes = np.zeros((_VIEW_PLANES, LOCAL_VIEW * LOCAL_VIEW), dtype=bool)
    if len(near):
        sub = geom.subset(near)
        # The cell center only (no slack): the counts must match what a
        # click at that exact spot would do.
        inside = sub.inside(cx + offsets[:, 0], cy + offsets[:, 1])
        planes[0] = inside[:, ~sub.is_rect].any(axis=1)
        planes[1] = inside[:, sub.is_rect].any(axis=1)
        tap, sweep = interaction_counts(sub, inside, intersected)
        tap, sweep = np.minimum(tap, 3), np.minimum(sweep, 3)
        planes[2], planes[3] = tap & 1, tap >> 1
        planes[4], planes[5] = sweep & 1, sweep >> 1
        if next_id is not None and next_id in sub.index:
            k = sub.index[next_id]
            planes[6] = inside[:, k]
            on_block = inside[:, sub.is_block].any(axis=1)
            planes[7] = inside[:, k] & (~on_block | ~sub.is_rect[k])
    return np.packbits(planes.ravel())


# Whiskers: exact distances along WHISKER_DIRS rays from the cursor (the
# policy's frame), in world units — what the 4-unit view cells and the
# center-ranked obstacle list blur. bc12 started every planned stroke but
# its Wrongs were ~all sweeps into walls, its move off the teacher's by
# 12-15 units/tick just before (TRAIN_DIARY.md 2026-09-29 "whiskers").
# Per ray, three game facts, no chosen route or point:
#   enter  how far a held stroke goes before it newly triggers an object
#          (the view's sweep layer: already-intersected objects don't
#          count; not holding = every object, 0 if the cursor is on one)
#   exit   how far before it leaves an intersected object it is inside
#          (leaving a cover and re-entering it is a fresh Wrong)
#   next   how far to the next unjudged note's object
# Each as log1p(d / 4) / log1p(WHISKER_RANGE / 4), capped at 1 (none in range).
WHISKER_DIRS = 16
WHISKER_RANGE = 128.0
WHISKER_DIM = 3 * WHISKER_DIRS
_WHISKER_UNIT = None


def _whisker_units() -> np.ndarray:
    global _WHISKER_UNIT
    if _WHISKER_UNIT is None:
        a = np.arange(WHISKER_DIRS) * (2 * np.pi / WHISKER_DIRS)
        _WHISKER_UNIT = np.stack([np.cos(a), np.sin(a)], axis=1)
    return _WHISKER_UNIT


def _ray_spans(geom: LiveGeometry, x: float, y: float, dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Slab test of rays from (x, y) along dirs [R, 2] against every OBB.
    -> (t_in, t_out) [R, N]; no hit ahead = (inf, -inf). t_in is 0 when the
    point is already inside."""
    dx, dy = x - geom.cx, y - geom.cy
    ox = dx * geom.cos + dy * geom.sin
    oy = -dx * geom.sin + dy * geom.cos
    ux = dirs[:, 0:1] * geom.cos + dirs[:, 1:2] * geom.sin
    uy = -dirs[:, 0:1] * geom.sin + dirs[:, 1:2] * geom.cos
    t_in = np.zeros((len(dirs), len(geom.cx)))
    t_out = np.full((len(dirs), len(geom.cx)), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for o, u, h in ((ox, ux, geom.hw), (oy, uy, geom.hh)):
            a = (-h - o) / u
            b = (h - o) / u
            lo, hi = np.minimum(a, b), np.maximum(a, b)
            # Parallel to this slab: inside it for all t, or never.
            par = u == 0
            inside_slab = np.abs(o) <= h
            lo = np.where(par, np.where(inside_slab, -np.inf, np.inf), lo)
            hi = np.where(par, np.where(inside_slab, np.inf, -np.inf), hi)
            t_in = np.maximum(t_in, lo)
            t_out = np.minimum(t_out, hi)
    miss = t_out < t_in
    return np.where(miss, np.inf, t_in), np.where(miss, -np.inf, t_out)


def whisker_features(chart, t_ms: float, cursor_norm, mode: str, intersected: set[str] | None,
                     next_id: str | None) -> np.ndarray:
    """-> [WHISKER_DIM] float32: enter[16], exit[16], next[16] (see above);
    ray i points along policy-frame angle 2*pi*i/16."""
    from augment import INVERSE_MODES, transform_vector

    dirs = _whisker_units()
    if mode != "identity":
        dirs = transform_vector(torch.from_numpy(dirs), INVERSE_MODES[mode]).numpy()
    cx, cy = chart.world_xy(*cursor_norm)
    geom = LiveGeometry(chart, t_ms)
    near = geom.near(cx, cy, WHISKER_RANGE)
    enter = np.full(WHISKER_DIRS, np.inf)
    exit_ = np.full(WHISKER_DIRS, np.inf)
    nxt = np.full(WHISKER_DIRS, np.inf)
    if len(near):
        sub = geom.subset(near)
        t_in, t_out = _ray_spans(sub, cx, cy, dirs)
        held = np.array([intersected is not None and oid in intersected for oid in sub.ids], dtype=bool)
        if (~held).any():
            enter = t_in[:, ~held].min(axis=1)
        inside_now = sub.inside(np.array([cx]), np.array([cy]))[0] & held
        if inside_now.any():
            exit_ = np.clip(t_out[:, inside_now], 0, None).min(axis=1)
        if next_id is not None and next_id in sub.index:
            nxt = t_in[:, sub.index[next_id]]
    scale = np.log1p(WHISKER_RANGE / 4.0)
    out = [np.log1p(np.minimum(d, WHISKER_RANGE) / 4.0) / scale for d in (enter, exit_, nxt)]
    return np.concatenate(out).astype(np.float32)


# Carrier: the object the held stroke is riding — the smallest
# intersected object the cursor is inside (a carried block; in a maze, the
# still cover rects). Once its note is hit it leaves the per-object list, so
# without this the policy couldn't tell how the thing under the cursor
# moves, and bc14 drifted off carried blocks and re-entered them (a fresh
# entry = Wrong) — TRAIN_DIARY.md 2026-09-29 "carrier". Game state, in the
# policy frame: [riding, vx, vy (world units per tick / 8, clipped to +-2),
# cursor offset from its center / its half-extent r (so +-1 at the edge;
# offset / 64 made a maze cover's hundreds of units an input of 5-10 and
# wrecked bc16), log1p(r / 16) / 4].
CARRIER_DIM = 6


def carrier_features(chart, t_ms: float, dt_ms: float, cursor_norm, mode: str,
                     intersected: set[str] | None) -> np.ndarray:
    from augment import transform_vector

    out = np.zeros(CARRIER_DIM, dtype=np.float32)
    if not intersected:
        return out
    cx, cy = chart.world_xy(*cursor_norm)
    geom = LiveGeometry(chart, t_ms)
    inside = geom.inside(np.array([cx]), np.array([cy]))[0]
    ks = [k for k in np.flatnonzero(inside) if geom.ids[k] in intersected]
    if not ks:
        return out
    k = min(ks, key=lambda k: geom.hw[k] * geom.hh[k])
    prev = LiveGeometry(chart, t_ms - dt_ms)
    j = prev.index.get(geom.ids[k])
    v = np.zeros(2) if j is None else np.array([geom.cx[k] - prev.cx[j], geom.cy[k] - prev.cy[j]])
    off = np.array([cx - geom.cx[k], cy - geom.cy[k]])
    if mode != "identity":
        v = transform_vector(torch.from_numpy(v).float(), mode).numpy()
        off = transform_vector(torch.from_numpy(off).float(), mode).numpy()
    r = float(max(geom.hw[k], geom.hh[k], 1e-6))
    v = np.clip(v / 8.0, -2.0, 2.0)
    off = np.clip(off / r, -1.0, 1.0)
    out[:] = [1.0, v[0], v[1], off[0], off[1], np.log1p(r / 16.0) / 4.0]
    return out


def unpack_views(packed: torch.Tensor) -> torch.Tensor:
    """[B, PACKED_VIEW_BYTES] uint8 -> [B, 6, V, V] float: block, rect,
    tap/3, sweep/3, next, next tap."""
    bits = torch.arange(7, -1, -1, device=packed.device, dtype=torch.uint8)
    p = ((packed.unsqueeze(-1) >> bits) & 1).reshape(packed.shape[0], _VIEW_PLANES, LOCAL_VIEW, LOCAL_VIEW).float()
    tap = (p[:, 2] + 2 * p[:, 3]) / 3.0
    sweep = (p[:, 4] + 2 * p[:, 5]) / 3.0
    return torch.stack([p[:, 0], p[:, 1], tap, sweep, p[:, 6], p[:, 7]], dim=1)
