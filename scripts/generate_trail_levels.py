"""Procedural .yblevel generator for teaching trail strokes.

Only two real charts need a held stroke (迷宮🗣️🔥, 只因為你那渴望自由的心臟🫀),
one stroke start each per episode — too few examples for behavior cloning
to learn when to start a stroke instead of clicking (TRAIN_DIARY.md
2026-09-27 "trail"). This writes many short levels built on the same two
mechanics, in the real .yblevel format so they go through
scripts/encodeFrames.js like any chart and can be opened in ybnote-web:

- maze: a random perfect maze of silent 30-unit group-rect walls, under one
  or more big silent group rects. Clicking the goal would also score the
  big rects (Wrongs); the way through is a stroke started on the start
  block (startedOnBlock marks the big rects intersected without firing)
  and held down the corridors into the goal on its beat.
- carrier: a note block that an autoplay track scales up and carries onto
  a series of drum blocks, arriving on each drum's beat. The carrier covers
  the drum, so a click also scores the carrier (Wrong); a stroke started on
  the carrier's own note rides inside it and steps into each drum on time.
  The carrier then returns home for one more note, which needs the stroke
  released first.

- rects (--kind rects): group rects with blocks inside, notes mostly on the
  rects and sometimes on a block inside one. A click on a block inside a
  rect scores only the block (startedOnBlock), so a rect's note needs a
  click on the rect's free space — often a narrow gap. No stroke; these
  teach reading the local view's interaction semantics (nav_map), since
  only ~150 of the real charts' notes are rects.

The maze and carrier kinds add ordinary clickable notes before and after, so
the stroke start is a choice in context. Usage:

    python scripts/generate_trail_levels.py --out input_synth --count 200 --seed 1
    python scripts/generate_trail_levels.py --out input_synth_test --count 4 --seed 99 --with-audio "input/迷宮🗣️🔥.yblevel"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import string
import zipfile

BLOCK = 60.0
WALL = 30.0
PITCHES = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5", "D5", "E5", "F5", "G5"]


def uid(rng: random.Random, prefix: str) -> str:
    return prefix + "-" + "".join(rng.choice(string.ascii_lowercase + string.digits) for _ in range(8))


class Level:
    def __init__(self, rng: random.Random, title: str):
        self.rng = rng
        self.title = title
        self.blocks: list[dict] = []
        self.rects: list[dict] = []
        self.tracks: list[dict] = []
        self.notes: list[tuple[float, str, str, str, str]] = []  # (ms, target id, type, pitch, instrument)
        self.note_track = uid(rng, "noteTrack")
        self.lane_track = uid(rng, "noteTrack")

    def block(self, cx: float, cy: float, pitch: str | None = None, instrument: str = "piano",
              volume: float = 1.0, block_id: str | None = None, carried_by: str | None = None) -> dict:
        b = {
            "id": block_id or uid(self.rng, "noteblock"),
            "x": cx - BLOCK / 2, "y": cy - BLOCK / 2,
            "pitch": pitch or self.rng.choice(PITCHES), "instrument": instrument, "volume": volume,
        }
        if carried_by:
            b["carriedByTrackId"] = carried_by
        self.blocks.append(b)
        return b

    def rect(self, x: float, y: float, w: float, h: float, color: str = "red", volume: float = 0.0) -> dict:
        r = {"enabled": True, "x": x, "y": y, "w": w, "h": h, "name": " ", "volume": volume, "bg": color,
             "id": uid(self.rng, "groupRect")}
        self.rects.append(r)
        return r

    def note_on_block(self, t_ms: float, b: dict):
        self.notes.append((t_ms, b["id"], "block", b["pitch"], b["instrument"]))

    def note_on_rect(self, t_ms: float, r: dict):
        self.notes.append((t_ms, r["id"], "groupRect", "lane0", "object_track"))

    def clickable_run(self, t0: float, count: int, area: tuple[float, float, float, float],
                      avoid: list[tuple[float, float, float, float]]) -> float:
        """`count` ordinary click notes on fresh blocks inside `area`
        (x0, y0, x1, y1), clear of every `avoid` box; returns the last time."""
        t = t0
        placed = []
        for _ in range(count):
            for _attempt in range(200):
                cx = self.rng.uniform(area[0] + BLOCK, area[2] - BLOCK)
                cy = self.rng.uniform(area[1] + BLOCK, area[3] - BLOCK)
                box = (cx - BLOCK, cy - BLOCK, cx + BLOCK, cy + BLOCK)
                if all(not _overlap(box, a) for a in avoid + placed):
                    break
            else:
                return t
            placed.append(box)
            b = self.block(cx, cy)
            self.note_on_block(t, b)
            t += self.rng.choice([250.0, 375.0, 500.0])
        return t - 0.0

    def to_level_txt(self, chart_end_s: float) -> str:
        notes = sorted(self.notes)
        events, midi = [], []
        for t_ms, target, kind, pitch, instrument in notes:
            events.append({"time": t_ms, "pitch": pitch, "instrument": instrument, "blockId": target, "blockType": kind})
            lane = kind == "groupRect"
            midi.append({
                "id": uid(self.rng, "note"), "pitch": 12 if lane else 60, "name": pitch, "timeStart": t_ms / 1000,
                "duration": 0.25, "velocity": 1, "targetId": target, "targetType": kind,
                "trackId": self.lane_track if lane else self.note_track, "trackName": "Track",
                "trackInstrument": instrument,
            })
        xs = [b["x"] for b in self.blocks] + [r["x"] for r in self.rects]
        ys = [b["y"] for b in self.blocks] + [r["y"] for r in self.rects]
        payload = {
            "blocks": self.blocks,
            "groupRects": self.rects,
            "tracks": self.tracks,
            "widgets": [],
            "camera": {"zoom": 0.6, "x": (min(xs) + max(xs)) / 2 if xs else 0, "y": (min(ys) + max(ys)) / 2 if ys else 0},
            "lyrics": [],
            "lyricsStyle": {"anchor": "bottom", "offsetY": 80, "fontSize": 32, "color": "#ffffff",
                            "bg": "rgba(0,0,0,0.5)", "bold": True, "align": "center"},
            "markers": [],
            "noteTracks": [
                {"id": self.note_track, "name": "Track 1", "instrument": "piano", "isBackground": False,
                 "muted": False, "ghostVisible": False},
                {"id": self.lane_track, "name": "Track 2", "instrument": "object_track", "isBackground": False,
                 "muted": False, "ghostVisible": False},
            ],
            "events": events,
            "midiNotes": midi,
        }
        header = "\n".join([
            "VERSION:3", "BPM:120", "AUDIO_START_TIME:0", f"CHART_END:{chart_end_s:.3f}", "PREVIEW_START:0",
            "AUDIO_VOLUME:0", "AUDIO_FADE_IN:0", "AUDIO_FADE_OUT:0", "BLOCKS_DRAGGABLE:0", "GROUP_RECTS_DRAGGABLE:0",
            "GROUP_RECTS_RESIZABLE:0", "TRACKS_DRAGGABLE:0", "WIDGETS_DRAGGABLE:0", "TRACK_CARRY_EDITABLE:0",
            "KEY_BINDING_EDITABLE:0", "MATCH_BY_PITCH_INSTRUMENT:0", "GAME_SPEEDS:0.25,0.5,0.75,1,1.25,1.5",
            f"TITLE:{self.title}", "AUTHOR:generate_trail_levels.py",
            "DESCRIPTION:Synthetic trail-stroke training level.", "MUSIC_CREDIT:", "MUSIC_NAME:", "MUSIC_AUTHOR:",
        ])
        return header + "\n\n\n[JSON]\n" + json.dumps(payload, ensure_ascii=False)


def _overlap(a, b) -> bool:
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _maze_cells(rng: random.Random, cols: int, rows: int) -> set[tuple[int, int, int, int]]:
    """Randomized DFS perfect maze -> set of open passages ((c, r), (c2, r2))."""
    seen = {(0, 0)}
    stack = [(0, 0)]
    open_edges = set()
    while stack:
        c, r = stack[-1]
        nbrs = [(c + dc, r + dr) for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1))
                if 0 <= c + dc < cols and 0 <= r + dr < rows and (c + dc, r + dr) not in seen]
        if not nbrs:
            stack.pop()
            continue
        n = rng.choice(nbrs)
        open_edges.add((c, r, n[0], n[1]))
        open_edges.add((n[0], n[1], c, r))
        seen.add(n)
        stack.append(n)
    return open_edges


def make_maze(rng: random.Random, title: str, big: bool = False) -> tuple[Level, float]:
    """big: like the real 迷宮🗣️🔥 (16 x 16 cells of 30-wide corridors, one
    57s stroke at ~120 units/s) — the 3-9 cell mazes run 1-5s at 200-900/s,
    and bc17 sat at the real maze's start for 56s, then cut through the
    walls (TRAIN_DIARY.md 2026-09-30 "big mazes")."""
    lv = Level(rng, title)
    if big:
        cols, rows = rng.randint(8, 16), rng.randint(8, 16)
        corridor = rng.choice([30.0, 30.0, 45.0])
    else:
        cols, rows = rng.randint(3, 9), rng.randint(3, 9)
        corridor = rng.choice([30.0, 30.0, 45.0, 60.0])
    pitch = corridor + WALL
    edges = _maze_cells(rng, cols, rows)
    x0, y0 = 0.0, 0.0
    width, height = cols * pitch + WALL, rows * pitch + WALL

    def cell_center(c, r):
        return x0 + WALL + c * pitch + corridor / 2, y0 + WALL + r * pitch + corridor / 2

    # Entrance on the right edge at the start cell's row, start block just outside it.
    start_cell = (cols - 1, rng.randrange(rows))
    # Goal: the cell farthest from the start by maze distance.
    dist = {start_cell: 0}
    queue = [start_cell]
    for c, r in queue:
        for dc, dr in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (c + dc, r + dr)
            if (c, r, n[0], n[1]) in edges and n not in dist:
                dist[n] = dist[(c, r)] + 1
                queue.append(n)
    goal_cell = max(dist, key=dist.get)
    path_cells = dist[goal_cell] + 1

    # Walls: horizontal runs along each grid line row, vertical along each column line.
    for r in range(rows + 1):
        run = None
        for c in range(cols):
            closed = r in (0, rows) or (c, r - 1, c, r) not in edges
            if closed and run is None:
                run = c
            if (not closed or c == cols - 1) and run is not None:
                end = c if closed else c - 1
                if end >= run:
                    lv.rect(x0 + run * pitch, y0 + r * pitch, (end - run + 1) * pitch + WALL, WALL)
                run = None
    for c in range(cols + 1):
        run = None
        for r in range(rows):
            closed = c in (0, cols) or (c - 1, r, c, r) not in edges
            if c == cols and r == start_cell[1]:
                closed = False  # entrance
            if closed and run is None:
                run = r
            if (not closed or r == rows - 1) and run is not None:
                end = r if closed else r - 1
                if end >= run:
                    lv.rect(x0 + c * pitch, y0 + WALL + run * pitch, WALL, (end - run + 1) * pitch - WALL)
                run = None

    sx, sy = cell_center(*start_cell)
    start_x = x0 + width + BLOCK / 2 + rng.uniform(5, 40)
    margin = 150.0
    for _ in range(rng.randint(1, 3)):
        jitter = rng.uniform(0, 60)
        lv.rect(x0 - margin - jitter, y0 - margin - jitter, width + BLOCK + 2 * margin + 80 + 2 * jitter,
                height + 2 * margin + 2 * jitter, color="#00000000")
    gx, gy = cell_center(*goal_cell)
    goal = lv.rect(gx - corridor / 2, gy - corridor / 2, corridor, corridor, color="#00ff00ff")
    start_block = lv.block(start_x, sy)

    # Clickable notes off to the left of the covered area, before and after.
    cover_box = (x0 - margin - 80, y0 - margin - 80, x0 + width + BLOCK + margin + 160, y0 + height + margin + 80)
    side = (cover_box[0] - 600, y0 - 200, cover_box[0] - 60, y0 + height + 200)
    t = 500.0
    t = lv.clickable_run(t, rng.randint(1, 4), side, [cover_box]) + rng.uniform(600, 1200)
    t_start = t
    lv.note_on_block(t_start, start_block)
    speed = rng.uniform(80, 300) if big else rng.uniform(200, 900)  # world units / s
    route = (path_cells * pitch + (start_x - sx)) * 1.4
    t_goal = t_start + max(800.0, route / speed * 1000.0) + rng.uniform(300, 2500)
    lv.note_on_rect(t_goal, goal)
    t_end = lv.clickable_run(t_goal + rng.uniform(900, 1500), rng.randint(0, 3), side, [cover_box])
    return lv, max(t_end, t_goal) / 1000 + 1.5


def make_carrier(rng: random.Random, title: str) -> tuple[Level, float]:
    lv = Level(rng, title)
    track_id = uid(rng, "track")
    n_drums = rng.randint(3, 7)
    radius = rng.uniform(180, 330)
    home = (rng.uniform(-40, 40), rng.uniform(-40, 40))
    phase = rng.uniform(0, 2 * math.pi)
    drums = []
    for i in range(n_drums):
        a = phase + 2 * math.pi * i / n_drums
        drums.append((home[0] + radius * math.cos(a), home[1] + radius * math.sin(a)))
    if rng.random() < 0.5:
        drums.reverse()
    beat = rng.choice([1000.0, 1500.0, 2000.0])
    scale = rng.choice([2.0, 2.0, 2.5])
    spin = rng.choice([0.0, 0.0, 360.0, 720.0])

    carrier = lv.block(home[0], home[1], pitch="D5", carried_by=track_id)
    drum_blocks = [lv.block(x, y, pitch="kick", instrument="percussion", volume=0) for x, y in drums]

    # A few ordinary notes on other blocks before, away from the drum ring.
    ring = (home[0] - radius - 120, home[1] - radius - 120, home[0] + radius + 120, home[1] + radius + 120)
    side = (ring[2] + 60, ring[1], ring[2] + 600, ring[3])
    t = lv.clickable_run(500.0, rng.randint(1, 4), side, [ring])
    t_start = t + rng.uniform(500, 1000)
    lv.note_on_block(t_start, carrier)

    s = t_start / 1000
    grow = s + rng.uniform(0.3, 0.6)
    pos = [(s - 0.5, home), (grow, home)]
    t_hit = t_start + rng.uniform(1200, 2000)
    for (dx, dy), blk in zip(drums, drum_blocks):
        pos.append((t_hit / 1000, (dx, dy)))
        lv.note_on_block(t_hit, blk)
        t_hit += beat
    back = (t_hit - beat) / 1000 + rng.uniform(1.0, 1.5)
    pos.append((back, home))
    t_home = back * 1000 + rng.uniform(400, 900)
    lv.note_on_block(t_home, carrier)

    def kf(prefix, t_s, value):
        return {"id": uid(rng, prefix), "t": round(t_s, 4), "value": value}

    channels = {
        "position": {"property": "position",
                     "keyframes": [kf("node", t_s, {"x": x, "y": y}) for t_s, (x, y) in pos]},
        "scale": {"property": "scale", "keyframes": [
            kf("marker", s, 1), kf("marker", grow, scale), kf("marker", back - 0.4, scale), kf("marker", back, 1)]},
    }
    if spin:
        channels["rotation"] = {"property": "rotation", "keyframes": [
            kf("marker", grow, 0), kf("marker", back, spin)]}
    lv.tracks.append({
        "enabled": True, "channels": channels, "bpm": 120, "loop": False, "timingMode": "keyframe", "id": track_id,
        "autoplay": True, "controlHandleOffset": {"x": -radius - 150, "y": -radius - 150},
        "controlHandleHidden": True, "carryLocked": True, "pathHidden": True,
    })
    return lv, t_home / 1000 + 1.5


def make_rects(rng: random.Random, title: str) -> tuple[Level, float]:
    lv = Level(rng, title)
    rects = []
    for _ in range(rng.randint(3, 8)):
        for _attempt in range(100):
            w, h = rng.choice([90.0, 120.0, 150.0, 180.0, 240.0]), rng.choice([90.0, 120.0, 150.0, 180.0, 240.0])
            x, y = rng.uniform(-500, 500 - w), rng.uniform(-500, 500 - h)
            box = (x - 40, y - 40, x + w + 40, y + h + 40)
            if all(not _overlap(box, r["box"]) for r in rects):
                break
        else:
            continue
        rect = lv.rect(x, y, w, h, color="#3060ff80", volume=1.0)
        inner = []
        # Blocks inside: anywhere in the rect, not overlapping each other
        # (a click on one would score both), sometimes packed so only a
        # narrow gap of rect space is left.
        placed = []
        for _ in range(rng.randint(1, 4)):
            for _attempt in range(50):
                bx = rng.uniform(x + BLOCK / 2, x + w - BLOCK / 2)
                by = rng.uniform(y + BLOCK / 2, y + h - BLOCK / 2)
                if all(abs(bx - px) >= BLOCK + 2 or abs(by - py) >= BLOCK + 2 for px, py in placed):
                    placed.append((bx, by))
                    inner.append(lv.block(bx, by))
                    break
        rects.append({"rect": rect, "blocks": inner, "box": box})
    if not rects:
        return make_rects(rng, title)
    t = 800.0
    for _ in range(rng.randint(12, 30)):
        r = rng.choice(rects)
        if rng.random() < 0.25:
            lv.note_on_block(t, rng.choice(r["blocks"]))
        else:
            lv.note_on_rect(t, r["rect"])
        t += rng.choice([300.0, 400.0, 500.0, 600.0, 800.0])
    return lv, t / 1000 + 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--count", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kind", choices=["strokes", "rects", "bigmaze"], default="strokes",
                   help="strokes: alternate maze / carrier; rects: group rects with blocks inside; "
                        "bigmaze: 8-16 cell mazes walked slowly")
    p.add_argument("--with-audio", default="", help="copy audio.mp3 from this .yblevel so the level plays in ybnote-web")
    args = p.parse_args()
    rng = random.Random(args.seed)
    audio = None
    if args.with_audio:
        with zipfile.ZipFile(args.with_audio) as z:
            audio = z.read("audio.mp3")
    os.makedirs(args.out, exist_ok=True)
    for i in range(args.count):
        if args.kind in ("rects", "bigmaze"):
            kind = args.kind
        else:
            kind = "maze" if i % 2 == 0 else "carrier"
        title = f"synth_{kind}_{args.seed}_{i:04d}"
        make = {"maze": make_maze, "carrier": make_carrier, "rects": make_rects,
                "bigmaze": lambda r, t: make_maze(r, t, big=True)}[kind]
        lv, end_s = make(rng, title)
        with zipfile.ZipFile(os.path.join(args.out, title + ".yblevel"), "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("level.txt", lv.to_level_txt(end_s))
            if audio is not None:
                z.writestr("audio.mp3", audio)
    print(f"wrote {args.count} levels to {args.out}")


if __name__ == "__main__":
    main()
