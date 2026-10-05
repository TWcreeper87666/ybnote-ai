"""Dump one REAL local-view sample (nav_map.render_local_view) for the report figure.

    python scripts/export_local_view_sample.py   # -> reports/report_figs/local_view_real.npz

Level: JAWNY - Honeypie, the note on block `noteblock-q4lgzar4` (event 94),
100 ms before it is due, cursor 11/-11 world units off the block's live center.
Saved: the 8 packed bit-planes (as the CNN input is stored), the objects
inside the 128x128 window (world coords relative to the cursor), and the
raw frames.json frame / events.json event for that moment.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "training"))
os.chdir(os.path.join(ROOT, "training"))
from data import ChartData  # noqa: E402
from nav_map import LOCAL_VIEW, LOCAL_VIEW_CELL_WORLD, _VIEW_PLANES, render_local_view  # noqa: E402

LEVEL = "JAWNY - Honeypie"
EVENT_INDEX, DX, DY = 94, 11, -11
base = os.path.join(ROOT, "data", "output", LEVEL)
chart = ChartData(base + ".frames.csv", base + ".events.json")
ev = chart.events[EVENT_INDEX]
t = ev["time"] - 100
b = chart.bounds
# the target block may be carried by a track: place the cursor off its LIVE center
live = next(c for c in chart.live_collidables_at(t) if c["id"] == ev["id"])
lx, ly = chart.world_xy(live["x"], live["y"])
half_w = live["w"] * (b["maxX"] - b["minX"]) / 2
cwx, cwy = lx + half_w + DX, ly + half_w + DY
cur = ((cwx - b["minX"]) / (b["maxX"] - b["minX"]), (cwy - b["minY"]) / (b["maxY"] - b["minY"]))
packed = render_local_view(chart, t, cur, "identity", None, ev["id"])
planes = np.unpackbits(packed).reshape(_VIEW_PLANES, LOCAL_VIEW, LOCAL_VIEW)

half = LOCAL_VIEW * LOCAL_VIEW_CELL_WORLD / 2
objs = []
for c in chart.live_collidables_at(t):
    x0, y0 = chart.world_xy(c["x"], c["y"])
    w = c["w"] * (b["maxX"] - b["minX"])
    h = c["h"] * (b["maxY"] - b["minY"])
    if x0 + w < cwx - half or x0 > cwx + half or y0 + h < cwy - half or y0 > cwy + half:
        continue
    objs.append({"id": c["id"], "type": c["type"], "x": x0 - cwx, "y": y0 - cwy, "w": w, "h": h,
                 "rotation_deg": c.get("rotation_deg", 0)})

frame_json = json.load(open(base + ".frames.json", encoding="utf8"))
step = int(np.argmin(np.abs(chart.t_ms - t)))
frame = frame_json["frames"][step]

out = os.path.join(ROOT, "reports", "report_figs", "local_view_real.npz")
np.savez_compressed(
    out,
    planes=planes,
    meta=json.dumps({"level": LEVEL, "event_index": EVENT_INDEX, "event": {k: v for k, v in ev.items() if k != "_uid"},
                     "t_ms": t, "next_id": ev["id"], "objects": objs, "frame": frame,
                     "cell_world": LOCAL_VIEW_CELL_WORLD, "view": LOCAL_VIEW}, ensure_ascii=False),
)
print(out, "objects in window:", len(objs), [o["type"] for o in objs])
print("plane sums:", planes.sum(axis=(1, 2)).tolist())
