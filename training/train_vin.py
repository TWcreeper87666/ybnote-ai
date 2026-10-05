"""Pretrain the map branch's value iteration (nav_map.ValueIteration) on
its own, before behavior cloning uses it frozen.

Maps are rendered from the stroke levels' own geometry the way rl_env
does mid-stroke: the objects around the stroke's start point are the ones
the cursor is inside, what stays put is drawn, the stroke's next note is
the goal. The target is the flood-fill distance on the same map (training
only); what's checked is how well the learned propagation reproduces it on
mazes it never saw (--test-dir).

In train_bc the value maps are cached per distinct map; retraining the
value iteration every iteration (the first --map design) threw that cache
away each time, and at 256x256 x 1280 passes that is ~0.1s a map.

  python training/train_vin.py --save training/models/vin1.pt
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from augment import MODES  # noqa: E402
from data import ChartData  # noqa: E402
from nav_map import MapPlanner, MapRenderer, ValueIteration  # noqa: E402
from niceness import be_nice  # noqa: E402
from trail_plan import LiveGeometry, get_trail_plan  # noqa: E402
from train_rl import find_chart_pairs  # noqa: E402


def stroke_maps(chart: ChartData, per_stroke: int, rng: random.Random) -> list[np.ndarray]:
    renderer = MapRenderer(chart)
    out = []
    for s in get_trail_plan(chart).strokes:
        g0 = LiveGeometry(chart, s.t_start)
        k0 = g0.index.get(chart.events[s.start_uid]["id"])
        if k0 is None:
            continue
        sx, sy = g0.cx[k0], g0.cy[k0]
        inside = {g0.ids[i] for i in np.flatnonzero(g0.inside(np.array([sx]), np.array([sy]))[0])}
        for _ in range(per_stroke):
            t = rng.uniform(s.t_start, s.t_end)
            goal = next((u for u in s.hit_uids if chart.events[u]["time"] >= t - 100), s.hit_uids[-1])
            out.append(renderer.render(t, inside, goal, rng.choice(MODES)))
    return out


def collect(dirs: list[str], names: set[str], charts_dir: str, per_stroke: int, rng) -> list[np.ndarray]:
    pairs = [pr for d in dirs for pr in find_chart_pairs(d)]
    pairs += [pr for pr in find_chart_pairs(charts_dir) if os.path.basename(pr[0])[: -len(".frames.csv")] in names]
    maps = []
    for fp, ep in pairs:
        if not os.path.exists(ep[: -len(".events.json")] + ".trailplan.json"):
            continue
        maps += stroke_maps(ChartData(fp, ep), per_stroke, rng)
    return maps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--save", required=True)
    p.add_argument("--train-dirs", nargs="+", default=["data/output_synth"])
    p.add_argument("--test-dir", default="data/output_synth_test2")
    p.add_argument("--charts-dir", default="data/output")
    p.add_argument("--real", default="只因為你那渴望自由的心臟🫀,迷宮🗣️🔥", help="real stroke charts added to training")
    p.add_argument("--per-stroke", type=int, default=6)
    p.add_argument("--iters", type=int, default=1280, help="propagation passes (>= longest route in cells)")
    p.add_argument("--grad-iters", type=int, default=16)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--init", default="")
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    be_nice(args.threads)
    rng = random.Random(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    train = collect(args.train_dirs, set(filter(None, args.real.split(","))), args.charts_dir, args.per_stroke, rng)
    test = collect([args.test_dir], set(), args.charts_dir, 2, rng)
    print(f"[vin] {len(train)} train maps, {len(test)} test maps ({time.time() - t0:.0f}s)", flush=True)
    vin = ValueIteration().to(device)
    if args.init:
        vin.load_state_dict(torch.load(args.init, map_location="cpu")["vin_state_dict"])
    planner = MapPlanner(vin, device, iters=args.iters)
    opt = torch.optim.Adam(vin.parameters(), lr=args.lr)
    train_arr = np.stack(train)
    test_arr = np.stack(test)
    best = float("inf")
    for step in range(1, args.steps + 1):
        pick = np.random.randint(0, len(train_arr), size=args.batch)
        loss, mae = planner.step_loss(train_arr[pick])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vin.parameters(), 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[vin] step {step} loss={float(loss):.6f} one-pass mae={mae:.2f}cells "
                  f"elapsed={time.time() - t0:.0f}s", flush=True)
        if step % 250 == 0 or step == args.steps:
            with torch.no_grad():
                maes = []
                for i in range(0, len(test_arr), 8):
                    _, m = planner.distance_loss(test_arr[i : i + 8], grad_iters=0)
                    maes.append(m)
            test_mae = float(np.mean(maes))
            print(f"[vin] step {step} TEST mae={test_mae:.2f}cells", flush=True)
            if test_mae < best:
                best = test_mae
                torch.save({"vin_state_dict": {k: v.cpu() for k, v in vin.state_dict().items()},
                            "map_iters": args.iters, "test_mae": test_mae, "step": step}, args.save)
                print(f"[vin] saved {args.save}", flush=True)


if __name__ == "__main__":
    main()
