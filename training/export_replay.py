"""Runs a trained readout layer over a chart (no learning — pure inference)
and dumps the action log ybnote-web's AiReplayDriver plays back
(src/engine/interaction/tools/AiReplayDriver.ts), for the admin-only AI
Replay panel (AiReplayAdminPanel.tsx).

Output schema (matches AiReplayEntry in AiReplayDriver.ts):
    [{ "t": <chart ms>, "cursorX": <world x>, "cursorY": <world y>,
       "attack": <bool, edge-triggered>, "trailHeld": <bool, level-triggered>,
       "keybindsFired": [<key>, ...] }, ...]

cursorX/cursorY are WORLD coordinates (denormalized via chart.bounds from
test.events.json) — AiReplayDriver feeds them straight into
PixiApproachCircleManager.checkTrailIntersection(), which expects world
space, not the 0..1 normalized space training/reward.py works in.

Usage:
    python export_replay.py --frames ../output/test.frames.csv --events ../output/test.events.json \
        --connectome connectome.csv --roles roles.json --weights trained_readout.pt --out replay.json
"""

import argparse
import json
from pathlib import Path

import torch

import config
from connectome import load_connectome, load_roles, synthetic_roles
from cursor_readout import SmoothedCursor
from data import ChartData
from reward import ActionDecoder
from readout import ReadoutLayer
from snn_model import SparseLIFNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--connectome", default=None)
    p.add_argument("--roles", default=None)
    p.add_argument("--weights", required=True, help="trained readout .pt (train.py --save)")
    p.add_argument("--out", default="replay.json")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    edge_index, weights, num_neurons, root_id_to_idx = load_connectome(args.connectome, seed=args.seed)
    if args.roles:
        input_roles = load_roles(args.roles, root_id_to_idx, num_neurons)
    else:
        print("[export_replay] no --roles given: using synthetic placeholder input roles")
        input_roles = synthetic_roles(num_neurons, seed=args.seed)

    chart = ChartData(args.frames, args.events)
    network = SparseLIFNetwork(edge_index, weights, num_neurons)
    decoder = ActionDecoder(input_roles)

    blob = torch.load(args.weights, weights_only=True)
    readout = ReadoutLayer(num_neurons, decoder.num_readout_units)
    readout.W = blob["W"]
    print(f"Loaded readout from {args.weights} "
          f"(epoch {blob.get('best_epoch', '?')}, {blob.get('best_hits', '?')}/"
          f"{blob.get('total_notes', '?')} hits at save time)")

    bx0, bx1 = chart.bounds["minX"], chart.bounds["maxX"]
    by0, by1 = chart.bounds["minY"], chart.bounds["maxY"]

    def to_world(nx: float, ny: float) -> tuple[float, float]:
        return nx * (bx1 - bx0) + bx0, ny * (by1 - by0) + by0

    print(f"Running {chart.num_steps} steps (inference only, no learning)...")
    entries = []
    cursor_source = SmoothedCursor()
    for step in range(chart.num_steps):
        t_ms = float(chart.t_ms[step])
        features = chart.input_features_at(step)

        current = decoder.build_input_current(features, network.n)
        reservoir_spikes = network.step(current)
        readout_spikes = readout.step(reservoir_spikes)
        cursor = cursor_source.update(features)
        action = decoder.decode(readout_spikes, cursor)

        world_x, world_y = to_world(*action["cursor"])
        entries.append({
            "t": t_ms,
            "cursorX": round(world_x, 2),
            "cursorY": round(world_y, 2),
            "attack": bool(action["attack_fired"]),
            "trailHeld": bool(action["trail_held"]),
            "keybindsFired": sorted(action["keybind_fired"]),
        })
        if (step + 1) % 2000 == 0:
            print(f"  {step + 1}/{chart.num_steps}")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(entries), encoding="utf-8")
    attacks = sum(1 for e in entries if e["attack"])
    trail_steps = sum(1 for e in entries if e["trailHeld"])
    print(f"Wrote {out_path}: {len(entries)} entries, {attacks} attack triggers, "
          f"{trail_steps} steps with trail held")


if __name__ == "__main__":
    main()
