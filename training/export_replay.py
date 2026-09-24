"""Runs a policy over a chart (no learning — pure inference) and dumps the
action log ybnote-web's AiReplayDriver plays back
(src/engine/interaction/tools/AiReplayDriver.ts), for the admin-only AI
Replay panel (AiReplayAdminPanel.tsx).

Three policies, pick with --policy:
    dl          (recommended) ChartPolicyNet (dl_model.py) — plain
                backprop-trained feedforward net, cursor regression + attack
                classification from raw per-object features. Requires
                --weights (train_dl.py --save output). See TRAIN_DIARY.md
                2026-09-24 for why this replaced the SNN/R-STDP path.
    neural      real connectome reservoir + two trained readouts:
                CursorReadout (aim) and ReadoutLayer (attack/trail/keybind)
                — requires --weights (train.py --save output). Kept for the
                record — see TRAIN_DIARY.md, this underperforms `dl`.
    engineered  pure threshold+refractory rule reading target_xy() directly
                (engineered_policy.py) — no neural network anywhere, a
                deliberate non-neural comparison point, not the intended
                final answer. --weights ignored/not needed.
All three rate-limit their aim through the same SmoothedCursor (see
TRAIN_DIARY.md 2026-09-23 #13c) — a movement-speed constraint, not a
decision source.

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

    python export_replay.py --frames ../output/test.frames.csv --events ../output/test.events.json \
        --policy engineered --out replay_engineered.json
"""

import argparse
import json
from pathlib import Path

import torch

import config
from connectome import load_connectome, load_roles, synthetic_roles
from cursor_readout import CursorReadout, SmoothedCursor, target_info
from data import ChartData
from dl_model import ChartPolicyNet
from engineered_policy import EngineeredPolicy
from reward import ActionDecoder
from readout import ReadoutLayer
from snn_model import SparseLIFNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--connectome", default=None)
    p.add_argument("--roles", default=None)
    p.add_argument("--policy", choices=["dl", "neural", "engineered"], default="dl")
    p.add_argument("--weights", default=None,
                    help="trained model .pt — train_dl.py --save for --policy dl, "
                         "train.py --save for --policy neural")
    p.add_argument("--out", default="replay.json")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def build_dl_policy(args, chart: ChartData):
    if not args.weights:
        raise SystemExit("--weights is required for --policy dl")
    blob = torch.load(args.weights, weights_only=True)
    features_per_obj = blob.get("features_per_obj", 4)
    model = ChartPolicyNet(blob["max_objects"], features_per_obj, hidden=blob["hidden"])
    model.load_state_dict(blob["state_dict"])
    model.eval()
    print(f"Loaded DL policy from {args.weights} "
          f"({blob.get('best_hits', blob.get('best_holdout_hits', '?'))}/{blob.get('total_notes', '?')} hits at save time)")

    threshold = blob["attack_threshold"]
    refractory_steps = round(blob["refractory_ms"] / config.DT_MS)

    class DLPolicy:
        def __init__(self):
            self.cursor_source = SmoothedCursor()
            self.refractory_left = 0

        def decide(self, features):
            with torch.no_grad():
                cursor_pred, action_logit = model(features.reshape(1, -1))
            cursor = self.cursor_source.step(tuple(cursor_pred[0].tolist()))

            attack_fired = False
            keybind_fired = set()
            if self.refractory_left > 0:
                self.refractory_left -= 1
            elif torch.sigmoid(action_logit).item() > threshold:
                # WHICH action (click vs. which key) is read off the
                # currently-targeted object's own features, not classified —
                # see dl_model.py / cursor_readout.py's target_info().
                info = target_info(features)
                if info is not None:
                    if info["key"] is not None:
                        keybind_fired.add(info["key"])
                    else:
                        attack_fired = True
                    self.refractory_left = refractory_steps

            return {
                "attack_fired": attack_fired, "trail_held": False,
                "keybind_fired": keybind_fired, "cursor": cursor, "output_spike_total": 0,
            }

    return DLPolicy()


def build_policy(args, chart: ChartData):
    if args.policy == "dl":
        return build_dl_policy(args, chart)

    if args.policy == "engineered":
        return EngineeredPolicy()

    if not args.weights:
        raise SystemExit("--weights is required for --policy neural")

    edge_index, weights, num_neurons, root_id_to_idx = load_connectome(args.connectome, seed=args.seed)
    if args.roles:
        input_roles = load_roles(args.roles, root_id_to_idx, num_neurons)
    else:
        print("[export_replay] no --roles given: using synthetic placeholder input roles")
        input_roles = synthetic_roles(num_neurons, seed=args.seed)

    network = SparseLIFNetwork(edge_index, weights, num_neurons)
    decoder = ActionDecoder(input_roles)
    blob = torch.load(args.weights, weights_only=True)
    readout = ReadoutLayer(num_neurons, decoder.num_readout_units)
    readout.W = blob["W"]
    cursor_readout = CursorReadout(num_neurons)
    cursor_readout.W = blob["cursor_W"]
    print(f"Loaded readout from {args.weights} "
          f"(epoch {blob.get('best_epoch', '?')}, {blob.get('best_hits', '?')}/"
          f"{blob.get('total_notes', '?')} hits at save time)")

    class NeuralPolicy:
        def decide(self, features):
            current = decoder.build_input_current(features, network.n)
            reservoir_spikes = network.step(current)
            readout_spikes = readout.step(reservoir_spikes)
            cursor = _cursor_source.step(cursor_readout.predict(reservoir_spikes))
            return decoder.decode(readout_spikes, cursor)

    _cursor_source = SmoothedCursor()
    return NeuralPolicy()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    chart = ChartData(args.frames, args.events)
    policy = build_policy(args, chart)
    print(f"[export_replay] policy={args.policy}")

    bx0, bx1 = chart.bounds["minX"], chart.bounds["maxX"]
    by0, by1 = chart.bounds["minY"], chart.bounds["maxY"]

    def to_world(nx: float, ny: float) -> tuple[float, float]:
        return nx * (bx1 - bx0) + bx0, ny * (by1 - by0) + by0

    print(f"Running {chart.num_steps} steps (inference only, no learning)...")
    entries = []
    for step in range(chart.num_steps):
        t_ms = float(chart.t_ms[step])
        features = chart.input_features_at(step)

        action = policy.decide(features)

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
