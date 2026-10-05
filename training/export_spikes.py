"""Runs one pass over a chart and records which real neurons (by bodyId)
spike on every timestep — the data the 3D viewer animates against.

The reservoir is frozen (see snn_model.py / TRAIN_DIARY.md 2026-09-23 #2),
so this recording is purely a function of the connectome + visual input —
it does NOT depend on how (or whether) the readout layer has been trained.
No --weights flag here for that reason; train.py's --save output is the
readout layer, which has nothing to do with what the reservoir itself does.
Reuses fetch_skeletons.py's neuron set and fetch_real_connectome.py's
connectome.csv, so bodyIds line up with skeletons.json's keys — see
build_viewer_bundle.py, which merges this output with skeletons.json into
what the Artifact actually loads.

Usage:
    python export_spikes.py --frames ../data/output/test.frames.csv --events ../data/output/test.events.json \
        --connectome connectome.csv --roles roles.json --out spikes.json
"""

import argparse
import json
from pathlib import Path

import torch

import config
from connectome import load_connectome, load_roles, synthetic_roles
from data import ChartData
from reward import ActionDecoder
from snn_model import SparseLIFNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--connectome", default=None)
    p.add_argument("--roles", default=None)
    p.add_argument("--out", default="spikes.json")
    p.add_argument("--max-steps", type=int, default=None, help="cap steps for a shorter preview export")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    edge_index, weights, num_neurons, root_id_to_idx = load_connectome(args.connectome, seed=args.seed)

    if args.roles:
        input_roles = load_roles(args.roles, root_id_to_idx, num_neurons)
    else:
        print("[export_spikes] no --roles given: using synthetic placeholder input roles")
        input_roles = synthetic_roles(num_neurons, seed=args.seed)

    chart = ChartData(args.frames, args.events)
    network = SparseLIFNetwork(edge_index, weights, num_neurons)
    decoder = ActionDecoder(input_roles)

    steps = chart.num_steps if args.max_steps is None else min(args.max_steps, chart.num_steps)
    print(f"Running {steps} steps over {num_neurons} neurons / {edge_index.shape[1]} synapses...")

    spikes_per_step: list[list[int]] = []
    for step in range(steps):
        features = chart.input_features_at(step)
        current = decoder.build_input_current(features, network.n)
        spikes = network.step(current)
        active = spikes.nonzero().flatten().tolist()
        spikes_per_step.append(active)
        if (step + 1) % 2000 == 0:
            print(f"  {step + 1}/{steps}")

    neuron_order: list[int | None] = [None] * num_neurons
    if root_id_to_idx:
        for root_id, idx in root_id_to_idx.items():
            neuron_order[idx] = root_id
    else:
        neuron_order = list(range(num_neurons))  # synthetic connectome — no real bodyIds

    out = {
        "dtMs": config.DT_MS,
        "neuronOrder": neuron_order,
        "spikesPerStep": spikes_per_step,
    }
    out_path = Path(args.out)
    out_path.write_text(json.dumps(out), encoding="utf-8")
    total_spikes = sum(len(s) for s in spikes_per_step)
    print(f"Wrote {out_path}: {steps} steps, {total_spikes} total spikes "
          f"({total_spikes / max(1, steps * num_neurons) * 100:.3f}% avg activity)")


if __name__ == "__main__":
    main()
