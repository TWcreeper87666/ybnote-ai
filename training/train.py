"""Local training loop: FlyWire-connectome LIF network learns to play an
ybnote chart via energy-penalized Reward-Modulated STDP.

Usage:
    python train.py --frames ../output/test.frames.csv --events ../output/test.events.json
    python train.py --frames ... --events ... --connectome my_connectome.csv --roles roles.json
    python train.py --frames ... --events ... --epochs 20 --save trained_weights.pt

With no --connectome/--roles, runs on a synthetic random graph + placeholder
role assignment — useful to confirm the pipeline runs before plugging in your
real 71-neuron/2027-synapse FlyWire data.
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch

import config
from connectome import load_connectome, load_roles, synthetic_roles
from data import ChartData
from reward import ActionDecoder, Judge
from snn_model import SparseLIFNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="path to *.frames.csv")
    p.add_argument("--events", required=True, help="path to *.events.json")
    p.add_argument("--connectome", default=None, help=".pt or .csv connectome file (omit for synthetic)")
    p.add_argument("--roles", default=None, help="roles.json (omit for synthetic placeholder roles)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save", default=None, help="where to torch.save the trained weights/edge_index")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def run_epoch(network: SparseLIFNetwork, decoder_factory, judge_factory, chart: ChartData):
    network.reset_episode_state()
    decoder = decoder_factory()
    judge = judge_factory(chart)

    total_reward = 0.0
    total_energy = 0.0

    for step in range(chart.num_steps):
        t_ms = float(chart.t_ms[step])
        features = chart.input_features_at(step)

        current = decoder.build_input_current(features, network.n)
        spikes = network.step(current)
        action = decoder.decode(spikes)

        net_r = judge.step(t_ms, action)
        network.apply_reward(net_r)

        total_reward += net_r
        total_energy += config.ENERGY_COST_PER_SPIKE * action["output_spike_total"]

    grades = Counter(entry["judgment"] for entry in judge.log)
    return total_reward, total_energy, grades


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    edge_index, weights, num_neurons, root_id_to_idx = load_connectome(
        args.connectome, seed=args.seed
    )
    if args.roles:
        input_roles, output_roles = load_roles(args.roles, root_id_to_idx, num_neurons)
    else:
        print("[train] no --roles given: using synthetic placeholder role assignment "
              "(replace with roles.json mapped to your real visual/motor neurons for a real run)")
        input_roles, output_roles = synthetic_roles(num_neurons, seed=args.seed)

    chart = ChartData(args.frames, args.events)
    print(f"[train] loaded {chart.num_steps} steps ({chart.num_steps * config.DT_MS / 1000:.1f}s), "
          f"{len(chart.events)} notes, connectome: {num_neurons} neurons / {edge_index.shape[1]} synapses")

    network = SparseLIFNetwork(edge_index, weights, num_neurons)

    for epoch in range(1, args.epochs + 1):
        total_reward, total_energy, grades = run_epoch(
            network,
            lambda: ActionDecoder(input_roles, output_roles),
            lambda chart_: Judge(chart_),
            chart,
        )
        grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
        print(f"[epoch {epoch:3d}] net_reward={total_reward:+.2f}  energy_spent={total_energy:.2f}  {grade_str}")

    if args.save:
        torch.save(
            {
                "edge_index": network.edge_index.cpu(),
                "weights": network.weights.cpu(),
                "num_neurons": network.n,
                "root_ids": list(root_id_to_idx.keys()) if root_id_to_idx else None,
            },
            args.save,
        )
        print(f"[train] saved trained connectome weights -> {args.save}")


if __name__ == "__main__":
    main()
