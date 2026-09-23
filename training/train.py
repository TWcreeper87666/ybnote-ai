"""Local training loop: a frozen FlyWire-connectome LIF reservoir + a small
trainable readout layer learn to play an ybnote chart via energy-penalized
Reward-Modulated STDP. Cursor aim is looked up directly from the chart's own
data (cursor_readout.py's target_xy) — see readout.py / snn_model.py /
TRAIN_DIARY.md's 2026-09-23 #2 and #10/#11 entries for why it isn't learned.

Usage:
    python train.py --frames ../output/test.frames.csv --events ../output/test.events.json
    python train.py --frames ... --events ... --connectome connectome.csv --roles roles.json
    python train.py --frames ... --events ... --epochs 20 --save readout_weights.pt

With no --connectome/--roles, runs on a synthetic random graph + placeholder
input roles — useful to confirm the pipeline runs before plugging in your
real FlyWire data.
"""

import argparse
import random
from collections import Counter

import numpy as np
import torch

import config
from connectome import load_connectome, load_roles, synthetic_roles
from cursor_readout import SmoothedCursor
from data import ChartData
from reward import ActionDecoder, Judge
from readout import ReadoutLayer
from snn_model import SparseLIFNetwork


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True, help="path to *.frames.csv")
    p.add_argument("--events", required=True, help="path to *.events.json")
    p.add_argument("--connectome", default=None, help=".pt or .csv connectome file (omit for synthetic)")
    p.add_argument("--roles", default=None, help="roles.json (omit for synthetic placeholder input roles)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--save", default=None, help="where to torch.save the trained readout layer")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--max-steps", type=int, default=None, help="cap steps/epoch for fast iteration")
    return p.parse_args()


def run_pass(network: SparseLIFNetwork, readout: ReadoutLayer, decoder_factory,
             judge_factory, chart: ChartData, max_steps: int | None = None,
             lr_scale: float = 1.0, train: bool = True):
    """One full pass over the chart. train=False runs pure inference (no
    readout.apply_reward calls at all) — used for evaluate() below, so a
    checkpoint's reported hit count reflects what those exact weights
    actually do on their own, not a cumulative in-training tally that can
    include hits from an earlier, since-overwritten version of the weights
    (see TRAIN_DIARY.md 2026-09-23 #8). Cursor position is looked up
    directly from the chart (SmoothedCursor/target_xy) — never learned, see
    #11 — and rate-limited to a believable mouse speed, see #13."""
    network.reset_episode_state()
    readout.reset_episode_state()
    decoder = decoder_factory()
    judge = judge_factory(chart)
    cursor_source = SmoothedCursor()

    total_reward = 0.0
    total_energy = 0.0

    steps = chart.num_steps if max_steps is None else min(max_steps, chart.num_steps)
    for step in range(steps):
        t_ms = float(chart.t_ms[step])
        features = chart.input_features_at(step)

        current = decoder.build_input_current(features, network.n)
        reservoir_spikes = network.step(current)
        readout_spikes = readout.step(reservoir_spikes)
        cursor = cursor_source.update(features)
        action = decoder.decode(readout_spikes, cursor)

        net_r = judge.step(t_ms, action)
        if train:
            readout.apply_reward(net_r, lr_scale=lr_scale)

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
        input_roles = load_roles(args.roles, root_id_to_idx, num_neurons)
    else:
        print("[train] no --roles given: using synthetic placeholder input roles "
              "(replace with roles.json mapped to your real visual neurons for a real run)")
        input_roles = synthetic_roles(num_neurons, seed=args.seed)

    chart = ChartData(args.frames, args.events)
    print(f"[train] loaded {chart.num_steps} steps ({chart.num_steps * config.DT_MS / 1000:.1f}s), "
          f"{len(chart.events)} notes, connectome: {num_neurons} neurons / {edge_index.shape[1]} synapses")

    network = SparseLIFNetwork(edge_index, weights, num_neurons)
    num_readout_units = ActionDecoder(input_roles).num_readout_units
    readout = ReadoutLayer(num_neurons, num_readout_units)
    print(f"[train] readout layer: {num_readout_units} units "
          f"({config.READOUT_GROUPS}, keybind {config.READOUT_KEYBIND_GROUPS})")

    # R-STDP here tends to find a good solution mid-epoch and then wreck it
    # before the epoch ends (see TRAIN_DIARY.md 2026-09-23 #2/#5/#6/#8) — the
    # in-training hit tally is a cumulative, monotonically-growing count that
    # keeps crediting hits scored by an earlier (since-overwritten) version
    # of the weights, so it can't tell "these weights are good" from "these
    # weights used to be good". After every epoch, run a separate frozen
    # (train=False) evaluation pass with the CURRENT weights and checkpoint
    # off THAT — a true measurement of what this exact snapshot can do.
    best_hits = -1
    best_epoch = 0
    best_W = None

    for epoch in range(1, args.epochs + 1):
        lr_scale = max(0.2, 0.92 ** (epoch - 1))
        # Anneals TRAINING's hit radius from the loose exploration value down
        # to the real one over the run — eval below always uses the real one
        # regardless (Judge's default), never this annealed value. See
        # TRAIN_DIARY.md 2026-09-23 #9: conflating the two is exactly what
        # produced a checkpoint that scored 65% offline and ~0% in the real
        # game. Now that cursor is a direct lookup (not learned noise), this
        # curriculum may not even be necessary — worth trying a constant
        # real-radius run and comparing.
        radius_progress = min(1.0, (epoch - 1) / max(1, args.epochs - 1))
        train_radius = (
            config.HIT_RADIUS_NORM_START
            + (config.HIT_RADIUS_NORM_END - config.HIT_RADIUS_NORM_START) * radius_progress
        )
        total_reward, total_energy, grades = run_pass(
            network, readout, lambda: ActionDecoder(input_roles),
            lambda chart_: Judge(chart_, hit_radius=train_radius), chart,
            max_steps=args.max_steps, lr_scale=lr_scale, train=True,
        )
        train_grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
        train_hits = sum(v for k, v in grades.items() if k in ("Perfect", "Good", "Bad"))

        _, _, eval_grades = run_pass(
            network, readout, lambda: ActionDecoder(input_roles),
            lambda chart_: Judge(chart_), chart,
            max_steps=args.max_steps, train=False,
        )
        eval_grade_str = " ".join(f"{k}:{v}" for k, v in sorted(eval_grades.items()))
        eval_hits = sum(v for k, v in eval_grades.items() if k in ("Perfect", "Good", "Bad"))

        print(f"[epoch {epoch:3d}] lr_scale={lr_scale:.2f}  train_radius={train_radius:.3f}  "
              f"net_reward={total_reward:+.2f}  energy_spent={total_energy:.2f}  "
              f"train_hits={train_hits}/{len(chart.events)} ({train_grade_str})  "
              f"eval_hits(real radius)={eval_hits}/{len(chart.events)} ({eval_grade_str})")

        if eval_hits > best_hits:
            best_hits = eval_hits
            best_epoch = epoch
            best_W = readout.W.clone()

    print(f"[train] best epoch: {best_epoch} ({best_hits}/{len(chart.events)} eval hits)")

    if args.save:
        torch.save(
            {
                "W": best_W.cpu(),
                "n_in": readout.n_in,
                "n_out": readout.n_out,
                "readout_groups": config.READOUT_GROUPS,
                "readout_keybind_groups": config.READOUT_KEYBIND_GROUPS,
                "best_epoch": best_epoch,
                "best_hits": best_hits,
                "total_notes": len(chart.events),
            },
            args.save,
        )
        print(f"[train] saved BEST readout layer (epoch {best_epoch}) -> {args.save}")


if __name__ == "__main__":
    main()
