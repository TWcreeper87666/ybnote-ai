"""Loads the FlyWire connectome into edge_index/weights tensors, and resolves
which neurons play which role (visual input channel, motor output group) via
roles.json.

Two connectome source formats are accepted:
  - .pt  : torch.save({"edge_index": LongTensor[2,E], "weights": FloatTensor[E],
                        "num_neurons": int, "root_ids": list[int] (optional)})
  - .csv : columns pre_root_id,post_root_id,weight (typical FlyWire export) —
           arbitrary root IDs get remapped to a contiguous [0, N) index space.

If no connectome file is given, a small random sparse graph is generated so
the training loop can be smoke-tested without your real data.
"""

import json
import csv
import random
from pathlib import Path

import torch


def load_connectome(path: str | None, num_neurons_fallback: int = 71, seed: int = 0):
    """Returns (edge_index [2,E] long, weights [E] float, num_neurons, root_id_to_idx)."""
    if path is None:
        return _synthetic_connectome(num_neurons_fallback, seed)

    p = Path(path)
    if p.suffix == ".pt":
        blob = torch.load(p, weights_only=True)
        edge_index = blob["edge_index"].long()
        weights = blob["weights"].float()
        num_neurons = int(blob["num_neurons"])
        root_ids = blob.get("root_ids")
        root_id_to_idx = {rid: i for i, rid in enumerate(root_ids)} if root_ids else None
        return edge_index, weights, num_neurons, root_id_to_idx

    if p.suffix == ".csv":
        return _load_csv_connectome(p)

    raise ValueError(f"Unrecognized connectome file type: {path}")


# neuprint's "weight" is a raw synapse COUNT (can be dozens to hundreds),
# not a tuned LIF connection strength — feeding that straight into the
# recurrent-current sum saturates the whole network (every neuron spikes
# every step, ~100% activity, i.e. exactly the "epilepsy" the original
# notebook's cell-3 ran into with the same data before it applied its own
# `* 0.005` fudge factor). Applying that same order-of-magnitude scale here
# keeps CSV-sourced (real, unscaled) connectomes in the same regime as the
# hand-tuned synthetic one (whose weights already sit in +/-[0.2, 1.0]).
REAL_CONNECTOME_WEIGHT_SCALE = 0.001


def _load_csv_connectome(path: Path):
    root_id_to_idx: dict[int, int] = {}
    src, dst, w = [], [], []

    def idx_for(rid: int) -> int:
        if rid not in root_id_to_idx:
            root_id_to_idx[rid] = len(root_id_to_idx)
        return root_id_to_idx[rid]

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pre = idx_for(int(row["pre_root_id"]))
            post = idx_for(int(row["post_root_id"]))
            src.append(pre)
            dst.append(post)
            w.append(float(row["weight"]) * REAL_CONNECTOME_WEIGHT_SCALE)

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    weights = torch.tensor(w, dtype=torch.float32)
    return edge_index, weights, len(root_id_to_idx), root_id_to_idx


def _synthetic_connectome(num_neurons: int, seed: int):
    """Random sparse recurrent graph — structurally plausible stand-in for
    smoke-testing the training loop, NOT a real connectome."""
    rng = random.Random(seed)
    edges_per_neuron = 6
    src, dst, w = [], [], []
    for i in range(num_neurons):
        targets = rng.sample(range(num_neurons), k=min(edges_per_neuron, num_neurons - 1))
        for j in targets:
            if j == i:
                continue
            src.append(i)
            dst.append(j)
            sign = 1.0 if rng.random() < 0.8 else -1.0  # ~80% excitatory, like real cortex/connectome stats
            w.append(sign * rng.uniform(0.2, 1.0))

    edge_index = torch.tensor([src, dst], dtype=torch.long)
    weights = torch.tensor(w, dtype=torch.float32)
    return edge_index, weights, num_neurons, None


def load_roles(path: str, root_id_to_idx: dict[int, int] | None, num_neurons: int):
    """Reads roles.json (see roles.example.json) and resolves its
    "input_neurons" section to reservoir indices — real neurons the chart's
    visual features get injected into. "output_neurons" is no longer read:
    output roles now live in the separate trained ReadoutLayer (readout.py),
    laid out by config.READOUT_GROUPS instead of real bodyIds — see
    TRAIN_DIARY.md's 2026-09-23 #2 entry for why. If root_id_to_idx is None
    (synthetic/.pt-without-root_ids connectome), roles.json's lists are
    interpreted as raw indices already."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    def resolve(ids: list[int]) -> list[int]:
        if root_id_to_idx is None:
            return list(ids)
        return [root_id_to_idx[i] for i in ids]

    return {k: resolve(v) for k, v in raw["input_neurons"].items()}


def synthetic_roles(num_neurons: int, seed: int = 0):
    """Deterministic placeholder input-role assignment for smoke-testing when
    no roles.json / real FlyWire visual neuron IDs are available yet. Replace
    with load_roles() + your actual neuron manifest for real runs."""
    rng = random.Random(seed)
    pool = list(range(num_neurons))
    rng.shuffle(pool)

    def take(n):
        nonlocal pool
        chunk, pool = pool[:n], pool[n:]
        return chunk

    return {
        "proximity": take(4),
        "x": take(4),
        "y": take(4),
        "keybind": take(4),
    }
