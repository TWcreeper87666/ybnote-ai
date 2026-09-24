"""Nearby-obstacle "local vision" features: what's spatially close to a
query point (the cursor), independent of any note being due there. Existing
per-object features (ChartData.input_features_at) are keyed by TIME
(closest to its due moment) — an obstacle never has a due moment, so it's
invisible to that encoding no matter how the model looks at it. This module
gives the model the other half of what a real player sees: geometry near
where the mouse actually is, so it can notice (and eventually learn to
avoid) a wall it's about to trail through. See TRAIN_DIARY.md 2026-09-24
"obstacle perception"."""

import numpy as np
import torch

MAX_OBSTACLES = 8
OBSTACLE_FEATURE_DIM = 4  # dx, dy, half_w, half_h (relative to the query point)

# A collidable's x/y/w/h are normalized against THIS CHART'S OWN bounds
# (computeBounds() in encodeFrames.js — min/max of its note positions +10%
# padding), same as every other position this training pipeline uses. That
# bounds box only has to contain the NOTES, though — a chart whose notes all
# cluster in one tiny spot (a real one: a single-key "drum" chart whose
# bounds padding left a span of a few dozen world units) can have OTHER
# collidables sitting far outside it, normalizing to values wildly outside
# 0..1 — real observed cases ranged from x=-14.5 up to x=-414. A hard clip
# alone wasn't enough: on the worst chart EVERY one of the 8 obstacle slots
# saturated at the clip boundary simultaneously, an input pattern so far
# outside what any normal chart's (mostly near-zero, at most 1-2 slots ever
# near the boundary) training data looks like that it still produced a
# uniformly dead (sigmoid effectively 0.0) action head on held-out eval.
# tanh squashing instead: smooth, monotonic, and bounded for ANY input
# magnitude with no hard edge — stays close to linear (preserves the fine
# local detail that actually matters, e.g. distinguishing 0.05 from 0.15
# away) for nearby offsets, and gracefully saturates far offsets toward
# ±1 instead of pinning every one of them to the identical clipped value.
# See TRAIN_DIARY.md 2026-09-24 "trail path label".
_SQUASH_SCALE = 0.5


def build_collidable_arrays(collidables: list[dict]):
    """Precompute once per chart: (centers[M,2], half_sizes[M,2]) float32
    numpy arrays, or shape-(0,2) if the chart has no collidables at all."""
    if not collidables:
        empty = np.zeros((0, 2), dtype=np.float32)
        return empty, empty
    centers = np.array([[c["x"] + c["w"] / 2, c["y"] + c["h"] / 2] for c in collidables], dtype=np.float32)
    halves = np.array([[c["w"] / 2, c["h"] / 2] for c in collidables], dtype=np.float32)
    return centers, halves


def nearby_obstacle_features(
    query_xy: tuple[float, float], centers: np.ndarray, halves: np.ndarray, k: int = MAX_OBSTACLES
) -> torch.Tensor:
    """[k, OBSTACLE_FEATURE_DIM] tensor: the k nearest collidables to
    query_xy (by rect-center distance), nearest first, each as (dx, dy,
    half_w, half_h) relative to query_xy. Zero-padded if the chart has
    fewer than k collidables. Every existing chart still gets a
    (mostly-zero) tensor of this shape, so the model's input width never
    depends on which chart it's looking at."""
    out = np.zeros((k, OBSTACLE_FEATURE_DIM), dtype=np.float32)
    m = centers.shape[0]
    if m == 0:
        return torch.from_numpy(out)
    qx, qy = query_xy
    d = centers - np.array([qx, qy], dtype=np.float32)
    dist2 = (d ** 2).sum(axis=1)
    n = min(k, m)
    idx = np.argpartition(dist2, n - 1)[:n]
    idx = idx[np.argsort(dist2[idx])]
    out[:n, 0:2] = d[idx]
    out[:n, 2:4] = halves[idx]
    np.tanh(out / _SQUASH_SCALE, out=out)
    return torch.from_numpy(out)
