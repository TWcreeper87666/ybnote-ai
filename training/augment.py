"""D4 (rotate 0/90/180/270 x mirror) geometric augmentation for training
batches. The chart canvas has no inherent up/down or left/right meaning to
the model — proximity/timing/keybind features are orientation-independent,
and only x/y-ish quantities need transforming — so relabeling the same
frame under any of the 8 square symmetries is a label-preserving
transform, free extra training variety for every chart, not just the
maze-style ones (2026-09-24: the user's own suggestion for helping the
single-maze-chart generalize past memorizing that one layout).

Applied ON THE FLY per mini-batch in train_dl_multi.py rather than
pre-materialized 8x over the whole corpus — 26 charts' combined ~270k
steps at the current feature width already flirts with memory limits at
1x (see TRAIN_DIARY.md 2026-09-24 "keybind support"'s pandas/memory
incident); a random mode per batch gives the same expected variety across
epochs without holding 8 copies in RAM at once."""

import torch

MODES = ("identity", "rot90", "rot180", "rot270", "mirror", "mirror_rot90", "mirror_rot180", "mirror_rot270")

_SWAPS_HALFSIZE = frozenset({"rot90", "rot270", "mirror_rot90", "mirror_rot270"})


def _transform_point(xy: torch.Tensor, mode: str) -> torch.Tensor:
    """xy: [...,2] ABSOLUTE position in 0..1 space — transformed around the
    canvas center (0.5, 0.5)."""
    x, y = xy[..., 0], xy[..., 1]
    if mode == "identity":
        nx, ny = x, y
    elif mode == "rot90":
        nx, ny = 1 - y, x
    elif mode == "rot180":
        nx, ny = 1 - x, 1 - y
    elif mode == "rot270":
        nx, ny = y, 1 - x
    elif mode == "mirror":
        nx, ny = 1 - x, y
    elif mode == "mirror_rot90":
        nx, ny = 1 - y, 1 - x
    elif mode == "mirror_rot180":
        nx, ny = x, 1 - y
    elif mode == "mirror_rot270":
        nx, ny = y, x
    else:
        raise ValueError(mode)
    return torch.stack([nx, ny], dim=-1)


def _transform_vector(dxdy: torch.Tensor, mode: str) -> torch.Tensor:
    """dxdy: [...,2] RELATIVE offset (e.g. an obstacle's position minus the
    query point) — same linear map as _transform_point but with no center
    translation (a difference of two transformed points is just the
    transformed difference)."""
    dx, dy = dxdy[..., 0], dxdy[..., 1]
    if mode == "identity":
        ndx, ndy = dx, dy
    elif mode == "rot90":
        ndx, ndy = -dy, dx
    elif mode == "rot180":
        ndx, ndy = -dx, -dy
    elif mode == "rot270":
        ndx, ndy = dy, -dx
    elif mode == "mirror":
        ndx, ndy = -dx, dy
    elif mode == "mirror_rot90":
        ndx, ndy = -dy, -dx
    elif mode == "mirror_rot180":
        ndx, ndy = dx, -dy
    elif mode == "mirror_rot270":
        ndx, ndy = dy, dx
    else:
        raise ValueError(mode)
    return torch.stack([ndx, ndy], dim=-1)


def _transform_halfsize(hwhh: torch.Tensor, mode: str) -> torch.Tensor:
    """hwhh: [...,2] a rect's (half_w, half_h) — direction-less, but a 90°
    turn swaps which axis is "width" vs "height"."""
    return hwhh.flip(-1) if mode in _SWAPS_HALFSIZE else hwhh


def augment_batch(
    x_obj: torch.Tensor, x_obstacle: torch.Tensor, y_cursor: torch.Tensor, features_per_obj: int, mode: str
):
    """x_obj: [N, max_objects*features_per_obj] (each object block is
    proximity, x, y, keybind, key-one-hot...  only x/y at offset 1:3
    transform). x_obstacle: [N, num_obstacle_slots*4] (each slot is dx, dy,
    half_w, half_h). y_cursor: [N, 2]. Returns transformed copies; `mode ==
    "identity"` still clones (caller mutates freely without touching the
    originals)."""
    n = x_obj.shape[0]
    max_objects = x_obj.shape[1] // features_per_obj
    obj = x_obj.view(n, max_objects, features_per_obj).clone()
    obj[..., 1:3] = _transform_point(obj[..., 1:3], mode)
    obj = obj.reshape(n, -1)

    n_obstacle_slots = x_obstacle.shape[1] // 4
    obs = x_obstacle.view(n, n_obstacle_slots, 4).clone()
    obs[..., 0:2] = _transform_vector(obs[..., 0:2], mode)
    obs[..., 2:4] = _transform_halfsize(obs[..., 2:4], mode)
    obs = obs.reshape(n, -1)

    cursor = _transform_point(y_cursor, mode)
    return obj, obs, cursor
