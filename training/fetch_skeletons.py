"""Fetches simplified 3D skeleton geometry for every neuron in connectome.csv,
for the "whole fly brain" Three.js viewer.

neuprint's fetch_skeleton() returns every traced skeleton node (a single
Giant Fiber alone has 11,375 — 789 neurons untouched would be well past a
million points, way too much for a smooth browser scene). decimate_tree()
below keeps the skeleton's actual shape (root, every branch point, every
leaf) while thinning long straight-ish runs down to a target point budget,
and reparents kept nodes to their nearest surviving ancestor so the
simplified tree is still fully connected (no dangling segments).

Run this somewhere with internet + NEUPRINT_TOKEN (see fetch_real_connectome.py's
header for how — same .env loading here). Safe to re-run/interrupt: already-
fetched neurons are cached to skeletons/ and skipped on the next run.

Output:
    skeletons.json  {bodyId: {points: [[x,y,z], ...], parents: [parentIdx, ...]}, ...}
                    coordinates are re-centered and scaled to roughly fit a
                    [-50, 50] cube (see NORMALIZE_TARGET_RADIUS) — raw
                    hemibrain coordinates are in ~8nm voxel units, values in
                    the tens of thousands, not something to hand a WebGL
                    camera directly.
"""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from neuprint import Client, fetch_skeleton

OUT_DIR = Path(__file__).parent
CACHE_DIR = OUT_DIR / "skeletons"
TARGET_POINTS_PER_NEURON = 150
# Hard ceiling regardless of how branch-heavy a neuron is — decimate_tree()
# never thins branch points (every fork is kept unconditionally to preserve
# shape), so a densely-arborized neuron can still land far above
# TARGET_POINTS_PER_NEURON. cap_points() enforces this after the fact.
HARD_CAP_PER_NEURON = 300
MAX_WORKERS = 8
NORMALIZE_TARGET_RADIUS = 50.0


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(OUT_DIR / ".env")


def decimate_tree(df: pd.DataFrame, target_points: int) -> dict:
    """df: neuprint's raw skeleton dataframe (rowId, x, y, z, radius, link).
    Returns {"points": [[x,y,z],...], "parents": [parentIndex or -1, ...]},
    reindexed to 0..K-1 over only the kept points."""
    by_row = df.set_index("rowId")
    children_count: dict[int, int] = {}
    for link in df["link"]:
        if link != -1:
            children_count[link] = children_count.get(link, 0) + 1

    n = len(df)
    stride = max(1, n // max(1, target_points))

    kept_points: list[list[float]] = []
    kept_index_by_row: dict[int, int] = {}
    kept_parent: list[int] = []

    # child_of[rowId] built while walking parent->children in traversal order
    # below; a plain adjacency list from `link`.
    adjacency: dict[int, list[int]] = {}
    roots: list[int] = []
    for row_id, link in zip(df["rowId"], df["link"]):
        if link == -1:
            roots.append(row_id)
        else:
            adjacency.setdefault(link, []).append(row_id)

    def keep(row_id: int, parent_kept_idx: int):
        row = by_row.loc[row_id]
        idx = len(kept_points)
        kept_points.append([float(row["x"]), float(row["y"]), float(row["z"])])
        kept_parent.append(parent_kept_idx)
        kept_index_by_row[row_id] = idx
        return idx

    # Iterative DFS: (row_id, nearest_kept_ancestor_idx, steps_since_kept)
    for root in roots:
        stack = [(root, -1, 0)]
        while stack:
            row_id, ancestor_idx, steps = stack.pop()
            is_leaf = row_id not in adjacency
            is_branch = children_count.get(row_id, 0) >= 2
            is_root = ancestor_idx == -1
            should_keep = is_root or is_leaf or is_branch or steps >= stride

            if should_keep:
                this_idx = keep(row_id, ancestor_idx)
                next_ancestor, next_steps = this_idx, 0
            else:
                next_ancestor, next_steps = ancestor_idx, steps + 1

            for child in adjacency.get(row_id, []):
                stack.append((child, next_ancestor, next_steps))

    return {"points": kept_points, "parents": kept_parent}


def cap_points(points: list, parents: list, hard_cap: int, seed: int = 0):
    """Forces the point count under hard_cap by random subsampling (root
    always kept), reparenting each surviving point to its nearest surviving
    ancestor so the result is still one connected tree — used when
    decimate_tree()'s branch-point-always-kept rule alone isn't enough
    (a densely-arborized neuron can have thousands of forks)."""
    n = len(points)
    if n <= hard_cap:
        return points, parents

    rng = random.Random(seed)
    candidates = list(range(1, n))
    keep = {0, *rng.sample(candidates, hard_cap - 1)}
    keep_sorted = sorted(keep)
    old_to_new = {old: new for new, old in enumerate(keep_sorted)}

    def nearest_kept_ancestor(i: int) -> int:
        p = parents[i]
        while p != -1 and p not in keep:
            p = parents[p]
        return p

    new_points = [points[i] for i in keep_sorted]
    new_parents = []
    for i in keep_sorted:
        if i == 0:
            new_parents.append(-1)
        else:
            ancestor = nearest_kept_ancestor(i)
            new_parents.append(old_to_new[ancestor] if ancestor != -1 else -1)
    return new_points, new_parents


def fetch_one(body_id: int, client: Client) -> dict | None:
    cache_path = CACHE_DIR / f"{body_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    try:
        df = fetch_skeleton(body_id, heal=True, client=client)
    except Exception as e:  # neuprint raises assorted errors for untraced/missing bodies
        print(f"  skip {body_id}: {e}")
        return None
    if df is None or len(df) == 0:
        return None
    simplified = decimate_tree(df, TARGET_POINTS_PER_NEURON)
    points, parents = cap_points(
        simplified["points"], simplified["parents"], HARD_CAP_PER_NEURON, seed=body_id
    )
    simplified = {"points": points, "parents": parents}
    cache_path.write_text(json.dumps(simplified), encoding="utf-8")
    return simplified


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit("Set NEUPRINT_TOKEN (env var or training/.env) first.")
    client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=token)
    CACHE_DIR.mkdir(exist_ok=True)

    conn = pd.read_csv(OUT_DIR / "connectome.csv")
    body_ids = sorted(set(conn["pre_root_id"]) | set(conn["post_root_id"]))
    print(f"Fetching skeletons for {len(body_ids)} neurons "
          f"(cached ones in {CACHE_DIR}/ are skipped)...")

    skeletons: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(fetch_one, bid, client): bid for bid in body_ids}
        done = 0
        for future in as_completed(futures):
            bid = futures[future]
            result = future.result()
            done += 1
            if result is not None:
                skeletons[str(bid)] = result
            if done % 50 == 0 or done == len(body_ids):
                print(f"  {done}/{len(body_ids)} done ({len(skeletons)} with geometry)")

    normalize_in_place(skeletons)

    out_path = OUT_DIR / "skeletons.json"
    out_path.write_text(json.dumps(skeletons), encoding="utf-8")
    total_points = sum(len(s["points"]) for s in skeletons.values())
    print(f"Wrote {out_path}: {len(skeletons)} neurons, {total_points} points total.")


def normalize_in_place(skeletons: dict[str, dict]) -> None:
    """Centers on the combined bounding-box center and scales so the longest
    axis spans NORMALIZE_TARGET_RADIUS*2 — same transform applied to every
    neuron, so relative position/shape between neurons is preserved."""
    min_c = [float("inf")] * 3
    max_c = [float("-inf")] * 3
    for s in skeletons.values():
        for p in s["points"]:
            for i in range(3):
                min_c[i] = min(min_c[i], p[i])
                max_c[i] = max(max_c[i], p[i])

    center = [(min_c[i] + max_c[i]) / 2 for i in range(3)]
    span = max(max_c[i] - min_c[i] for i in range(3)) or 1.0
    scale = (2 * NORMALIZE_TARGET_RADIUS) / span

    for s in skeletons.values():
        # Rounded to 2dp — plenty of precision in a [-50,50] normalized
        # scene, and roughly halves the exported JSON's size versus full
        # float repr (789 neurons x up to 300 points each adds up fast).
        s["points"] = [
            [round((p[i] - center[i]) * scale, 2) for i in range(3)]
            for p in s["points"]
        ]


if __name__ == "__main__":
    main()
