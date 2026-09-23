"""Fetches a REAL visual->motor pathway from the hemibrain connectome via
neuprint, instead of the mushroom-body (MB.*) pool the original notebook
used (which has no natural visual-input or motor-output population).

Circuit: LC4 / LPLC2 (looming-detecting visual projection neurons — an
approach circle shrinking IS a looming stimulus) -> ... -> DNp01 (the Giant
Fiber descending neuron, triggers the jump/takeoff escape response). This is
one of the best-characterized sensorimotor circuits in the hemibrain dataset
(von Reyn et al. 2017), so "visual in, motor out" actually means something
biologically here.

Verified against neuprint-python 0.6.3's actual source (connectivity.py) —
fetch_shortest_paths/fetch_paths take ONE bodyId each for upstream/
downstream, not a type or a list, so this resolves LC4/LPLC2/DNp01 to
concrete bodyIds first (fetch_neurons) and loops pairwise.

Run this somewhere with internet + a neuprint account (Colab, or locally).
Needs a token: set it as an env var, NEVER hardcode it in a script/notebook
you might share or commit —
    export NEUPRINT_TOKEN="..."          (bash)
    $env:NEUPRINT_TOKEN = "..."           (PowerShell)
    os.environ["NEUPRINT_TOKEN"] = "..."  (Colab cell, from a Colab Secret)

Output (written next to this script):
    connectome.csv    pre_root_id,post_root_id,weight — feed straight into
                      ybnote-ai/training/connectome.py's load_connectome()
    roles.draft.json  bodyIds split into a "visual_pool" (nearer the LC4/
                      LPLC2 sources) and "motor_pool" (nearer DNp01) — a
                      STARTING point, not a finished roles.json: you still
                      need to split visual_pool across proximity/x/y/keybind
                      input channels and motor_pool across cursor_x/cursor_y/
                      attack_gate (this pathway has no natural trail_gate/
                      keybind_groups analog — those will need other neurons,
                      or stay an unbiological trained readout like the
                      original notebook's W_out already is).
"""

import json
import os
from pathlib import Path

from neuprint import Client, NeuronCriteria as NC
from neuprint import fetch_neurons, fetch_paths, fetch_adjacencies

OUT_DIR = Path(__file__).parent


def _load_dotenv(path: Path) -> None:
    """Tiny no-dependency .env loader — only sets vars not already in the
    environment, same precedence as python-dotenv's default."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(OUT_DIR / ".env")

VISUAL_SOURCE_TYPES = ["LC4", "LPLC2"]
# hemibrain:v1.2.1 names this neuron "Giant Fiber" (type/instance), not
# "DNp01" — that name is from a different dataset (FlyWire/FANC). Verified
# against the live API: fetch_neurons(NC(type="Giant Fiber")) returns exactly
# one bodyId (2307027729, "Giant Fiber_R") in this dataset, which is expected
# — hemibrain only traced one hemisphere, so bilateral pairs like GF appear
# once here.
MOTOR_TARGET_TYPES = ["Giant Fiber"]
MAX_SOURCES_PER_TYPE = 8  # caps the O(sources x targets) pairwise path search
MAX_HOPS = 4
MIN_PATH_WEIGHT = 3  # drop noise-level synapse counts along candidate paths
PATH_QUERY_TIMEOUT_S = 15.0


def main():
    token = os.environ.get("NEUPRINT_TOKEN")
    if not token:
        raise SystemExit(
            "Set NEUPRINT_TOKEN as an environment variable first (see this "
            "file's header comment) — never hardcode it."
        )
    client = Client("neuprint.janelia.org", dataset="hemibrain:v1.2.1", token=token)

    sources = resolve_body_ids(VISUAL_SOURCE_TYPES, MAX_SOURCES_PER_TYPE, client)
    targets = resolve_body_ids(MOTOR_TARGET_TYPES, MAX_SOURCES_PER_TYPE, client)
    print(f"Resolved {len(sources)} visual source neurons, {len(targets)} motor targets.")

    print(f"Searching for paths (<= {MAX_HOPS} hops, min_weight={MIN_PATH_WEIGHT})"
          f" across {len(sources) * len(targets)} source/target pairs — this can take a while...")
    all_paths = []
    for src in sources:
        for dst in targets:
            df = fetch_paths(
                src,
                dst,
                min_weight=MIN_PATH_WEIGHT,
                max_path_length=MAX_HOPS,
                timeout=PATH_QUERY_TIMEOUT_S,
                client=client,
            )
            if df.empty:
                continue
            # fetch_paths' own `path` column restarts at 0 per call — offset
            # it so groupby stays correct once every pair's result is
            # concatenated below.
            df["path"] += sum(len(p["path"].unique()) for p in all_paths)
            all_paths.append(df)
            print(f"  {src} -> {dst}: {df['path'].nunique()} path(s)")

    if not all_paths:
        raise SystemExit(
            "No path found for any source/target pair within the current "
            "MAX_HOPS/MIN_PATH_WEIGHT/MAX_SOURCES_PER_TYPE — widen them and retry."
        )

    import pandas as pd
    paths_df = pd.concat(all_paths, ignore_index=True)

    path_body_ids = list(dict.fromkeys(int(b) for b in paths_df["bodyId"]))  # de-duped, first-seen order
    print(f"Path search touched {len(path_body_ids)} distinct neurons.")

    print("Fetching full adjacency among those neurons...")
    criteria = NC(bodyId=path_body_ids)
    _neuron_df, conn_df = fetch_adjacencies(sources=criteria, targets=criteria, client=client)

    conn_df = conn_df.rename(
        columns={"bodyId_pre": "pre_root_id", "bodyId_post": "post_root_id"}
    )[["pre_root_id", "post_root_id", "weight"]]
    conn_df.to_csv(OUT_DIR / "connectome.csv", index=False)
    print(f"Wrote {OUT_DIR / 'connectome.csv'} "
          f"({len(path_body_ids)} neurons, {len(conn_df)} synapses).")

    visual_pool, motor_pool = split_by_distance(paths_df, path_body_ids)
    roles_draft = {
        "_comment": (
            "DRAFT — not a finished roles.json. visual_pool/motor_pool are "
            "just distance-to-source-ranked bodyId lists; you still need to "
            "split visual_pool across proximity/x/y/keybind input channels "
            "and motor_pool across cursor_x/cursor_y/attack_gate (this "
            "pathway has no natural trail_gate/keybind_groups analog — see "
            "this script's header comment)."
        ),
        "visual_pool": visual_pool,
        "motor_pool": motor_pool,
    }
    with open(OUT_DIR / "roles.draft.json", "w", encoding="utf-8") as f:
        json.dump(roles_draft, f, indent=2)
    print(f"Wrote {OUT_DIR / 'roles.draft.json'} "
          f"({len(visual_pool)} visual-side, {len(motor_pool)} motor-side).")


def resolve_body_ids(types: list[str], max_per_type: int, client: Client) -> list[int]:
    ids: list[int] = []
    for t in types:
        neuron_df, _ = fetch_neurons(NC(type=t), client=client)
        ids.extend(int(b) for b in neuron_df["bodyId"].tolist()[:max_per_type])
    return ids


def split_by_distance(paths_df, path_body_ids):
    """Majority vote across every (path, position) occurrence of a neuron:
    within THAT SAME path instance, is it closer to the path's start (a
    visual source) or its end (Giant Fiber)? Comparing d_start/d_end within
    one occurrence (rather than each's independent min across all
    occurrences, which mixes unrelated paths of different lengths and badly
    skews long multi-hop searches toward motor_pool) is what makes this a
    fair per-occurrence comparison. Ties go to motor_pool."""
    visual_votes: dict[int, int] = {}
    total_votes: dict[int, int] = {}
    for _path_id, group in paths_df.groupby("path"):
        body_ids = [int(b) for b in group["bodyId"].tolist()]
        n = len(body_ids)
        for i, body_id in enumerate(body_ids):
            d_start, d_end = i, n - 1 - i
            total_votes[body_id] = total_votes.get(body_id, 0) + 1
            if d_start < d_end:
                visual_votes[body_id] = visual_votes.get(body_id, 0) + 1

    visual_pool, motor_pool = [], []
    for body_id in path_body_ids:
        votes, total = visual_votes.get(body_id, 0), total_votes.get(body_id, 1)
        (visual_pool if votes > total / 2 else motor_pool).append(body_id)
    return visual_pool, motor_pool


if __name__ == "__main__":
    main()
