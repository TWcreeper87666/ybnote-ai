"""Merges skeletons.json (fetch_skeletons.py) + spikes.json (export_spikes.py)
+ roles.json into the single bundle.json the 3D Artifact viewer loads.

Drops any neuron missing skeleton geometry (untraced bodies fetch_skeletons.py
skipped) — the viewer only shows neurons it can actually draw. Tags each
neuron's role (visual/hidden) from roles.json's "input_neurons" purely for
coloring. There's no "motor" tag anymore: output actions are decoded by the
separate trained ReadoutLayer (readout.py), not by any real neuron — see
TRAIN_DIARY.md's 2026-09-23 #2 entry — so no bodyId is a motor neuron here.
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent


def role_lookup(roles: dict) -> dict[int, str]:
    lookup: dict[int, str] = {}
    for ids in roles.get("input_neurons", {}).values():
        for bid in ids:
            lookup[bid] = "visual"
    return lookup


def main():
    skeletons = json.loads((OUT_DIR / "skeletons.json").read_text(encoding="utf-8"))
    spikes = json.loads((OUT_DIR / "spikes.json").read_text(encoding="utf-8"))
    roles_path = OUT_DIR / "roles.json"
    roles = json.loads(roles_path.read_text(encoding="utf-8")) if roles_path.exists() else {}
    role_by_body_id = role_lookup(roles)

    neuron_order = spikes["neuronOrder"]
    neurons = []
    # old index (into neuron_order/spikesPerStep) -> new index (into `neurons`,
    # skipping any bodyId with no skeleton) — spikesPerStep gets remapped
    # through this so the viewer never has to check "do I have geometry".
    old_to_new: dict[int, int] = {}
    for old_idx, body_id in enumerate(neuron_order):
        skel = skeletons.get(str(body_id))
        if skel is None:
            continue
        old_to_new[old_idx] = len(neurons)
        neurons.append({
            "bodyId": body_id,
            "points": skel["points"],
            "parents": skel["parents"],
            "role": role_by_body_id.get(body_id, "hidden"),
        })

    remapped_spikes = [
        [old_to_new[i] for i in step if i in old_to_new]
        for step in spikes["spikesPerStep"]
    ]

    bundle = {
        "dtMs": spikes["dtMs"],
        "neurons": neurons,
        "spikesPerStep": remapped_spikes,
    }
    out_path = OUT_DIR / "bundle.json"
    out_path.write_text(json.dumps(bundle), encoding="utf-8")
    print(f"Wrote {out_path}: {len(neurons)} neurons "
          f"(dropped {len(neuron_order) - len(neurons)} without skeleton geometry), "
          f"{len(remapped_spikes)} steps")


if __name__ == "__main__":
    main()
