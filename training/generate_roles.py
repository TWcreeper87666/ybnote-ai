"""Turns roles.draft.json (fetch_real_connectome.py's output — just a
visual/motor bodyId split) into a real roles.json train.py can load.

Purely mechanical slicing, no neuroscience judgment involved: takes
consecutive chunks of visual_pool for the four input channels and of
motor_pool for the output populations. visual_pool only has ~35 neurons (a
handful of hops from LC4/LPLC2), so input channels get small groups;
motor_pool has ~750 (most of the network sits "closer to" the single Giant
Fiber sink — see fetch_real_connectome.py's split_by_distance), so output
groups can be generous. trail_gate and the keybind groups have no biological
counterpart in this pathway (see ENCODING_DESIGN.md's known-simplifications
section) — they're just handed arbitrary motor_pool neurons and have to
learn their role from scratch during training, same as attack_gate would if
this weren't a real connectome at all.
"""

import json
from pathlib import Path

OUT_DIR = Path(__file__).parent

# How many neurons each role gets — tune freely, just keep sums <= pool sizes.
# x/y weighted heavily (2026-09-24): they're now a population code (reward.py's
# inject_population — see TRAIN_DIARY.md's "one unified model" entry), so
# more neurons = finer spatial resolution, directly limiting how accurately
# CursorReadout can ever reconstruct position. proximity/keybind are plain
# scalar broadcasts, which don't benefit from more neurons the same way — a
# handful is enough. visual_pool has 35 real neurons total; this uses all of
# them.
INPUT_SIZES = {"proximity": 3, "x": 15, "y": 14, "keybind": 3}
OUTPUT_SIZES = {
    "cursor_x": 10,
    "cursor_y": 10,
    "attack_gate": 10,
    "trail_gate": 10,
}
KEYBIND_GROUP_SIZES = {"a": 5, "k": 5}


def take(pool: list[int], sizes: dict[str, int]) -> dict[str, list[int]]:
    result = {}
    i = 0
    for name, n in sizes.items():
        chunk = pool[i : i + n]
        if len(chunk) < n:
            raise SystemExit(
                f"Pool exhausted allocating '{name}': needed {n}, only "
                f"{len(chunk)} left. Shrink the *_SIZES dicts or fetch a "
                f"bigger connectome."
            )
        result[name] = chunk
        i += n
    return result, i


def main():
    draft = json.loads((OUT_DIR / "roles.draft.json").read_text(encoding="utf-8"))
    visual_pool, motor_pool = draft["visual_pool"], draft["motor_pool"]

    input_roles, _ = take(visual_pool, INPUT_SIZES)

    output_roles, used = take(motor_pool, OUTPUT_SIZES)
    keybind_groups, _ = take(motor_pool[used:], KEYBIND_GROUP_SIZES)
    output_roles["keybind_groups"] = keybind_groups

    roles = {"input_neurons": input_roles, "output_neurons": output_roles}
    out_path = OUT_DIR / "roles.json"
    out_path.write_text(json.dumps(roles, indent=2), encoding="utf-8")

    print(f"Wrote {out_path}")
    for name, ids in input_roles.items():
        print(f"  input.{name}: {len(ids)} neurons")
    for name, ids in output_roles.items():
        if name == "keybind_groups":
            for k, kids in ids.items():
                print(f"  output.keybind_groups.{k}: {len(kids)} neurons")
        else:
            print(f"  output.{name}: {len(ids)} neurons")


if __name__ == "__main__":
    main()
