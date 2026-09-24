"""Trains ChartPolicyNet (dl_model.py) with ordinary backprop — supervised,
not RL: cursor position regresses against target_xy() (proximity-weighted
nearest-due object), action is a single binary "act now" label placed at
each note's exact hit time (mouse or keyboard alike — WHICH action is read
off the targeted object's own features at decode time, not classified; see
dl_model.py / cursor_readout.py's target_info() and TRAIN_DIARY.md
2026-09-24 "no output patching"). See TRAIN_DIARY.md 2026-09-24.

Usage:
    python train_dl.py --frames ../output/test.frames.csv --events ../output/test.events.json \
        --epochs 300 --save dl_policy.pt
"""

import argparse

import torch
import torch.nn as nn

import config
from cursor_readout import SmoothedCursor, target_info, target_xy
from data import ChartData
from dl_model import ChartPolicyNet
from reward import Judge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", required=True)
    p.add_argument("--events", required=True)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--attack-tolerance-steps", type=int, default=1,
                    help="label this many steps on either side of a note's exact hit "
                         "step as a positive 'act now' example too (label smoothing)")
    p.add_argument("--attack-threshold", type=float, default=0.5,
                    help="predicted probability above this triggers the action at inference")
    p.add_argument("--refractory-ms", type=float, default=140.0)
    p.add_argument("--save", default="dl_policy.pt")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def build_labels(chart: ChartData, tolerance_steps: int):
    """Y_cursor: [T,2] (masked where nothing's active — mask returned too).
    Y_action: [T] binary, 1 within `tolerance_steps` of ANY note's (mouse OR
    keyboard) nearest step — one unified "act now" label. Splitting this by
    action type used to need a 68-way keybind classifier fed by a wildly
    imbalanced dataset per key (see TRAIN_DIARY.md 2026-09-24 "no output
    patching"); WHICH action fires is read off input features at decode
    time instead, so there's nothing left to split."""
    T = chart.num_steps
    y_cursor = torch.full((T, 2), 0.5)
    cursor_mask = torch.zeros(T, dtype=torch.bool)
    y_action = torch.zeros(T)

    for step in range(T):
        target = target_xy(chart.input_features_at(step))
        if target is not None:
            y_cursor[step] = torch.tensor(target)
            cursor_mask[step] = True

    t0 = float(chart.t_ms[0])
    for ev in chart.events:
        # Nearest step index to this note's exact hit time — t_ms is
        # uniformly spaced (config.DT_MS apart), so this is a direct O(1)
        # division instead of an O(T) linear scan.
        step_idx = min(T - 1, max(0, round((ev["time"] - t0) / config.DT_MS)))
        lo = max(0, step_idx - tolerance_steps)
        hi = min(T - 1, step_idx + tolerance_steps)
        y_action[lo:hi + 1] = 1.0

    return y_cursor, cursor_mask, y_action


def evaluate(model: ChartPolicyNet, chart: ChartData, attack_threshold: float, refractory_ms: float):
    """Real Judge, real radius — same honesty rule as train.py (2026-09-23
    #8/#9): never report a number measured under looser conditions than the
    real game uses."""
    model.eval()
    cursor_source = SmoothedCursor()
    judge = Judge(chart)
    refractory_steps = round(refractory_ms / config.DT_MS)
    refractory_left = 0

    with torch.no_grad():
        for step in range(chart.num_steps):
            features = chart.input_features_at(step)
            x = features.reshape(1, -1)
            cursor_pred, action_logit = model(x)
            cursor = cursor_source.step(tuple(cursor_pred[0].tolist()))

            attack_fired = False
            keybind_fired = set()
            if refractory_left > 0:
                refractory_left -= 1
            elif torch.sigmoid(action_logit).item() > attack_threshold:
                info = target_info(features)
                if info is not None:
                    if info["key"] is not None:
                        keybind_fired.add(info["key"])
                    else:
                        attack_fired = True
                    refractory_left = refractory_steps

            action = {
                "attack_fired": attack_fired, "trail_held": False,
                "keybind_fired": keybind_fired, "cursor": cursor, "output_spike_total": 0,
            }
            judge.step(float(chart.t_ms[step]), action)

    grades = {}
    for e in judge.log:
        grades[e["judgment"]] = grades.get(e["judgment"], 0) + 1
    hits = sum(grades.get(k, 0) for k in ("Perfect", "Good", "Bad"))
    return hits, grades


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    chart = ChartData(args.frames, args.events)
    print(f"[train_dl] loaded {chart.num_steps} steps, {len(chart.events)} notes, "
          f"{chart.max_objects} object slots")

    y_cursor, cursor_mask, y_action = build_labels(chart, args.attack_tolerance_steps)
    x_all = chart.frame_tensor.reshape(chart.num_steps, -1)

    pos = float(y_action.sum())
    neg = float(len(y_action) - pos)
    pos_weight = torch.tensor(neg / max(1.0, pos))
    print(f"[train_dl] action labels: {int(pos)} positive / {int(neg)} negative "
          f"(pos_weight={pos_weight.item():.1f})")

    model = ChartPolicyNet(chart.max_objects, chart.features_per_obj, hidden=args.hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    action_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_hits = -1
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        cursor_pred, action_logit = model(x_all)

        cursor_loss = ((cursor_pred - y_cursor) ** 2).sum(dim=1)
        cursor_loss = (cursor_loss * cursor_mask.float()).sum() / cursor_mask.float().sum().clamp(min=1)
        action_loss = action_loss_fn(action_logit, y_action)
        loss = cursor_loss + action_loss

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == args.epochs:
            hits, grades = evaluate(model, chart, args.attack_threshold, args.refractory_ms)
            grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
            print(f"[epoch {epoch:4d}] loss={loss.item():.4f} "
                  f"(cursor={cursor_loss.item():.4f} action={action_loss.item():.4f})  "
                  f"eval_hits={hits}/{len(chart.events)} ({grade_str})")
            if hits > best_hits:
                best_hits = hits
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"[train_dl] best eval hits: {best_hits}/{len(chart.events)}")

    if args.save:
        torch.save(
            {
                "state_dict": best_state,
                "max_objects": chart.max_objects,
                "features_per_obj": chart.features_per_obj,
                "hidden": args.hidden,
                "attack_threshold": args.attack_threshold,
                "refractory_ms": args.refractory_ms,
                "best_hits": best_hits,
                "total_notes": len(chart.events),
            },
            args.save,
        )
        print(f"[train_dl] saved best model -> {args.save}")


if __name__ == "__main__":
    main()
