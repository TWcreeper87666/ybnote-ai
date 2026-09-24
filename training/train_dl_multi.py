import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
"""Trains ChartPolicyNet across MANY charts (not just one), holding a few
out entirely for evaluation — the actual test of "does this generalize to a
chart it's never seen" (2026-09-24: single-chart training memorizes that
one song's specific rhythm; this is the real fix).

Usage:
    python train_dl_multi.py --charts-dir ../output --holdout 5 --epochs 60 \
        --save dl_policy_multi.pt
"""

import argparse
import glob
import os
import random

import torch
import torch.nn as nn

import config
from augment import MODES, augment_batch
from cursor_readout import SmoothedCursor, target_info
from data import ChartData
from dl_model import ChartPolicyNet
from obstacles import MAX_OBSTACLES, nearby_obstacle_features
from pathing import build_cursor_and_obstacle_labels
from reward import Judge


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--charts-dir", required=True, help="directory of *.frames.csv / *.events.json pairs")
    p.add_argument("--holdout", type=int, default=5, help="how many charts to hold out entirely for eval")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--attack-tolerance-steps", type=int, default=10)
    p.add_argument("--attack-threshold", type=float, default=0.5)
    p.add_argument("--trail-threshold", type=float, default=0.5)
    p.add_argument("--refractory-ms", type=float, default=260.0)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="random D4 rotate/mirror per training batch (see augment.py)")
    p.add_argument("--save", default="dl_policy_multi.pt")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def find_chart_pairs(charts_dir: str):
    pairs = []
    for frames_path in sorted(glob.glob(os.path.join(charts_dir, "*.frames.csv"))):
        base = frames_path[: -len(".frames.csv")]
        events_path = base + ".events.json"
        if not os.path.exists(events_path):
            continue
        if os.path.getsize(frames_path) == 0:
            print(f"  skip (empty frames.csv): {os.path.basename(frames_path)}")
            continue
        with open(frames_path, "r", encoding="utf-8") as f:
            if len(f.readlines()) < 2:  # header only, no data rows
                print(f"  skip (no frame rows): {os.path.basename(frames_path)}")
                continue
        pairs.append((frames_path, events_path))
    return pairs


def build_labels(chart: ChartData, tolerance_steps: int):
    """y_action: [T] binary, 1 within `tolerance_steps` of ANY note's (mouse
    OR keyboard) nearest step — a single unified "act now" label, edge-
    triggered timing for precise click grading. y_cursor/obstacle_feats/
    y_trail come from pathing.py (obstacle-routed, not a plain straight
    line — see that module's docstring)."""
    T = chart.num_steps
    y_cursor, cursor_mask, obstacle_feats, y_trail = build_cursor_and_obstacle_labels(
        chart, config.CURSOR_MAX_SPEED_NORM_PER_STEP
    )
    y_action = torch.zeros(T)

    t0 = float(chart.t_ms[0])
    for ev in chart.events:
        # t_ms is uniformly spaced (config.DT_MS apart) and sorted, so the
        # nearest step is a direct O(1) division — the O(T) linear min() this
        # replaced was a real bottleneck once training scaled to 26 charts'
        # worth of events (see TRAIN_DIARY.md 2026-09-24 "keybind support").
        step_idx = min(T - 1, max(0, round((ev["time"] - t0) / config.DT_MS)))
        lo = max(0, step_idx - tolerance_steps)
        hi = min(T - 1, step_idx + tolerance_steps)
        y_action[lo:hi + 1] = 1.0

    return y_cursor, cursor_mask, obstacle_feats, y_trail, y_action


def evaluate_chart(model: ChartPolicyNet, chart: ChartData, attack_threshold: float,
                    trail_threshold: float, refractory_ms: float):
    model.eval()
    cursor_source = SmoothedCursor()
    judge = Judge(chart)
    refractory_steps = round(refractory_ms / config.DT_MS)
    refractory_left = 0

    with torch.no_grad():
        for step in range(chart.num_steps):
            features = chart.input_features_at(step)
            # Obstacle features relative to where the cursor CURRENTLY is
            # (before this step's move) — same relationship the training
            # labels use (pathing.py computes them from the pursuer's
            # position at the START of each step). See obstacles.py.
            obstacle_feats = nearby_obstacle_features(
                cursor_source.pos, chart.collidable_centers, chart.collidable_halves
            ).reshape(1, -1)
            x = torch.cat([features.reshape(1, -1), obstacle_feats], dim=1)
            cursor_pred, action_logit, trail_logit = model(x)
            cursor = cursor_source.step(tuple(cursor_pred[0].tolist()))

            attack_fired = False
            keybind_fired = set()
            if refractory_left > 0:
                refractory_left -= 1
            elif torch.sigmoid(action_logit).item() > attack_threshold:
                # WHICH action (click vs. which key) is read off the
                # currently-targeted object's own features, not classified —
                # see dl_model.py / cursor_readout.py's target_info().
                info = target_info(features)
                if info is not None:
                    if info["key"] is not None:
                        keybind_fired.add(info["key"])
                    else:
                        attack_fired = True
                    refractory_left = refractory_steps

            # Level-triggered, no refractory — "is trail down right now",
            # independent of the edge-triggered click/key decision above.
            trail_held = torch.sigmoid(trail_logit).item() > trail_threshold

            action = {
                "attack_fired": attack_fired, "trail_held": trail_held,
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
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = find_chart_pairs(args.charts_dir)
    if len(pairs) <= args.holdout:
        raise SystemExit(f"Only {len(pairs)} charts found, need more than --holdout ({args.holdout}).")

    rng = random.Random(args.seed)
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)
    holdout_pairs = pairs_shuffled[: args.holdout]
    train_pairs = pairs_shuffled[args.holdout :]

    print(f"[train_dl_multi] {len(pairs)} charts total: {len(train_pairs)} train, {len(holdout_pairs)} held out")
    for fp, _ in holdout_pairs:
        print(f"  held out: {os.path.basename(fp)}")

    print("[train_dl_multi] loading + labeling train charts...")
    train_charts = []
    xobj_chunks, xobs_chunks, yc_chunks, mask_chunks, ya_chunks, yt_chunks = [], [], [], [], [], []
    max_objects = None
    features_per_obj = None
    for frames_path, events_path in train_pairs:
        chart = ChartData(frames_path, events_path)
        if max_objects is None:
            max_objects = chart.max_objects
            features_per_obj = chart.features_per_obj
        elif chart.max_objects != max_objects or chart.features_per_obj != features_per_obj:
            print(f"  skip (feature shape mismatch): {frames_path}")
            continue
        y_cursor, cursor_mask, obstacle_feats, y_trail, y_action = build_labels(chart, args.attack_tolerance_steps)
        xobj_chunks.append(chart.frame_tensor.reshape(chart.num_steps, -1))
        xobs_chunks.append(obstacle_feats)
        yc_chunks.append(y_cursor)
        mask_chunks.append(cursor_mask)
        ya_chunks.append(y_action)
        yt_chunks.append(y_trail)
        train_charts.append(chart)
        n_collidables = len(chart.collidables)
        print(f"  {os.path.basename(frames_path)}: {chart.num_steps} steps, {len(chart.events)} notes, "
              f"{n_collidables} collidables")

    xobj_all = torch.cat(xobj_chunks, dim=0)
    xobs_all = torch.cat(xobs_chunks, dim=0)
    yc_all = torch.cat(yc_chunks, dim=0)
    mask_all = torch.cat(mask_chunks, dim=0)
    ya_all = torch.cat(ya_chunks, dim=0)
    yt_all = torch.cat(yt_chunks, dim=0)
    n_samples = xobj_all.shape[0]
    print(f"[train_dl_multi] combined training set: {n_samples} steps across {len(train_charts)} charts")

    holdout_charts = [ChartData(fp, ep) for fp, ep in holdout_pairs]

    pos = float(ya_all.sum())
    neg = float(n_samples - pos)
    action_pos_weight = torch.tensor(neg / max(1.0, pos))
    trail_pos = float(yt_all.sum())
    trail_neg = float(n_samples - trail_pos)
    trail_pos_weight = torch.tensor(trail_neg / max(1.0, trail_pos))
    print(f"[train_dl_multi] action labels: {int(pos)} positive / {int(neg)} negative "
          f"(pos_weight={action_pos_weight.item():.1f})")
    print(f"[train_dl_multi] trail labels: {int(trail_pos)} positive / {int(trail_neg)} negative "
          f"(pos_weight={trail_pos_weight.item():.1f})")

    model = ChartPolicyNet(max_objects, features_per_obj, hidden=args.hidden, obstacle_slots=MAX_OBSTACLES)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    action_loss_fn = nn.BCEWithLogitsLoss(pos_weight=action_pos_weight)
    trail_loss_fn = nn.BCEWithLogitsLoss(pos_weight=trail_pos_weight)

    best_holdout_hits = -1
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n_samples)
        total_loss = 0.0
        for start in range(0, n_samples, args.batch_size):
            idx = perm[start : start + args.batch_size]
            optimizer.zero_grad()

            xobj_b, xobs_b, yc_b = xobj_all[idx], xobs_all[idx], yc_all[idx]
            if args.augment:
                mode = MODES[random.randrange(len(MODES))]
                xobj_b, xobs_b, yc_b = augment_batch(xobj_b, xobs_b, yc_b, features_per_obj, mode)
            x_b = torch.cat([xobj_b, xobs_b], dim=1)

            cursor_pred, action_logit, trail_logit = model(x_b)

            cursor_loss = ((cursor_pred - yc_b) ** 2).sum(dim=1)
            m = mask_all[idx].float()
            cursor_loss = (cursor_loss * m).sum() / m.sum().clamp(min=1)
            action_loss = action_loss_fn(action_logit, ya_all[idx])
            trail_loss = trail_loss_fn(trail_logit, yt_all[idx])
            loss = cursor_loss + action_loss + trail_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(idx)

        avg_loss = total_loss / n_samples

        if epoch % 5 == 0 or epoch == args.epochs:
            holdout_total_hits = 0
            holdout_total_notes = 0
            for chart in holdout_charts:
                hits, _ = evaluate_chart(model, chart, args.attack_threshold, args.trail_threshold, args.refractory_ms)
                holdout_total_hits += hits
                holdout_total_notes += len(chart.events)
            pct = 100 * holdout_total_hits / max(1, holdout_total_notes)
            print(f"[epoch {epoch:4d}] avg_loss={avg_loss:.4f}  "
                  f"HOLDOUT hits={holdout_total_hits}/{holdout_total_notes} ({pct:.1f}%)")
            if holdout_total_hits > best_holdout_hits:
                best_holdout_hits = holdout_total_hits
                best_state = {k: v.clone() for k, v in model.state_dict().items()}

    print(f"[train_dl_multi] best holdout hits: {best_holdout_hits}")

    # Final per-chart breakdown on the holdout set, using the best checkpoint.
    model.load_state_dict(best_state)
    print("[train_dl_multi] per-chart holdout breakdown:")
    for chart, (fp, _) in zip(holdout_charts, holdout_pairs):
        hits, grades = evaluate_chart(model, chart, args.attack_threshold, args.trail_threshold, args.refractory_ms)
        grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
        print(f"  {os.path.basename(fp)}: {hits}/{len(chart.events)} ({grade_str})")

    if args.save:
        torch.save(
            {
                "state_dict": best_state,
                "max_objects": max_objects,
                "features_per_obj": features_per_obj,
                "hidden": args.hidden,
                "obstacle_slots": MAX_OBSTACLES,
                "attack_threshold": args.attack_threshold,
                "trail_threshold": args.trail_threshold,
                "refractory_ms": args.refractory_ms,
                "best_holdout_hits": best_holdout_hits,
                "num_train_charts": len(train_charts),
                "num_holdout_charts": len(holdout_charts),
            },
            args.save,
        )
        print(f"[train_dl_multi] saved best model -> {args.save}")


if __name__ == "__main__":
    main()
