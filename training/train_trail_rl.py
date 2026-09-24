import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
"""Trains ONLY trail_head with policy-gradient reinforcement learning
(REINFORCE, a per-step contextual bandit — Judge's reward is already dense/
immediate per step, not a delayed episode return, so there's no temporal
credit-assignment problem to solve) against the REAL Judge reward, instead
of imitating a hand-designed "when should trail be held" label.

Why: train_dl_multi.py's y_trail label (see pathing.py) went through three
supervised-imitation attempts — "trail whenever heading toward a mouse
note", "trail only on chart-level needs_routing charts", "trail only on a
per-step geometrically-safe segment" — and the user correctly called out
that all three were still ME writing a rule for the model to imitate, not
the model deciding for itself from outcomes. This is the real answer:
trail_head samples a Bernoulli(sigmoid(trail_logit)) action stochastically,
Judge scores it (Wrong for touching a not-due collidable while trailing,
positive reward for a resulting hit, nothing otherwise), and the policy
gradient step increases the log-probability of actions that beat the
rollout's average reward and decreases it for ones that fall short.

Scope: ONLY trail_head's own weights update. cursor_head/action_head/trunk
stay exactly as train_dl_multi.py trained them (supervised — that part
already works and isn't in question here); trail_head reads the SAME
frozen trunk features but through a detached (no-grad) tensor, so no
gradient from this script's loss reaches the trunk. This is a deliberately
bounded RL problem (one Bernoulli decision per step) specifically because
the project's one earlier full-network reward-learning attempt (R-STDP
over a spiking reservoir, see TRAIN_DIARY.md 2026-09-24 "全面轉向 DL") kept
collapsing to total silence — a small policy-gradient head on top of an
already-stable supervised backbone doesn't carry that failure mode.

Usage:
    python train_trail_rl.py --charts-dir ../output --weights dl_policy_multi.pt \
        --holdout 5 --epochs 60 --save dl_policy_multi.pt
"""

import argparse
import glob
import os
import random

import torch

import config
from cursor_readout import SmoothedCursor, target_info
from data import ChartData
from dl_model import ChartPolicyNet
from obstacles import nearby_obstacle_features
from reward import Judge
from train_dl_multi import find_chart_pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--charts-dir", required=True)
    p.add_argument("--weights", required=True, help="train_dl_multi.py --save checkpoint to start from")
    p.add_argument("--holdout", type=int, default=5)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--save", default="dl_policy_multi.pt")
    return p.parse_args()


def rollout(model: ChartPolicyNet, chart: ChartData, attack_threshold: float, refractory_ms: float, stochastic: bool):
    """One pass over `chart`. trail_head's Bernoulli(sigmoid(trail_logit))
    is SAMPLED when `stochastic` (training — REINFORCE needs an actual
    random action and its log-probability), else taken as the deterministic
    >0.5 policy an eval/inference run would use. cursor_head/action_head
    are always deterministic (frozen, already supervised-trained — nothing
    here changes them). Returns (hits, grades, log_probs, rewards);
    log_probs/rewards are only meaningful when `stochastic`."""
    cursor_source = SmoothedCursor()
    judge = Judge(chart)
    refractory_steps = round(refractory_ms / config.DT_MS)
    refractory_left = 0
    log_probs = []
    rewards = []

    for step in range(chart.num_steps):
        features = chart.input_features_at(step)
        obstacle_feats = nearby_obstacle_features(
            cursor_source.pos, chart.collidable_centers, chart.collidable_halves
        ).reshape(1, -1)
        x = torch.cat([features.reshape(1, -1), obstacle_feats], dim=1)

        with torch.no_grad():
            h = model.trunk(x)
            cursor_pred = torch.sigmoid(model.cursor_head(h))
            object_features = x[..., : model.object_dim]
            action_logit = model.action_head(model.action_trunk(object_features)).squeeze(-1)
        # trail_head is the only part of the forward pass that needs a
        # live graph (its own weights are what this script trains) — h is
        # detached first so no gradient reaches the (frozen) trunk.
        trail_logit = model.trail_head(h.detach()).squeeze(-1)
        trail_prob = torch.sigmoid(trail_logit)

        if stochastic:
            trail_action = torch.bernoulli(trail_prob)
            eps = 1e-6
            log_prob = trail_action * torch.log(trail_prob + eps) + (1 - trail_action) * torch.log(1 - trail_prob + eps)
            log_probs.append(log_prob)
            trail_held = bool(trail_action.item())
        else:
            trail_held = trail_prob.item() > 0.5

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
            "attack_fired": attack_fired, "trail_held": trail_held,
            "keybind_fired": keybind_fired, "cursor": cursor, "output_spike_total": 0,
        }
        step_reward = judge.step(float(chart.t_ms[step]), action)
        if stochastic:
            rewards.append(step_reward)

    grades = {}
    for e in judge.log:
        grades[e["judgment"]] = grades.get(e["judgment"], 0) + 1
    hits = sum(grades.get(k, 0) for k in ("Perfect", "Good", "Bad"))
    return hits, grades, log_probs, rewards


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = find_chart_pairs(args.charts_dir)
    rng = random.Random(args.seed)
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)
    holdout_pairs = pairs_shuffled[: args.holdout]
    train_pairs = pairs_shuffled[args.holdout :]
    print(f"[train_trail_rl] {len(pairs)} charts: {len(train_pairs)} train, {len(holdout_pairs)} held out")

    blob = torch.load(args.weights, weights_only=False)
    model = ChartPolicyNet(
        blob["max_objects"], blob["features_per_obj"], hidden=blob["hidden"], obstacle_slots=blob["obstacle_slots"]
    )
    model.load_state_dict(blob["state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    for p in model.trail_head.parameters():
        p.requires_grad = True

    attack_threshold = blob["attack_threshold"]
    refractory_ms = blob["refractory_ms"]

    optimizer = torch.optim.Adam(model.trail_head.parameters(), lr=args.lr)

    train_charts = [ChartData(fp, ep) for fp, ep in train_pairs]
    holdout_charts = [ChartData(fp, ep) for fp, ep in holdout_pairs]

    def evaluate(charts):
        total_hits = total_notes = total_wrong = 0
        for chart in charts:
            hits, grades, _, _ = rollout(model, chart, attack_threshold, refractory_ms, stochastic=False)
            total_hits += hits
            total_notes += len(chart.events)
            total_wrong += grades.get("Wrong", 0)
        return total_hits, total_notes, total_wrong

    hits0, notes0, wrong0 = evaluate(holdout_charts)
    print(f"[train_trail_rl] before RL: HOLDOUT hits={hits0}/{notes0} wrong={wrong0}")

    best_hits, best_wrong = hits0, wrong0
    best_state = {k: v.clone() for k, v in model.trail_head.state_dict().items()}

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(train_charts)
        epoch_reward_sum = 0.0
        epoch_steps = 0
        for chart in train_charts:
            _, _, log_probs, rewards = rollout(model, chart, attack_threshold, refractory_ms, stochastic=True)
            if not log_probs:
                continue
            rewards_t = torch.tensor(rewards)
            baseline = rewards_t.mean()
            advantage = rewards_t - baseline
            std = advantage.std()
            if std > 1e-6:
                advantage = advantage / std
            loss = -(torch.cat(log_probs) * advantage).sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_reward_sum += float(rewards_t.sum())
            epoch_steps += len(rewards)

        if epoch % 5 == 0 or epoch == args.epochs:
            hits, notes, wrong = evaluate(holdout_charts)
            pct = 100 * hits / max(1, notes)
            print(f"[epoch {epoch:4d}] train_mean_reward={epoch_reward_sum / max(1, epoch_steps):.5f}  "
                  f"HOLDOUT hits={hits}/{notes} ({pct:.1f}%) wrong={wrong}")
            # Prefer fewer Wrong at similar hit count — that's the entire
            # point of this script (trail's real cost is Wrong exposure),
            # not just raw hits.
            if hits >= best_hits - 5 and wrong < best_wrong:
                best_hits, best_wrong = hits, wrong
                best_state = {k: v.clone() for k, v in model.trail_head.state_dict().items()}
            elif hits > best_hits:
                best_hits, best_wrong = hits, wrong
                best_state = {k: v.clone() for k, v in model.trail_head.state_dict().items()}

    model.trail_head.load_state_dict(best_state)
    print(f"[train_trail_rl] best: hits={best_hits} wrong={best_wrong}")
    print("[train_trail_rl] per-chart holdout breakdown:")
    for chart, (fp, _) in zip(holdout_charts, holdout_pairs):
        hits, grades, _, _ = rollout(model, chart, attack_threshold, refractory_ms, stochastic=False)
        grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
        print(f"  {os.path.basename(fp)}: {hits}/{len(chart.events)} ({grade_str})")

    blob["state_dict"] = model.state_dict()
    blob["best_holdout_hits"] = best_hits
    torch.save(blob, args.save)
    print(f"[train_trail_rl] saved -> {args.save}")


if __name__ == "__main__":
    main()
