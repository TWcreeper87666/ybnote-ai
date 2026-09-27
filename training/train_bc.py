"""Behavior cloning with DAgger: train the RL ActorNet to imitate
bc_expert.ScriptedExpert inside TrailRLEnv.

PPO from scratch plateaued around 37% macro validation accuracy while the
scripted demonstrator, reading the SAME observation under the SAME world
speed ceiling, scores ~99.7% (TRAIN_DIARY.md 2026-09-26 "behavior
cloning"). So the bottleneck is exploration/credit assignment, not missing
information. DAgger (Ross et al. 2011): roll out a mix of expert and
student, label every visited observation with the expert's action, and fit
the student on the aggregated dataset — the student learns to recover from
its own mistakes, not only to follow the expert's trajectory.

The output checkpoint has train_rl.py's format (actor + critic), so it can
be evaluated, exported (export_replay --policy rl) or fine-tuned with PPO
(train_rl.py --resume-from).

Usage:
    python training/train_bc.py --charts-dir output --init training/rl_policy_v7_groupfix.pt \
        --save training/rl_policy_bc.pt
"""

from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

import config
import rl_env
from bc_expert import ScriptedExpert
from data import ChartData
from rl_env import NUM_PRESS, PRESS_KEY, PRESS_NONE, TrailRLEnv, decode_action, obs_features_per_obj
from rl_policy import ACTION_SPACE, CURSOR_COMPONENT_LIMIT, ActorNet, CriticNet, migrate_pre_split_checkpoint
from train_rl import BALANCED_VALIDATION_CHARTS, evaluate_holdout, find_chart_pairs, make_env


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--charts-dir", required=True)
    p.add_argument("--init", default="", help="train_rl.py checkpoint to start from (actor+critic); empty = fresh")
    p.add_argument("--save", default="training/rl_policy_bc.pt")
    p.add_argument("--iterations", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--steps-per-env", type=int, default=512, help="collected per env per iteration")
    p.add_argument("--buffer-size", type=int, default=120_000)
    p.add_argument("--updates-per-iter", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-final", type=float, default=None,
                   help="cosine-anneal the lr to this by the last iteration (bc2's evals swung 79-92%% "
                        "between checkpoints at a constant lr); default: constant")
    p.add_argument("--beta-decay", type=float, default=0.8,
                   help="DAgger: probability the EXPERT drives a rollout step = beta_decay**(iteration-1)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--cursor-weight", type=float, default=50.0,
                   help="weight of the cursor MSE (targets are speed-ceiling fractions, |x| <= 0.71)")
    p.add_argument("--timing-feature", action="store_true",
                   help="fill the hit_timing_at() column (zeroed by default: it hurt PPO, v8b/v8c)")
    p.add_argument("--attack-clock", action="store_true",
                   help="show the policy its own ticks_since_attack (off by default: causal confusion, see rl_env)")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


class Buffer:
    """FIFO of (obs float16, cursor target, press class, toggle target)."""

    def __init__(self, size: int, obs_dim: int):
        self.obs = np.zeros((size, obs_dim), dtype=np.float16)
        self.cursor = np.zeros((size, 2), dtype=np.float32)
        self.press = np.zeros(size, dtype=np.int64)
        self.size, self.n, self.pos = size, 0, 0

    def add(self, obs, cursor, press):
        self.obs[self.pos] = obs
        self.cursor[self.pos] = cursor
        self.press[self.pos] = press
        self.pos = (self.pos + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch: int):
        idx = np.random.randint(0, self.n, size=batch)
        return (
            torch.from_numpy(self.obs[idx].astype(np.float32)),
            torch.from_numpy(self.cursor[idx]),
            torch.from_numpy(self.press[idx]),
        )


def bc_loss(actor: ActorNet, obs, cursor_target, press_target, cursor_weight: float):
    """Cursor: MSE of the deterministic action (tanh(mean) * limit) to the
    expert delta. Press: when-head CE (press vs none) plus how-head BCE on
    press samples only (expert always clicks). Trail: toggle BCE toward 0
    (the expert never trails)."""
    out = actor.forward(obs)
    cursor_pred = torch.tanh(out["cursor_loc"]) * CURSOR_COMPONENT_LIMIT
    cursor_loss = F.mse_loss(cursor_pred, cursor_target)
    is_press = (press_target != PRESS_NONE).long()
    when_loss = F.cross_entropy(out["action_logits"], is_press)
    press_mask = is_press.bool()
    if press_mask.any():
        how_loss = F.binary_cross_entropy_with_logits(
            out["input_path_logit"][press_mask], (press_target[press_mask] == PRESS_KEY).float()
        )
    else:
        how_loss = cursor_loss.new_zeros(())
    toggle_loss = F.binary_cross_entropy_with_logits(
        out["trail_toggle_logit"], torch.zeros_like(out["trail_toggle_logit"])
    )
    total = cursor_weight * cursor_loss + when_loss + how_loss + 0.1 * toggle_loss
    return total, {"cursor": float(cursor_loss), "when": float(when_loss), "how": float(how_loss)}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rl_env.TIMING_FEATURE_ENABLED = args.timing_feature
    rl_env.ATTACK_CLOCK_FEATURE_ENABLED = args.attack_clock

    pairs = find_chart_pairs(args.charts_dir)
    by_name = {os.path.basename(fp)[: -len(".frames.csv")]: (fp, ep) for fp, ep in pairs}
    holdout = [ChartData(*by_name[n]) for n in BALANCED_VALIDATION_CHARTS]
    train = [ChartData(fp, ep) for name, (fp, ep) in by_name.items() if name not in BALANCED_VALIDATION_CHARTS]
    print(f"[bc] {len(train)} train charts, {len(holdout)} validation charts")

    max_objects = train[0].max_objects
    fpo = obs_features_per_obj(train[0].features_per_obj)
    actor = ActorNet(max_objects, fpo)
    critic = CriticNet(max_objects, fpo)
    if args.init:
        ckpt = migrate_pre_split_checkpoint(torch.load(args.init, map_location="cpu", weights_only=False))
        actor.load_state_dict(ckpt["actor_state_dict"])
        critic.load_state_dict(ckpt["critic_state_dict"])
        print(f"[bc] initialized from {args.init}")
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    expert = ScriptedExpert()

    envs, env_obs = [], []
    for _ in range(args.num_envs):
        envs.append(make_env(train[rng.randrange(len(train))], True, rng, 0, 0, True))
        env_obs.append(envs[-1].reset())
    buffer = Buffer(args.buffer_size, envs[0]._per_step_dim * len(rl_env.HISTORY_STRIDES))

    best, since_best = float("-inf"), 0
    started = time.time()
    for it in range(1, args.iterations + 1):
        beta = args.beta_decay ** (it - 1)
        student_steps = 0
        for i in range(len(envs)):
            for _ in range(args.steps_per_env):
                env, obs = envs[i], env_obs[i]  # envs[i] is replaced when an episode ends
                target_delta, target_action = expert.act(env, obs)
                press, _ = decode_action(target_action)
                buffer.add(
                    obs,
                    np.clip(target_delta, -CURSOR_COMPONENT_LIMIT + 1e-4, CURSOR_COMPONENT_LIMIT - 1e-4),
                    press,
                )
                if rng.random() < beta:
                    delta, action = target_delta, target_action
                else:
                    with torch.no_grad():
                        act = actor.act(torch.from_numpy(obs).float(), deterministic=True)
                    delta, action = act["cursor_delta"], act["action_type"]
                    student_steps += 1
                env_obs[i], *_ = env.step(delta, action)
                if env.done:
                    envs[i] = make_env(train[rng.randrange(len(train))], True, rng, 0, 0, True)
                    env_obs[i] = envs[i].reset()

        if args.lr_final is not None:
            progress = (it - 1) / max(1, args.iterations - 1)
            lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
        stats = {"cursor": 0.0, "when": 0.0, "how": 0.0}
        for _ in range(args.updates_per_iter):
            obs_b, cur_b, press_b = buffer.sample(args.batch_size)
            loss, parts = bc_loss(actor, obs_b, cur_b, press_b, args.cursor_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            for k in stats:
                stats[k] += parts[k] / args.updates_per_iter
        press_rate = float(np.mean(buffer.press[: buffer.n] != PRESS_NONE))
        print(
            f"[bc] iter {it} beta={beta:.3f} student_steps={student_steps} buffer={buffer.n} "
            f"lr={optimizer.param_groups[0]['lr']:.1e} press_rate={press_rate:.4f} cursor_mse={stats['cursor']:.5f} when_ce={stats['when']:.4f} "
            f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

        if it % args.eval_every == 0 or it == args.iterations:
            hits, notes, grades, macro = evaluate_holdout(actor, holdout)
            print(f"[bc eval {it}] hits={hits}/{notes} macro={macro:.2f}% {grades}", flush=True)
            ckpt = {
                "actor_state_dict": actor.state_dict(),
                "critic_state_dict": critic.state_dict(),
                "max_objects": max_objects,
                "features_per_obj": fpo,
                "action_space": ACTION_SPACE,
                "obs_flags": rl_env.current_obs_flags(),
                "hidden": 256,
                "best_holdout_hits": hits,
                "best_holdout_score": macro,
                "best_holdout_metric": "macro_chart_weighted_accuracy_pct",
                "best_holdout_grades": grades,
                "iteration": it,
                "trained_by": "train_bc.py (DAgger on bc_expert.ScriptedExpert)",
            }
            # Also keep the latest eval: once validation saturates (bc7 held
            # 99.81% from iteration 10 on), the strict-best file would stay
            # at the earliest, least-trained checkpoint.
            torch.save(ckpt, args.save[: -len(".pt")] + "_last.pt")
            if macro > best:
                best, since_best = macro, 0
                torch.save(ckpt, args.save)
                print(f"[bc] new best macro={macro:.2f}% -> {args.save}", flush=True)
            else:
                since_best += 1
                if since_best >= args.early_stop_patience:
                    print(f"[bc] early stopping at iteration {it}; best macro={best:.2f}%")
                    break
    print(f"[bc] done. best macro={best:.2f}%")


if __name__ == "__main__":
    main()
