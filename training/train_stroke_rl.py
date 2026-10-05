"""PPO fine-tuning of the cursor during held trail strokes, on the game's
own judgments.

Behavior cloning plateaued on stroke steering (TRAIN_DIARY.md 2026-09-29):
the student starts every planned stroke, but sweeps into walls because the
teacher's move depends on its private route and pacing (an imitation gap,
ADVISOR). Here the policy plays generated stroke levels itself and the
reward is only what the judge says — hits, Wrongs, Misses.

What RL may change: the cursor move on ticks where the trail is held. The
press / hold heads act greedily as trained, and a frozen copy of the
starting policy anchors them and the cursor on every other tick, so the
clicking the BC policy already does (validation 99.7%) is kept.

  python training/train_stroke_rl.py --init training/models/rl_policy_bc15.pt --save training/models/rl_policy_rl1.pt
"""

from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
import rl_env  # noqa: E402
from data import ChartData  # noqa: E402
from nav_map import unpack_views  # noqa: E402
from niceness import be_nice  # noqa: E402
from ppo import RunningNorm, compute_gae  # noqa: E402
from rl_env import TrailRLEnv, apply_obs_flags  # noqa: E402
from rl_policy import CURSOR_COMPONENT_LIMIT, ActorNet, CriticNet, migrate_pre_split_checkpoint  # noqa: E402
from train_rl import BALANCED_VALIDATION_CHARTS, evaluate_holdout, find_chart_pairs, make_env  # noqa: E402
from trail_plan import get_trail_plan  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="train_bc.py checkpoint (with the local view)")
    p.add_argument("--save", required=True)
    p.add_argument("--synth-dir", nargs="+", default=["data/output_synth"])
    p.add_argument("--eval-dir", default="data/output_synth_test2")
    p.add_argument("--charts-dir", default="data/output", help="validation charts (clicking must not regress)")
    p.add_argument("--updates", type=int, default=300)
    p.add_argument("--num-envs", type=int, default=6)
    p.add_argument("--steps-per-env", type=int, default=512)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--critic-lr", type=float, default=3e-4)
    p.add_argument("--critic-warmup", type=int, default=5, help="updates that fit only the critic")
    p.add_argument("--sigma", type=float, default=0.05,
                   help="exploration std of the pre-tanh cursor (x tanh'(0)*limit ~0.7 x 40 units: 0.05 ~ 1.4 units/tick)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--wrong-scale", type=float, default=4.0, help="Wrong reward multiplier (-0.25 -> -1.0)")
    p.add_argument("--off-penalty", type=float, default=0.0,
                   help="reward per tick per object off which the held stroke started inside (a carried block)")
    p.add_argument("--jerk-penalty", type=float, default=0.0,
                   help="reward per (world units/tick)^2 change of the held-stroke move")
    p.add_argument("--real", default="", help="comma-separated real charts (in --charts-dir) whose strokes are "
                   "added as windowed levels, e.g. the heart song")
    p.add_argument("--real-share", type=float, default=0.25, help="share of episodes from --real stroke windows")
    p.add_argument("--match", default="", help="only levels whose file name contains this (e.g. carrier)")
    p.add_argument("--anchor-weight", type=float, default=50.0,
                   help="pull toward the starting policy off-stroke (cursor) and everywhere (press/hold)")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--val-every", type=int, default=30, help="validation (clicking) eval period, in updates")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def load(path: str):
    ckpt = migrate_pre_split_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    actor = ActorNet(ckpt["max_objects"], ckpt["features_per_obj"], hidden=ckpt["hidden"],
                     use_map=bool(ckpt.get("use_map")), use_view=bool(ckpt.get("use_view")),
                     hold_head=bool(ckpt.get("hold_head")))
    actor.load_state_dict(ckpt["actor_state_dict"])
    if not actor.use_view:
        raise SystemExit("expects a checkpoint with the local view")
    apply_obs_flags(ckpt.get("obs_flags", {}))
    return ckpt, actor


def views_of(envs) -> np.ndarray:
    return np.stack([e.current_view for e in envs])


def main():
    args = parse_args()
    be_nice(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt, actor = load(args.init)
    actor.to(device)
    anchor = copy.deepcopy(actor).eval()
    for p_ in anchor.parameters():
        p_.requires_grad_(False)
    critic = CriticNet(ckpt["max_objects"], ckpt["features_per_obj"]).to(device)
    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.critic_lr)
    reward_norm = RunningNorm()

    # Only levels with a planned stroke: that is what is being trained.
    pairs = [pr for d in args.synth_dir for pr in find_chart_pairs(d)]
    pairs = [pr for pr in pairs if os.path.exists(pr[1][: -len(".events.json")] + ".trailplan.json")
             and args.match in os.path.basename(pr[0])]
    levels = [pr for pr in pairs if get_trail_plan(ChartData(*pr)).strokes]
    print(f"[rl] {len(levels)} stroke levels, device {device}", flush=True)
    eval_charts = [ChartData(*pr) for pr in find_chart_pairs(args.eval_dir) if args.match in os.path.basename(pr[0])]
    by_name = {os.path.basename(fp)[: -len(".frames.csv")]: (fp, ep) for fp, ep in find_chart_pairs(args.charts_dir)}
    holdout = [ChartData(*by_name[n]) for n in BALANCED_VALIDATION_CHARTS]

    # Real charts' strokes as windows: 1.5s before the stroke to 0.5s after.
    real_windows = []
    for name in filter(None, args.real.split(",")):
        chart = ChartData(*by_name[name])
        for s in get_trail_plan(chart).strokes:
            a = int(np.searchsorted(chart.t_ms, s.t_start - 1500.0))
            b = int(np.searchsorted(chart.t_ms, s.t_end + 500.0))
            real_windows.append((chart, a, b - a))
    print(f"[rl] {len(real_windows)} real stroke windows", flush=True)
    eval_charts += [ChartData(*by_name[n]) for n in filter(None, args.real.split(","))]

    def new_env():
        if real_windows and rng.random() < args.real_share:
            from augment import MODES as AUGMENT_MODES
            chart, start, length = real_windows[rng.randrange(len(real_windows))]
            env = TrailRLEnv(chart, window=(start, length), augmentation_mode=rng.choice(AUGMENT_MODES))
        else:
            env = make_env(ChartData(*levels[rng.randrange(len(levels))]), True, rng, 0, 0, True)
        env.compute_guide = False  # no teacher here
        return env

    def feat_fn(env, obs):
        return {"view": unpack_views(torch.from_numpy(env.current_view[None]).to(device))}

    envs = [new_env() for _ in range(args.num_envs)]
    obs = [e.reset() for e in envs]
    ride: list[set] = [set() for _ in envs]  # objects each held stroke started inside
    prev_move: list = [None for _ in envs]
    held_col = envs[0]._per_step_dim - rl_env.OWN_STATE_DIM + 2
    best = float("-inf")
    started = time.time()
    ep_stats = {"Wrong": 0, "hits": 0, "Miss": 0, "off_ticks": 0}

    for update in range(1, args.updates + 1):
        T, E = args.steps_per_env, args.num_envs
        buf_obs = np.zeros((T, E, obs[0].shape[0]), dtype=np.float32)
        buf_view = np.zeros((T, E, envs[0].current_view.shape[0]), dtype=np.uint8)
        buf_u = np.zeros((T, E, 2), dtype=np.float32)
        buf_logp = np.zeros((T, E), dtype=np.float32)
        buf_val = np.zeros((T, E), dtype=np.float32)
        buf_rew = np.zeros((T, E), dtype=np.float32)
        buf_done = np.zeros((T, E), dtype=bool)
        buf_held = np.zeros((T, E), dtype=bool)
        for t in range(T):
            x = torch.from_numpy(np.stack(obs)).float().to(device)
            vw = torch.from_numpy(views_of(envs)).to(device)
            with torch.no_grad():
                out = actor.forward(x, None, unpack_views(vw))
                loc = out["cursor_loc"]
                u = loc + args.sigma * torch.randn_like(loc)
                logp = (-0.5 * ((u - loc) / args.sigma) ** 2).sum(-1)
                value = critic(x)
                _, actions = actor.act_batch(x, view=unpack_views(vw))
            held_before = x[:, held_col] > 0.5
            # Explore only while holding (or on the tick a stroke starts):
            # everywhere else the policy moves exactly as it did.
            det = torch.tanh(loc) * CURSOR_COMPONENT_LIMIT
            move = torch.tanh(u) * CURSOR_COMPONENT_LIMIT
            cursor = torch.where(held_before[:, None], move, det).cpu().numpy()
            buf_obs[t] = x.cpu().numpy()
            buf_view[t] = vw.cpu().numpy()
            buf_u[t] = u.cpu().numpy()
            buf_logp[t] = logp.cpu().numpy()
            buf_val[t] = value.cpu().numpy()
            buf_held[t] = held_before.cpu().numpy()
            for i, env in enumerate(envs):
                was_held = env.trail_held
                obs[i], _r, done, info = env.step((float(cursor[i, 0]), float(cursor[i, 1])), int(actions[i]))
                r = info["judgment_reward"]
                if r < 0:
                    r *= args.wrong_scale
                # Riding: every tick spent off an object the stroke started
                # inside (the carried block) costs `off_penalty`. rl2 charged
                # only the leaving tick and then forgot the block, and since
                # getting back on is a fresh entry (a Wrong), it learned to
                # hover beside the block once it slipped: 17-24% of held ticks
                # off it, the heart song's "running outside". A reward, never
                # shown to the policy.
                if env.trail_held:
                    inside = set(env.judge._inside_collidables)
                    if not was_held:
                        ride[i] = inside
                        prev_move[i] = None
                    else:
                        off = len(ride[i] - inside)
                        ep_stats["off_ticks"] += int(off > 0)
                        r -= args.off_penalty * off
                    # Smoothness: squared change of the move (world units a
                    # tick) — the teacher reverses ~2% of held ticks
                    # relative to the block, the students ~10%.
                    mv = np.asarray(cursor[i], dtype=np.float64) * env.reach * env.chart.world_span
                    if prev_move[i] is not None:
                        r -= args.jerk_penalty * float(((mv - prev_move[i]) ** 2).sum())
                    prev_move[i] = mv
                else:
                    ride[i] = set()
                    prev_move[i] = None
                buf_rew[t, i] = r
                buf_done[t, i] = done
                if done:
                    for e in env.judge.log:
                        if e["judgment"] in ("Wrong", "Miss"):
                            ep_stats[e["judgment"]] += 1
                    ep_stats["hits"] += env.judge.hit_count
                    envs[i] = new_env()
                    obs[i] = envs[i].reset()
                    ride[i] = set()
                    prev_move[i] = None

        # GAE per env over the normalized reward.
        reward_norm.update(buf_rew.ravel())
        with torch.no_grad():
            last_v = critic(torch.from_numpy(np.stack(obs)).float().to(device)).cpu().numpy()
        adv = np.zeros((T, E), dtype=np.float32)
        ret = np.zeros((T, E), dtype=np.float32)
        for i in range(E):
            a, r = compute_gae(buf_rew[:, i] / (np.sqrt(reward_norm.var) + 1e-8), buf_val[:, i], buf_done[:, i],
                               float(last_v[i]), args.gamma, args.lam)
            adv[:, i], ret[:, i] = a, r
        flat = lambda a: a.reshape(T * E, *a.shape[2:])  # noqa: E731
        f_obs, f_view, f_u, f_logp = flat(buf_obs), flat(buf_view), flat(buf_u), flat(buf_logp)
        f_adv, f_ret, f_held = flat(adv), flat(ret), flat(buf_held)
        held_idx = np.flatnonzero(f_held)
        if len(held_idx) > 1:
            m, s = f_adv[held_idx].mean(), f_adv[held_idx].std() + 1e-8
            f_adv = (f_adv - m) / s

        stats = {"pi": 0.0, "v": 0.0, "anchor": 0.0, "n": 0}
        for _ in range(args.epochs):
            perm = np.random.permutation(T * E)
            for k in range(0, len(perm), args.minibatch):
                idx = perm[k : k + args.minibatch]
                x = torch.from_numpy(f_obs[idx]).to(device)
                view = unpack_views(torch.from_numpy(f_view[idx]).to(device))
                v_pred = critic(x)
                v_loss = F.mse_loss(v_pred, torch.from_numpy(f_ret[idx]).to(device))
                critic_opt.zero_grad()
                v_loss.backward()
                nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
                critic_opt.step()
                stats["v"] += float(v_loss)
                stats["n"] += 1
                if update <= args.critic_warmup:
                    continue
                out = actor.forward(x, None, view)
                with torch.no_grad():
                    ref = anchor.forward(x, None, view)
                held = torch.from_numpy(f_held[idx]).to(device)
                loss = x.new_zeros(())
                if held.any():
                    u = torch.from_numpy(f_u[idx]).to(device)[held]
                    loc = out["cursor_loc"][held]
                    logp = (-0.5 * ((u - loc) / args.sigma) ** 2).sum(-1)
                    ratio = (logp - torch.from_numpy(f_logp[idx]).to(device)[held]).clamp(-2, 2).exp()
                    a = torch.from_numpy(f_adv[idx]).to(device)[held]
                    pi_loss = -torch.min(ratio * a, ratio.clamp(1 - args.clip, 1 + args.clip) * a).mean()
                    loss = loss + pi_loss
                    stats["pi"] += float(pi_loss)
                # Anchor: cursor off-stroke, press/hold heads everywhere.
                cur = torch.tanh(out["cursor_loc"]) * CURSOR_COMPONENT_LIMIT
                cur_ref = torch.tanh(ref["cursor_loc"]) * CURSOR_COMPONENT_LIMIT
                free = ~held
                anchor_loss = x.new_zeros(())
                if free.any():
                    anchor_loss = anchor_loss + F.mse_loss(cur[free], cur_ref[free])
                anchor_loss = anchor_loss + 1e-3 * (
                    F.mse_loss(out["action_logits"], ref["action_logits"])
                    + F.mse_loss(out["hold_logit"], ref["hold_logit"])
                    + F.mse_loss(out["input_path_logit"], ref["input_path_logit"])
                )
                loss = loss + args.anchor_weight * anchor_loss
                stats["anchor"] += float(anchor_loss)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
                opt.step()
        n = max(1, stats["n"])
        print(f"[rl] update {update} held={f_held.mean():.3f} rew/step={buf_rew.mean():+.4f} "
              f"episodes: hits={ep_stats['hits']} Wrong={ep_stats['Wrong']} Miss={ep_stats['Miss']} "
              f"off_ticks={ep_stats['off_ticks']} "
              f"pi={stats['pi'] / n:+.4f} v={stats['v'] / n:.4f} anchor={stats['anchor'] / n:.6f} "
              f"elapsed={time.time() - started:.0f}s", flush=True)
        ep_stats = {"Wrong": 0, "hits": 0, "Miss": 0, "off_ticks": 0}

        if update % args.eval_every == 0 or update == args.updates:
            actor.eval()
            _h, _n, grades, t_macro = evaluate_holdout(actor, eval_charts, feat_fn=feat_fn)
            print(f"[rl eval {update}] stroke test hits={_h}/{_n} macro={t_macro:.2f}% {grades}", flush=True)
            score = t_macro
            if update % args.val_every == 0 or update == args.updates:
                _h, _n, vgrades, v_macro = evaluate_holdout(actor, holdout, feat_fn=feat_fn)
                print(f"[rl eval {update}] validation hits={_h}/{_n} macro={v_macro:.2f}% {vgrades}", flush=True)
            actor.train()
            out_ckpt = {**ckpt, "actor_state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
                        "iteration": update, "best_holdout_score": score,
                        "trained_by": f"train_stroke_rl.py from {args.init}",
                        "obs_flags": rl_env.current_obs_flags(), "own_state_dim": rl_env.OWN_STATE_DIM}
            torch.save(out_ckpt, args.save[: -len(".pt")] + "_last.pt")
            if score > best:
                best = score
                torch.save(out_ckpt, args.save)
                print(f"[rl] new best stroke-test macro={score:.2f}% -> {args.save}", flush=True)


if __name__ == "__main__":
    main()
