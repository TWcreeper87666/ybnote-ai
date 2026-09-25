import sys, io, time

_T0 = time.time()


def _boot_log(msg: str) -> None:
    """Diagnostic-only: prints BEFORE the real stdout wrapper is even set
    up, using the raw default stdout, so a hang anywhere in this file's
    top-level imports is visible instead of silent. Two mysterious
    reports so far: this process consumes real CPU/memory (confirmed via
    Task Manager / Get-Process — not a sandbox artifact) but prints
    NOTHING, not even this module's first print() call, which should be
    unreachable in under a few seconds. Remove once the actual stall
    point is identified."""
    print(f"[boot {time.time()-_T0:6.2f}s] {msg}", flush=True)


_boot_log("process started, about to import torch")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")
"""PPO training entrypoint for the end-to-end RL agent — RL_DESIGN.md.

Usage:
    python train_rl.py --charts-dir ../output --holdout 5 --iterations 2000 \
        --save rl_policy.pt

Curriculum (§10): Stage A (random short windows from random charts) until
--stage-b-after iterations, then Stage B (full charts) for the remainder.
Final eval is always full-chart, deterministic, held-out (§10/§13).
"""

import argparse
import glob
import os
import random

_boot_log("stdlib imports done, importing torch")
import torch

_boot_log("torch imported, configuring thread pool")

# Every actor/critic call in the rollout loop is a single unbatched forward
# on a small MLP (rl_policy.py) — torch's default BLAS/OpenMP thread pool
# (one worker per core) adds pure synchronization overhead for ops this
# tiny, and under whatever CPU scheduling a background/throttled process
# gets, that overhead compounds into the process barely making progress
# (observed: 15-18% CPU utilization, zero output after 10+ minutes on a
# rollout that takes single-digit seconds interactively). Single-threaded
# is strictly faster here.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

_boot_log("torch thread pool configured, importing project modules")
import config
_boot_log("config imported")
from data import ChartData
_boot_log("data imported")
from ppo import PPOTrainer, RolloutBuffer
_boot_log("ppo imported")
from rl_env import TrailRLEnv, sample_window
_boot_log("rl_env imported")
from rl_policy import ActorNet, CriticNet
_boot_log("rl_policy imported — all imports done")


def find_chart_pairs(charts_dir: str):
    pairs = []
    for frames_path in sorted(glob.glob(os.path.join(charts_dir, "*.frames.csv"))):
        base = frames_path[: -len(".frames.csv")]
        events_path = base + ".events.json"
        if not os.path.exists(events_path):
            continue
        if os.path.getsize(frames_path) == 0:
            continue
        with open(frames_path, "r", encoding="utf-8") as f:
            if len(f.readlines()) < 2:
                continue
        pairs.append((frames_path, events_path))
    return pairs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--charts-dir", required=True)
    p.add_argument("--holdout", type=int, default=5)
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--rollout-steps", type=int, default=4096, help="§11: PPO rollout buffer size per update")
    p.add_argument("--stage-b-after", type=int, default=1500, help="switch to full-chart episodes after this many iterations (§10)")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--window-min", type=int, default=500)
    p.add_argument("--window-max", type=int, default=1500)
    p.add_argument(
        "--warm-start",
        default="dl_policy_multi.pt",
        help="path to a supervised ChartPolicyNet checkpoint for best-effort partial warm-start "
             "(RL_DESIGN.md §14 open question — resolved here as warm-start-by-default, the "
             "pragmatically safer option given this project's convergence history); "
             "pass empty string ('') for a genuine cold start",
    )
    p.add_argument(
        "--anneal-wrong-penalty",
        type=int,
        default=0,
        help="iterations over which to anneal the Wrong penalty up from ~0 to its real value "
             "(config.JUDGMENT_REWARD['Wrong']) — RL_DESIGN.md §14's cold-start mitigation; "
             "only meaningful without --warm-start (0 disables annealing)",
    )
    p.add_argument(
        "--resume-from",
        default="",
        help="path to a PRIOR rl_policy.pt-format checkpoint (this script's own --save output) to "
             "continue training from — loads actor_state_dict/critic_state_dict EXACTLY (same "
             "architecture, full match), unlike --warm-start's partial load from a different "
             "supervised-net architecture. Overrides --warm-start when set (a full-match resume is "
             "strictly more informative than a partial one). Note: this only resumes NETWORK "
             "WEIGHTS, not optimizer/iteration state — the run still starts counting from iteration "
             "1 and re-anneals/re-explores; the point is picking up from better-than-random weights, "
             "not a byte-exact continuation.",
    )
    p.add_argument("--save", default="rl_policy.pt")
    p.add_argument("--seed", type=int, default=config.SEED)
    return p.parse_args()


def best_effort_warm_start(actor: ActorNet, path: str):
    """Loads whatever shapes happen to match from a supervised
    ChartPolicyNet checkpoint (dl_model.py) into ActorNet. The two
    architectures diverge structurally — ActorNet's trunk takes a
    STACKED-history input (rl_env.HISTORY_STRIDES) the supervised net never
    saw, and cursor_head predicts a Gaussian's 4 params instead of a direct
    2-value regression — so this is NOT a full behavior-cloning warm start,
    only a partial initialization of whatever hidden->hidden layers happen
    to line up. See RL_DESIGN.md §14's open question for why warm-starting
    was chosen as this project's default anyway."""
    if not path or not os.path.exists(path):
        print(f"[train_rl] warm-start checkpoint not found ({path!r}) — cold start")
        return
    _boot_log(f"os.path.exists confirmed for {path!r}, calling torch.load now")
    ckpt = torch.load(path, map_location="cpu")
    _boot_log("torch.load returned")
    src = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    dst = actor.state_dict()
    loaded, skipped = [], []
    for key, tensor in src.items():
        if key in dst and dst[key].shape == tensor.shape:
            dst[key] = tensor
            loaded.append(key)
        else:
            skipped.append(key)
    actor.load_state_dict(dst)
    print(f"[train_rl] warm-start from {path}: loaded {len(loaded)} matching tensors, skipped {len(skipped)}")
    if loaded:
        print(f"  loaded: {loaded}")


def resume_from_checkpoint(actor: ActorNet, critic: CriticNet, path: str):
    """Loads actor_state_dict/critic_state_dict EXACTLY from a checkpoint
    this same script produced (--save's format) — full match, same
    architecture, unlike best_effort_warm_start's partial load from a
    different (supervised ChartPolicyNet) architecture."""
    _boot_log(f"resuming from {path!r}, calling torch.load")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    actor.load_state_dict(ckpt["actor_state_dict"])
    critic.load_state_dict(ckpt["critic_state_dict"])
    prior_hits = ckpt.get("best_holdout_hits", "?")
    print(f"[train_rl] resumed actor/critic weights from {path} (prior best_holdout_hits={prior_hits})")


def make_env(chart: ChartData, stage_b: bool, rng: random.Random, window_min: int, window_max: int) -> TrailRLEnv:
    window = None if stage_b else sample_window(chart, window_min, window_max, rng)
    return TrailRLEnv(chart, window=window)


@torch.no_grad()
def evaluate_holdout(actor: ActorNet, charts: list[ChartData]) -> tuple[int, int, dict]:
    """Full-chart, deterministic (mean action, no sampling) — §10/§13."""
    total_hits, total_notes = 0, 0
    grades_total: dict[str, int] = {}
    for chart in charts:
        env = TrailRLEnv(chart, window=None)
        obs = env.reset()
        while not env.done:
            obs_t = torch.from_numpy(obs).float()
            act = actor.act(obs_t, deterministic=True)
            obs, _reward, _done, _info = env.step(act["cursor_delta"], act["attack_raw"], act["trail_held"])
        for e in env.judge.log:
            grades_total[e["judgment"]] = grades_total.get(e["judgment"], 0) + 1
        total_hits += env.judge.hit_count
        total_notes += len(chart.events)
    return total_hits, total_notes, grades_total


def main():
    _boot_log("main() entered, parsing args")
    args = parse_args()
    _boot_log(f"args parsed: {args}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    _boot_log(f"scanning {args.charts_dir!r} for chart pairs")
    pairs = find_chart_pairs(args.charts_dir)
    _boot_log(f"found {len(pairs)} chart pairs")
    if len(pairs) <= args.holdout:
        raise SystemExit(f"Only {len(pairs)} charts found, need more than --holdout ({args.holdout}).")
    pairs_shuffled = pairs[:]
    rng.shuffle(pairs_shuffled)
    holdout_pairs = pairs_shuffled[: args.holdout]
    train_pairs = pairs_shuffled[args.holdout :]
    print(f"[train_rl] {len(pairs)} charts: {len(train_pairs)} train, {len(holdout_pairs)} held out")

    print("[train_rl] loading charts...")
    train_charts = []
    for i, (fp, ep) in enumerate(train_pairs):
        _boot_log(f"loading train chart {i+1}/{len(train_pairs)}: {os.path.basename(fp)}")
        train_charts.append(ChartData(fp, ep))
    holdout_charts = []
    for i, (fp, ep) in enumerate(holdout_pairs):
        _boot_log(f"loading holdout chart {i+1}/{len(holdout_pairs)}: {os.path.basename(fp)}")
        holdout_charts.append(ChartData(fp, ep))
    _boot_log("all charts loaded")
    max_objects = train_charts[0].max_objects
    features_per_obj = train_charts[0].features_per_obj
    for c in train_charts + holdout_charts:
        if c.max_objects != max_objects or c.features_per_obj != features_per_obj:
            raise SystemExit("feature shape mismatch across charts — not handled by this trainer")
    _boot_log(f"feature shapes checked (max_objects={max_objects}, features_per_obj={features_per_obj}), building ActorNet")

    actor = ActorNet(max_objects, features_per_obj, hidden=args.hidden)
    _boot_log("ActorNet built, building CriticNet")
    critic = CriticNet(max_objects, features_per_obj, hidden=args.hidden)
    _boot_log("CriticNet built")
    if args.resume_from:
        resume_from_checkpoint(actor, critic, args.resume_from)
    elif args.warm_start:
        _boot_log(f"warm-start requested from {args.warm_start!r}, calling torch.load")
        best_effort_warm_start(actor, args.warm_start)
        _boot_log("warm-start done")

    trainer = PPOTrainer(actor, critic, lr=args.lr)
    _boot_log("PPOTrainer built, entering training loop")

    wrong_final = config.JUDGMENT_REWARD["Wrong"]
    annealing = bool(args.anneal_wrong_penalty) and not args.warm_start and not args.resume_from

    current_chart = train_charts[rng.randrange(len(train_charts))]
    stage_b = False
    env = make_env(current_chart, stage_b, rng, args.window_min, args.window_max)
    _boot_log(f"first env ready: chart steps={env.length}, entering iteration loop")
    obs = env.reset()
    _boot_log("first env.reset() done")

    # Don't let a fresh run (this script has no train-state resumption —
    # each invocation restarts from iteration 1 and re-does warm-start)
    # clobber a better checkpoint an earlier run already left at
    # args.save: only overwrite it once a NEW run's holdout hits actually
    # beat whatever's already there.
    best_holdout_hits = -1
    if os.path.exists(args.save):
        try:
            existing = torch.load(args.save, map_location="cpu", weights_only=False)
            best_holdout_hits = existing.get("best_holdout_hits", -1)
            print(f"[train_rl] found existing checkpoint at {args.save!r} with "
                  f"{best_holdout_hits} best_holdout_hits — new run must beat that to overwrite it")
        except Exception as e:
            print(f"[train_rl] could not read existing checkpoint at {args.save!r} ({e}) — starting from -1")

    for it in range(1, args.iterations + 1):
        if not stage_b and it >= args.stage_b_after:
            stage_b = True
            print(f"[train_rl] iteration {it}: switching to Stage B (full-chart episodes)")

        if annealing:
            frac = min(1.0, it / args.anneal_wrong_penalty)
            config.JUDGMENT_REWARD["Wrong"] = wrong_final * frac

        buffer = RolloutBuffer()
        steps_collected = 0
        while steps_collected < args.rollout_steps:
            remaining = args.rollout_steps - steps_collected
            obs = trainer.collect_rollout(env, remaining, buffer, obs)
            steps_collected = len(buffer)
            if env.done:
                current_chart = train_charts[rng.randrange(len(train_charts))]
                env = make_env(current_chart, stage_b, rng, args.window_min, args.window_max)
                obs = env.reset()

        stats = trainer.update(buffer, obs, env.done)
        if it % 10 == 0 or it == 1:
            wrong_note = f" wrong_penalty={config.JUDGMENT_REWARD['Wrong']:.3f}" if annealing else ""
            print(
                f"[iter {it:5d}] policy_loss={stats['policy_loss']:.4f} "
                f"value_loss={stats['value_loss']:.4f} entropy={stats['entropy']:.4f}{wrong_note}"
            )

        if it % args.eval_every == 0 or it == args.iterations:
            config.JUDGMENT_REWARD["Wrong"] = wrong_final  # eval always uses the real rule, §10/§13
            hits, notes, grades = evaluate_holdout(actor, holdout_charts)
            pct = 100 * hits / max(1, notes)
            grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
            print(f"[iter {it:5d}] HOLDOUT hits={hits}/{notes} ({pct:.1f}%) {grade_str}")
            if hits > best_holdout_hits:
                best_holdout_hits = hits
                torch.save(
                    {
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "max_objects": max_objects,
                        "features_per_obj": features_per_obj,
                        "hidden": args.hidden,
                        "best_holdout_hits": best_holdout_hits,
                        "iteration": it,
                    },
                    args.save,
                )
                print(f"[train_rl] new best ({best_holdout_hits} hits) -> {args.save}")

    print(f"[train_rl] done. best holdout hits: {best_holdout_hits}")


if __name__ == "__main__":
    main()
