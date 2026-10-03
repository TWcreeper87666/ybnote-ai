import sys, io, time

_T0 = time.time()


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m {seconds:02d}s"


def _boot_log(msg: str) -> None:
    """Diagnostic-only startup and rollout progress logging."""
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

import config
_boot_log("config imported")
from data import ChartData
from augment import MODES as AUGMENT_MODES
_boot_log("data imported")
from ppo import PPOTrainer, RolloutBuffer
_boot_log("ppo imported")
import numpy as np

import rl_env
from rl_env import OWN_STATE_DIM, PRESS_NAMES, TrailRLEnv, decode_action, obs_features_per_obj, sample_window
_boot_log("rl_env imported")
from rl_policy import ACTION_SPACE, ActorNet, CriticNet, migrate_pre_split_checkpoint
_boot_log("rl_policy imported — all imports done")

BALANCED_VALIDATION_CHARTS = (
    "FALL FROM THE SKY PT. 2",                 # physical keyBinding + pitch match
    "Rhythm Hell",                             # pitch match, no physical keyBinding
    "JAWNY - Honeypie",                        # pitch match + many carried collidables
    "CHROMANCE – Wrap Me In Plastic",          # rotated targets, pitch match off
    "【imase】NIGHT DANCER",                   # dense ordinary mouse chart
)


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
    p.add_argument("--holdout", type=int, default=5, help="number of charts for random split policy")
    p.add_argument(
        "--split-policy",
        choices=("balanced", "random"),
        default="balanced",
        help="balanced uses a fixed archetype-spanning validation set; random uses --holdout and --seed",
    )
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--rollout-steps", type=int, default=4096, help="§11: PPO rollout buffer size per update")
    p.add_argument(
        "--num-envs",
        type=int,
        default=8,
        help="episodes (charts) kept alive at once; each update's --rollout-steps are split evenly "
        "across them so a batch mixes several charts instead of one chart's contiguous stretch",
    )
    p.add_argument("--stage-b-after", type=int, default=400, help="switch to full-chart episodes after this many iterations (§10)")
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument(
        "--augment",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="random D4 observation coordinate frame per training episode; evaluation remains unaugmented",
    )
    p.add_argument(
        "--early-stop-patience",
        type=int,
        default=3,
        help="stop after this many evaluations without a macro chart accuracy improvement; 0 disables early stopping",
    )
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument(
        "--wrong-penalty",
        type=float,
        default=-1.0,
        help="training-only penalty for touching a non-due object; evaluation always uses the real -0.25",
    )
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
    p.add_argument(
        "--critic-warmup",
        type=int,
        default=0,
        help="first N iterations update only the critic (actor frozen); use after resuming "
        "into a changed observation/reward so a stale value function doesn't steer the actor",
    )
    p.add_argument(
        "--no-timing-feature",
        action="store_true",
        help="ablation: zero the env's hit-timing object column (layout unchanged)",
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
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    _boot_log(f"resumed checkpoint loaded from {path!r}")
    old_action_space, old_fpo = ckpt.get("action_space"), ckpt.get("features_per_obj")
    ckpt = migrate_pre_split_checkpoint(ckpt)  # also pads newer env-appended object columns
    if old_action_space != ACTION_SPACE:
        print(
            f"[train_rl] migrated {path} from the pre-split no-op/attack/trail head to "
            f"press(click/key) + trail toggle (new input columns start at zero weight)"
        )
    elif old_fpo != ckpt["features_per_obj"]:
        print(
            f"[train_rl] padded {path} from {old_fpo} to {ckpt['features_per_obj']} features/object "
            f"(new env-appended columns start at zero weight)"
        )
    actor_result = actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
    critic_result = critic.load_state_dict(ckpt["critic_state_dict"], strict=False)
    prior_hits = ckpt.get("best_holdout_hits", "?")
    missing = actor_result.missing_keys + critic_result.missing_keys
    print(
        f"[train_rl] resumed actor/critic weights from {path} "
        f"(prior best_holdout_hits={prior_hits}, new_keys={missing or 'none'})"
    )


def make_env(
    chart: ChartData,
    stage_b: bool,
    rng: random.Random,
    window_min: int,
    window_max: int,
    augment: bool = True,
) -> TrailRLEnv:
    window = None if stage_b else sample_window(chart, window_min, window_max, rng)
    mode = rng.choice(AUGMENT_MODES) if augment else "identity"
    return TrailRLEnv(chart, window=window, augmentation_mode=mode)


def summarize_actions(buffers: list[RolloutBuffer], trail_held_index: int) -> str:
    """press none/click/key counts, trail toggles, and the share of ticks
    a stroke was held — enough to tell "never trails" from "flickers" from
    "holds strokes"."""
    press_counts = [0] * len(PRESS_NAMES)
    toggles = 0
    for action in (a for b in buffers for a in b.raw_action):
        press, toggle = decode_action(int(action.item()))
        press_counts[press] += 1
        toggles += toggle
    total = max(1, sum(len(b) for b in buffers))
    held = sum(float(obs[trail_held_index]) > 0.5 for b in buffers for obs in b.obs)
    press_text = " ".join(
        f"{name}={count}({100 * count / total:.1f}%)"
        for name, count in zip(PRESS_NAMES, press_counts)
    )
    return f"{press_text} trail_toggles={toggles} trail_held={100 * held / total:.1f}%"


@torch.no_grad()
def evaluate_holdout(actor: ActorNet, charts: list[ChartData], feat_fn=None) -> tuple[int, int, dict, float]:
    """Full-chart, deterministic (mean action, no sampling) — §10/§13.
    feat_fn(env, obs) -> extra act() kwargs (map_feat / view) for an actor
    with the map or view branch (train_bc --map / --local-view)."""
    device = next(actor.parameters()).device
    total_hits, total_notes = 0, 0
    grades_total: dict[str, int] = {}
    chart_accuracies = []
    # All charts step together, one batched forward per tick: one small
    # forward per chart per tick made a train_bc eval (11 charts, ~150k
    # ticks) take ~17 minutes, mostly GPU call overhead.
    envs = [TrailRLEnv(chart, window=None) for chart in charts]
    for env in envs:
        # The stroke guide is the teacher's; an evaluation has none (unless
        # the observation itself shows it).
        env.compute_guide = rl_env.STROKE_GUIDE_FEATURE_ENABLED
    obs_list = [env.reset() for env in envs]
    stats = [[0.0, 0, 0] for _ in envs]  # speed_sum, edge_ticks, ticks
    while True:
        live = [i for i, env in enumerate(envs) if not env.done]
        if not live:
            break
        x = torch.from_numpy(np.stack([obs_list[i] for i in live])).float().to(device)
        extra = {}
        if feat_fn:
            feats = [feat_fn(envs[i], obs_list[i]) for i in live]
            for key in feats[0]:
                # map_feat is [dim] per env, view [1, C, V, V].
                extra[key] = torch.cat([f[key].reshape(1, *f[key].shape[-3:]) if key == "view"
                                        else f[key].reshape(1, -1) for f in feats])
        cursors, actions = actor.act_batch(x, **extra)
        for row, i in enumerate(live):
            env = envs[i]
            prev = env.cursor
            obs_list[i], _reward, _done, _info = env.step(
                (float(cursors[row, 0]), float(cursors[row, 1])), int(actions[row])
            )
            s = stats[i]
            s[0] += ((env.cursor[0] - prev[0]) ** 2 + (env.cursor[1] - prev[1]) ** 2) ** 0.5
            s[1] += min(env.cursor) < 0.01 or max(env.cursor) > 0.99
            s[2] += 1
    for chart, env, (speed_sum, edge_ticks, ticks) in zip(charts, envs, stats):
        # Movement realism: average cursor speed in world units/s, and the
        # share of ticks pinned at the normalized-space edge (drift).
        world_speed = speed_sum / max(1, ticks) * chart.world_span * 1000.0 / config.DT_MS
        edge_share = edge_ticks / max(1, ticks)
        for e in env.judge.log:
            grades_total[e["judgment"]] = grades_total.get(e["judgment"], 0) + 1
        total_hits += env.judge.hit_count
        total_notes += len(chart.events)
        chart_grades = {
            name: sum(entry["judgment"] == name for entry in env.judge.log)
            for name in ("Perfect", "Good", "Bad", "Miss", "Wrong")
        }
        chart_hits = chart_grades["Perfect"] + chart_grades["Good"] + chart_grades["Bad"]
        chart_score = (
            chart_grades["Perfect"]
            + 0.75 * chart_grades["Good"]
            + 0.5 * chart_grades["Bad"]
            - 0.25 * chart_grades["Wrong"]
        )
        chart_notes = len(chart.events)
        chart_accuracy = 100 * chart_score / max(1, chart_notes)
        chart_accuracies.append(chart_accuracy)
        print(
            f"[eval chart] {chart.name} "
            f"hits={chart_hits}/{chart_notes} "
            f"accuracy={chart_accuracy:.1f}% "
            f"P:{chart_grades['Perfect']} G:{chart_grades['Good']} "
            f"B:{chart_grades['Bad']} M:{chart_grades['Miss']} W:{chart_grades['Wrong']} "
            f"speed={world_speed:.0f}/s edge={100 * edge_share:.0f}%",
            flush=True,
        )
    macro_accuracy = sum(chart_accuracies) / max(1, len(chart_accuracies))
    print(f"[eval] macro chart accuracy={macro_accuracy:.2f}% (each chart weighted equally)", flush=True)
    return total_hits, total_notes, grades_total, macro_accuracy


def main():
    args = parse_args()
    if args.no_timing_feature:
        import rl_env
        rl_env.TIMING_FEATURE_ENABLED = False
        print("[train_rl] ablation: hit-timing feature disabled (column zeroed)")
    _boot_log(f"args parsed: {args}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)

    pairs = find_chart_pairs(args.charts_dir)
    _boot_log(f"found {len(pairs)} chart pairs")
    if args.split_policy == "balanced":
        pairs_by_name = {
            os.path.basename(fp)[: -len(".frames.csv")]: (fp, ep)
            for fp, ep in pairs
        }
        missing = [name for name in BALANCED_VALIDATION_CHARTS if name not in pairs_by_name]
        if missing:
            raise SystemExit(
                "Balanced validation charts missing from --charts-dir: " + ", ".join(missing)
            )
        holdout_pairs = [pairs_by_name[name] for name in BALANCED_VALIDATION_CHARTS]
        holdout_names = set(BALANCED_VALIDATION_CHARTS)
        train_pairs = [
            pair for pair in pairs
            if os.path.basename(pair[0])[: -len(".frames.csv")] not in holdout_names
        ]
    else:
        if len(pairs) <= args.holdout:
            raise SystemExit(f"Only {len(pairs)} charts found, need more than --holdout ({args.holdout}).")
        pairs_shuffled = pairs[:]
        rng.shuffle(pairs_shuffled)
        holdout_pairs = pairs_shuffled[: args.holdout]
        train_pairs = pairs_shuffled[args.holdout :]
    print(f"[train_rl] {len(pairs)} charts: {len(train_pairs)} train, {len(holdout_pairs)} held out")
    print(
        f"[train_rl] {args.split_policy} validation split"
        f"{f' (seed={args.seed})' if args.split_policy == 'random' else ''}: "
        + ", ".join(os.path.basename(fp)[: -len(".frames.csv")] for fp, _ in holdout_pairs),
        flush=True,
    )

    print("[train_rl] loading charts...")
    train_charts = []
    for fp, ep in train_pairs:
        _boot_log(f"loading train chart: {os.path.basename(fp)}")
        train_charts.append(ChartData(fp, ep))
    holdout_charts = []
    for fp, ep in holdout_pairs:
        _boot_log(f"loading holdout chart: {os.path.basename(fp)}")
        holdout_charts.append(ChartData(fp, ep))
    _boot_log("all charts loaded")
    max_objects = train_charts[0].max_objects
    features_per_obj = train_charts[0].features_per_obj
    for c in train_charts + holdout_charts:
        if c.max_objects != max_objects or c.features_per_obj != features_per_obj:
            raise SystemExit("feature shape mismatch across charts — not handled by this trainer")
    # The env appends its own per-object columns (rl_env.EXTRA_OBJECT_FEATURES)
    # to the chart's encoded features; the nets see that widened layout.
    features_per_obj = obs_features_per_obj(features_per_obj)
    actor = ActorNet(max_objects, features_per_obj, hidden=args.hidden)
    _boot_log("ActorNet built")
    critic = CriticNet(max_objects, features_per_obj, hidden=args.hidden)
    _boot_log("CriticNet built")
    # Current-tick trail_held inside an observation (history slot 0's
    # own-state block, see rl_env._raw_features_vec), for rollout stats.
    trail_held_index = actor.object_dim + actor.obstacle_dim + 2
    if args.resume_from:
        resume_from_checkpoint(actor, critic, args.resume_from)
    elif args.warm_start:
        best_effort_warm_start(actor, args.warm_start)

    trainer = PPOTrainer(actor, critic, lr=args.lr)
    if args.resume_from:
        # The critic was fit to rewards normalized by the previous run's
        # running statistics; restarting them from (0, 1) rescales every
        # value target under the resumed critic.
        norm = torch.load(args.resume_from, map_location="cpu", weights_only=False).get("reward_norm")
        if norm:
            trainer.reward_norm.mean, trainer.reward_norm.var, trainer.reward_norm.count = (
                norm["mean"], norm["var"], norm["count"]
            )
            print(f"[train_rl] restored reward normalization (mean={norm['mean']:.4f}, var={norm['var']:.4f})")
    _boot_log("PPOTrainer built, entering training loop")

    wrong_final = config.JUDGMENT_REWARD["Wrong"]
    config.JUDGMENT_REWARD["Wrong"] = args.wrong_penalty
    annealing = bool(args.anneal_wrong_penalty) and not args.warm_start and not args.resume_from

    stage_b = False

    def pick_chart(in_use: list[ChartData]) -> ChartData:
        """Random training chart, preferring ones no other live env is on."""
        free = [c for c in train_charts if all(c is not u for u in in_use)]
        pool = free or train_charts
        return pool[rng.randrange(len(pool))]

    num_envs = max(1, min(args.num_envs, args.rollout_steps))
    env_charts: list[ChartData] = []
    envs: list[TrailRLEnv] = []
    env_obs = []
    for _ in range(num_envs):
        chart = pick_chart(env_charts)
        env_charts.append(chart)
        envs.append(make_env(chart, stage_b, rng, args.window_min, args.window_max, args.augment))
        env_obs.append(envs[-1].reset())
    steps_per_env = [args.rollout_steps // num_envs + (i < args.rollout_steps % num_envs) for i in range(num_envs)]
    print(f"[train_rl] {num_envs} concurrent envs, {steps_per_env[0]} steps each per update")
    _boot_log("first env.reset() done")
    training_started_at = time.time()

    # Don't let a fresh run (this script has no train-state resumption —
    # each invocation restarts from iteration 1 and re-does warm-start)
    # clobber a better checkpoint an earlier run already left at
    # args.save: only overwrite it once a NEW run's holdout score actually
    # beats whatever's already there. Legacy checkpoints without a score
    # keep using their stored hit count as the overwrite guard.
    best_holdout_hits = -1
    best_holdout_score = None
    best_holdout_metric = "macro_chart_weighted_accuracy_pct"
    phase_best_holdout_score = float("-inf")
    evaluations_without_improvement = 0
    if os.path.exists(args.save):
        try:
            existing = torch.load(args.save, map_location="cpu", weights_only=False)
            best_holdout_hits = existing.get("best_holdout_hits", -1)
            existing_metric = existing.get("best_holdout_metric")
            if existing_metric == best_holdout_metric:
                best_holdout_score = existing.get("best_holdout_score")
            existing_score_text = (
                f"{best_holdout_score:.2f}% macro"
                if best_holdout_score is not None
                else f"incompatible/legacy metric ({existing_metric or 'total-score'})"
            )
            print(f"[train_rl] found existing checkpoint at {args.save!r} with "
                  f"{best_holdout_hits} best_holdout_hits and "
                  f"{existing_score_text} best_holdout_score — new run must beat that to overwrite it")
        except Exception as e:
            print(f"[train_rl] could not read existing checkpoint at {args.save!r} ({e}) — starting from -1")

    for it in range(1, args.iterations + 1):
        elapsed = time.time() - training_started_at
        completed = it - 1
        average_iteration = elapsed / completed if completed else 0.0
        remaining = args.iterations - completed
        eta = average_iteration * remaining if completed else None
        eta_text = _format_duration(eta) if eta is not None else "calculating"
        print(
            f"[train] iteration {it}/{args.iterations} | "
            f"remaining: {args.iterations - it} | elapsed: {_format_duration(elapsed)} | "
            f"ETA: {eta_text}",
            flush=True,
        )
        if not stage_b and it >= args.stage_b_after:
            stage_b = True
            phase_best_holdout_score = float("-inf")
            evaluations_without_improvement = 0
            print(f"[train_rl] iteration {it}: switching to Stage B (full-chart episodes)")

        if annealing:
            frac = min(1.0, it / args.anneal_wrong_penalty)
            config.JUDGMENT_REWARD["Wrong"] = wrong_final * frac

        # One buffer per env: each is its own contiguous trajectory for GAE
        # (a finished episode inside it is marked by `done` and the env is
        # replaced by a fresh chart, preferring one no other env is on).
        buffers = []
        for i in range(num_envs):
            buffer = RolloutBuffer()
            while len(buffer) < steps_per_env[i]:
                env_obs[i] = trainer.collect_rollout(
                    envs[i],
                    steps_per_env[i] - len(buffer),
                    buffer,
                    env_obs[i],
                    progress_label=f"iter {it}/{args.iterations} env {i}",
                )
                if envs[i].done:
                    others = env_charts[:i] + env_charts[i + 1 :]
                    env_charts[i] = pick_chart(others)
                    envs[i] = make_env(env_charts[i], stage_b, rng, args.window_min, args.window_max, args.augment)
                    env_obs[i] = envs[i].reset()
            buffers.append(buffer)

        train_actor = it > args.critic_warmup
        if not train_actor and it == 1:
            print(f"[train_rl] critic warmup: actor frozen for iterations 1-{args.critic_warmup}")
        # A replaced env's buffer ended on a true episode end (done flag in
        # the buffer), so its bootstrap obs belongs to the NEW episode and
        # is only used when the buffer's last step wasn't terminal.
        stats = trainer.update(
            buffers,
            env_obs,
            [bool(b.dones[-1]) for b in buffers],
            train_actor=train_actor,
        )
        action_summary = summarize_actions(buffers, trail_held_index)
        elapsed = time.time() - training_started_at
        average_iteration = elapsed / it
        remaining_iterations = args.iterations - it
        print(
            f"[train] iteration {it}/{args.iterations} complete | "
            f"remaining: {remaining_iterations} | "
            f"elapsed: {_format_duration(elapsed)} | "
            f"ETA: {_format_duration(average_iteration * remaining_iterations)} | "
            f"policy_loss={stats['policy_loss']:.4f} value_loss={stats['value_loss']:.4f} "
            f"entropy={stats['entropy']:.4f} actions[{action_summary}]",
            flush=True,
        )
        if (it % args.eval_every == 0 or it == args.iterations) and train_actor:
            config.JUDGMENT_REWARD["Wrong"] = wrong_final  # eval always uses the real rule, §10/§13
            hits, notes, grades, macro_accuracy = evaluate_holdout(actor, holdout_charts)
            pct = 100 * hits / max(1, notes)
            grade_str = " ".join(f"{k}:{v}" for k, v in sorted(grades.items()))
            weighted = (
                grades.get("Perfect", 0)
                + 0.75 * grades.get("Good", 0)
                + 0.5 * grades.get("Bad", 0)
                - 0.25 * grades.get("Wrong", 0)
            )
            accuracy = 100 * weighted / max(1, notes)
            improved_this_phase = macro_accuracy > phase_best_holdout_score
            if improved_this_phase:
                phase_best_holdout_score = macro_accuracy
                evaluations_without_improvement = 0
            else:
                evaluations_without_improvement += 1
            print(
                f"[iter {it:5d}] hits={hits}/{notes} ({pct:.1f}%) "
                f"note-weighted={accuracy:.1f}% macro-chart={macro_accuracy:.1f}% {grade_str}"
            )
            beats_existing = (
                (best_holdout_score is None and hits > best_holdout_hits)
                or (best_holdout_score is not None and macro_accuracy > best_holdout_score)
            )
            if beats_existing:
                best_holdout_score = macro_accuracy
                best_holdout_hits = hits
                torch.save(
                    {
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "max_objects": max_objects,
                        "features_per_obj": features_per_obj,
                        "own_state_dim": OWN_STATE_DIM,
                        "action_space": ACTION_SPACE,
                        "hidden": args.hidden,
                        "best_holdout_hits": best_holdout_hits,
                        "best_holdout_score": best_holdout_score,
                        "best_holdout_metric": best_holdout_metric,
                        "best_holdout_note_weighted_accuracy": accuracy,
                        "best_holdout_accuracy": accuracy,
                        "best_holdout_grades": grades,
                        "iteration": it,
                        "reward_norm": {
                            "mean": trainer.reward_norm.mean,
                            "var": trainer.reward_norm.var,
                            "count": trainer.reward_norm.count,
                        },
                    },
                    args.save,
                )
                print(
                    f"[train_rl] new best (macro chart accuracy={best_holdout_score:.2f}%, "
                    f"{best_holdout_hits} hits) -> {args.save}"
                )
            elif stage_b:
                if args.early_stop_patience > 0:
                    print(
                        f"[train_rl] no macro chart accuracy improvement for "
                        f"{evaluations_without_improvement}/{args.early_stop_patience} evaluations"
                    )
                    if evaluations_without_improvement >= args.early_stop_patience:
                        print(
                            f"[train_rl] early stopping at iteration {it}: "
                            f"best Stage B macro chart accuracy={phase_best_holdout_score:.2f}%"
                        )
                        break

            config.JUDGMENT_REWARD["Wrong"] = args.wrong_penalty

    if best_holdout_score is None:
        saved_score_text = "unavailable for legacy checkpoint"
    else:
        saved_score_text = f"{best_holdout_score:.2f}"
    print(
        f"[train_rl] done. best holdout hits: {best_holdout_hits}, "
        f"best saved macro chart accuracy: {saved_score_text}"
    )


if __name__ == "__main__":
    main()
