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
    python training/train_bc.py --charts-dir data/output --init training/models/rl_policy_v7_groupfix.pt \
        --save training/models/rl_policy_bc.pt
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
from niceness import be_nice
from rl_env import NUM_PRESS, PRESS_KEY, PRESS_NONE, TrailRLEnv, decode_action, obs_features_per_obj
from rl_policy import ACTION_SPACE, CURSOR_COMPONENT_LIMIT, ActorNet, CriticNet, migrate_pre_split_checkpoint
from train_rl import BALANCED_VALIDATION_CHARTS, evaluate_holdout, find_chart_pairs, make_env
from trail_plan import get_trail_plan

# Positive-class weight of the toggle BCE (see bc_loss).
TOGGLE_POS_WEIGHT = 2.0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--charts-dir", required=True)
    p.add_argument("--init", default="", help="train_rl.py checkpoint to start from (actor+critic); empty = fresh")
    p.add_argument("--save", default="training/models/rl_policy_bc.pt")
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
    p.add_argument("--device", choices=["auto", "cpu"], default="auto", help="auto: the GPU if there is one")
    p.add_argument("--stroke-expert-share", type=float, default=0.5,
                   help="share of planned strokes the EXPERT drives from take-over to release (the rest the "
                        "student drives, labeled as usual). Per-tick mixing let the student let go right after "
                        "the expert's start, so a whole held stroke almost never reached the buffer (bc11: 0.05%% "
                        "of states held)")
    p.add_argument("--stroke-beta-min", type=float, default=0.3,
                   help="DAgger floor on the expert's share of steps on charts with a planned stroke: a student "
                        "that clicks the start note loses the stroke, so without the expert driving some of the "
                        "time no held states reach the buffer once beta has decayed")
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--cursor-weight", type=float, default=50.0,
                   help="weight of the cursor MSE (targets are speed-ceiling fractions, |x| <= 0.71)")
    p.add_argument("--stroke-cursor-weight", type=float, default=1.0,
                   help="cursor-loss weight of stroke states relative to the rest")
    p.add_argument("--toggle-weight", type=float, default=1.0, help="weight of the trail toggle BCE")
    p.add_argument("--how-weight", type=float, default=1.0,
                   help="weight of the click-vs-key BCE (bc20's how head barely separated a sole key from a shared one)")
    p.add_argument("--hold-head", action=argparse.BooleanOptionalAction, default=True,
                   help="trail head outputs the held state, toggles derived (rl_policy.ActorNet hold_head)")
    p.add_argument("--stroke-share", type=float, default=0.25,
                   help="share of each batch drawn from stroke-active states (see Buffer)")
    p.add_argument("--toggle-share", type=float, default=0.06,
                   help="share of each batch drawn from expert toggle (start/release) labels")
    p.add_argument("--stroke-chart-weight", type=float, default=4.0,
                   help="sampling weight of training charts that have a planned trail stroke (trail_plan), "
                        "vs 1 for the rest: 2 of ~27 charts, else strokes are too rare in the buffer")
    p.add_argument("--key-chart-weight", type=float, default=1.0,
                   help="sampling weight of training charts with key-bound notes, vs 1 for the rest: only 7 of ~33 "
                        "charts have them and generated levels have none, so bc18 (synth-share 0.85) barely saw "
                        "key states and left 戀愛循環 clicking (its how-head P(key) ~0)")
    p.add_argument("--synth-dir", nargs="*", default=[],
                   help="folder of encoded scripts/generate_trail_levels.py levels (plans precomputed with "
                        "precompute_trail_plans.py), loaded per episode")
    p.add_argument("--synth-share", type=float, default=0.4, help="share of episodes drawn from --synth-dir")
    p.add_argument("--trail-eval-dir", default="",
                   help="folder of held-out generated levels, scored with --trail-eval at every eval")
    p.add_argument("--trail-eval", default="只因為你那渴望自由的心臟🫀,迷宮🗣️🔥",
                   help="comma-separated training charts that need trail strokes, also scored at every eval; "
                        "the best checkpoint is picked on the macro over validation + these")
    p.add_argument("--student-obs", choices=["guided", "clean"], default="clean",
                   help="clean: no planner answers in the observation (stroke guide and safe click point off) "
                        "plus the note lookahead; guided: bc9_trail's observation")
    p.add_argument("--local-view", action=argparse.BooleanOptionalAction, default=None,
                   help="local view around the cursor (default: on with --student-obs clean)")
    p.add_argument("--real-window", type=int, nargs=2, default=[0, 0], metavar=("MIN", "MAX"),
                   help="play stroke-free real charts in random windows of MIN..MAX ticks (0 0 = whole songs)")
    p.add_argument("--whiskers", action="store_true",
                   help="exact ray distances to what a stroke would trigger/leave (nav_map.whisker_features)")
    p.add_argument("--carrier", action="store_true",
                   help="velocity of / offset on the object a held stroke rides (nav_map.carrier_features)")
    p.add_argument("--map", action="store_true",
                   help="whole-level map + learned value iteration (nav_map), trained on the GPU if there is one")
    p.add_argument("--map-iters", type=int, default=1280, help="value-iteration passes (route length in cells)")
    p.add_argument("--vin-init", default="", help="train_vin.py checkpoint (use with --vin-updates 0 to freeze it)")
    p.add_argument("--vin-batch", type=int, default=16, help="maps per value-iteration distance-loss step")
    p.add_argument("--vin-updates", type=int, default=20, help="distance-loss steps per iteration")
    p.add_argument("--vin-lr", type=float, default=1e-3)
    p.add_argument("--timing-feature", action="store_true",
                   help="fill the hit_timing_at() column (zeroed by default: it hurt PPO, v8b/v8c)")
    p.add_argument("--attack-clock", action="store_true",
                   help="show the policy its own ticks_since_attack (off by default: causal confusion, see rl_env)")
    p.add_argument("--no-expert-keys", action="store_true",
                   help="teacher always clicks (default: it presses the note's own key when no other object shares it)")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--threads", type=int, default=4, help="torch threads (below-normal priority either way)")
    return p.parse_args()


class Buffer:
    """FIFO of (obs float16, cursor target, press class, toggle target).

    Stroke states are rare (2 charts of 27; a stroke starts on ONE tick per
    episode), so sample() mixes in a fixed share of them: bc8's first run,
    sampling uniformly, left P(toggle) at 0.005 on the start tick while the
    press head rose to 0.6 and clicked the start note instead."""

    def __init__(self, size: int, obs_dim: int, map_bytes: int = 0, view_bytes: int = 0):
        self.obs = np.zeros((size, obs_dim), dtype=np.float16)
        self.views = np.zeros((size, view_bytes), dtype=np.uint8) if view_bytes else None
        # Maps (24KB each at 256x256) are shared: a stroke's map stays the
        # same for many ticks, so each sample holds an index into a pool of
        # distinct maps (-1 = no map: not holding).
        self.use_maps = bool(map_bytes)
        self.map_idx = np.full(size, -1, dtype=np.int32)
        self.map_pool: list[np.ndarray] = []
        self.map_key: dict[bytes, int] = {}
        self.cursor = np.zeros((size, 2), dtype=np.float32)
        self.press = np.zeros(size, dtype=np.int64)
        self.toggle = np.zeros(size, dtype=np.float32)
        # Teacher-side flag (a planned stroke is in play), for sampling only;
        # never part of the observation.
        self.stroke = np.zeros(size, dtype=bool)
        self.size, self.n, self.pos = size, 0, 0

    def add(self, obs, cursor, press, toggle, stroke, packed_map=None, packed_view=None):
        self.obs[self.pos] = obs
        if self.views is not None:
            self.views[self.pos] = packed_view
        if self.use_maps:
            self.map_idx[self.pos] = -1 if packed_map is None else self._map_slot(packed_map)
        self.cursor[self.pos] = cursor
        self.press[self.pos] = press
        self.toggle[self.pos] = toggle
        self.stroke[self.pos] = stroke
        self.pos = (self.pos + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def _map_slot(self, packed: np.ndarray) -> int:
        key = packed.tobytes()
        slot = self.map_key.get(key)
        if slot is None:
            if len(self.map_pool) >= 4096:
                self._compact_maps()
            slot = len(self.map_pool)
            self.map_pool.append(packed.copy())
            self.map_key[key] = slot
        return slot

    def _compact_maps(self):
        """Drop pool maps no live sample refers to any more."""
        live = np.unique(self.map_idx[: self.n][self.map_idx[: self.n] >= 0])
        remap = np.full(len(self.map_pool) + 1, -1, dtype=np.int32)
        remap[live] = np.arange(len(live))
        self.map_pool = [self.map_pool[i] for i in live]
        self.map_key = {m.tobytes(): i for i, m in enumerate(self.map_pool)}
        self.map_idx = remap[self.map_idx]  # -1 indexes the trailing -1

    def maps_at(self, idx: np.ndarray) -> list[np.ndarray | None]:
        return [None if i < 0 else self.map_pool[i] for i in self.map_idx[idx]]

    def sample(self, batch: int, stroke_share: float, toggle_share: float):
        idx = np.random.randint(0, self.n, size=batch)
        live = slice(0, self.n)
        stroke_idx = np.flatnonzero(self.stroke[live])
        toggle_idx = np.flatnonzero(self.toggle[live] > 0.5)
        k = 0
        for pool, share in ((toggle_idx, toggle_share), (stroke_idx, stroke_share)):
            m = int(round(batch * share)) if len(pool) else 0
            idx[k : k + m] = pool[np.random.randint(0, len(pool), size=m)]
            k += m
        return (
            torch.from_numpy(self.obs[idx].astype(np.float32)),
            torch.from_numpy(self.cursor[idx]),
            torch.from_numpy(self.press[idx]),
            torch.from_numpy(self.toggle[idx]),
            self.maps_at(idx) if self.use_maps else None,
            None if self.views is None else self.views[idx],
            torch.from_numpy(self.stroke[idx]),
        )


def bc_loss(actor: ActorNet, obs, cursor_target, press_target, toggle_target, cursor_weight: float,
            toggle_weight: float, map_feat=None, view=None, held_col: int = 0, stroke=None,
            stroke_cursor_weight: float = 1.0, how_weight: float = 1.0):
    """Cursor: MSE of the deterministic action (tanh(mean) * limit) to the
    expert delta. Press: when-head CE (press vs none) plus how-head BCE on
    press samples only (expert always clicks). Trail: toggle BCE toward the
    expert's toggle (trail_plan strokes: start, hold, release)."""
    out = actor.forward(obs, map_feat, view)
    cursor_pred = torch.tanh(out["cursor_loc"]) * CURSOR_COMPONENT_LIMIT
    per = ((cursor_pred - cursor_target) ** 2).mean(-1)
    stroke_mse = float(per[stroke].mean()) if stroke is not None and bool(stroke.any()) else 0.0
    if stroke is not None and stroke_cursor_weight != 1.0:
        # Stroke moves are ~1 world unit a tick against a 40-unit ceiling;
        # plain MSE on the ceiling fraction barely notices a 3-unit steering
        # error there, which is what sweeps a held trail into a wall.
        w = torch.where(stroke, torch.full_like(per, stroke_cursor_weight), torch.ones_like(per))
        cursor_loss = (w * per).mean()  # other states keep their old weight
    else:
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
    if out["hold_logit"] is not None:
        # Held-state head: the target is whether the teacher has a stroke
        # held after this tick (held now XOR its toggle) — every stroke tick
        # is a positive, no rare-event weighting needed.
        held_now = (obs[:, held_col] > 0.5).float()
        hold_target = (held_now + toggle_target) % 2
        toggle_loss = F.binary_cross_entropy_with_logits(out["hold_logit"], hold_target)
    else:
        # Toggles are ~2 ticks per stroke among thousands: weight the positives
        # so the head doesn't learn "never" (bc7's old target) from the imbalance.
        toggle_loss = F.binary_cross_entropy_with_logits(
            out["trail_toggle_logit"], toggle_target, pos_weight=torch.tensor(TOGGLE_POS_WEIGHT, device=obs.device)
        )
    total = cursor_weight * cursor_loss + when_loss + how_weight * how_loss + toggle_weight * toggle_loss
    return total, {"cursor": float(cursor_loss), "when": float(when_loss), "how": float(how_loss),
                   "toggle": float(toggle_loss), "cursor_stroke": stroke_mse}


def main():
    args = parse_args()
    be_nice(args.threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rl_env.TIMING_FEATURE_ENABLED = args.timing_feature
    rl_env.ATTACK_CLOCK_FEATURE_ENABLED = args.attack_clock
    clean = args.student_obs == "clean"
    rl_env.STROKE_GUIDE_FEATURE_ENABLED = not clean
    rl_env.SAFE_CLICK_HINT_ENABLED = not clean
    rl_env.LOOKAHEAD_FEATURE_ENABLED = clean
    rl_env.MAP_FEATURE_ENABLED = args.map
    use_view = clean if args.local_view is None else args.local_view
    rl_env.LOCAL_VIEW_ENABLED = use_view
    rl_env.WHISKER_FEATURE_ENABLED = args.whiskers
    rl_env.CARRIER_FEATURE_ENABLED = args.carrier
    device = "cuda" if torch.cuda.is_available() and args.device != "cpu" else "cpu"
    print(f"[bc] device {device}")

    pairs = find_chart_pairs(args.charts_dir)
    by_name = {os.path.basename(fp)[: -len(".frames.csv")]: (fp, ep) for fp, ep in pairs}
    holdout = [ChartData(*by_name[n]) for n in BALANCED_VALIDATION_CHARTS]
    train = [ChartData(fp, ep) for name, (fp, ep) in by_name.items() if name not in BALANCED_VALIDATION_CHARTS]
    print(f"[bc] {len(train)} train charts, {len(holdout)} validation charts")
    trail_eval = [c for c in train if c.name in set(filter(None, args.trail_eval.split(",")))]
    if args.trail_eval_dir:
        trail_eval += [ChartData(fp, ep) for fp, ep in find_chart_pairs(args.trail_eval_dir)]
    synth_pairs = [pair for d in args.synth_dir for pair in find_chart_pairs(d)]
    print(f"[bc] {len(synth_pairs)} generated levels, {len(trail_eval)} trail eval charts")
    chart_weights = []
    for chart in train:
        strokes = len(get_trail_plan(chart).strokes)
        has_keys = any(ev.get("hasKeyBinding") for ev in chart.events)
        chart_weights.append(args.stroke_chart_weight if strokes else args.key_chart_weight if has_keys else 1.0)
        if strokes:
            print(f"[bc] {chart.name}: {strokes} planned trail stroke(s)")

    def pick_chart():
        if synth_pairs and rng.random() < args.synth_share:
            return ChartData(*synth_pairs[rng.randrange(len(synth_pairs))])
        return rng.choices(train, weights=chart_weights)[0]

    stroke_charts = {c.name for c in train if len(get_trail_plan(c).strokes)}

    def new_env():
        """A whole real song is 20-50k ticks, 40-100 iterations of one env
        at 512 steps: the FIFO buffer then holds a few songs at a time and
        drifts with them (bc12-bc14 swung between evals). A random window
        of a stroke-free song mixes many; stroke charts and generated
        levels stay whole so no window cuts a stroke."""
        chart = pick_chart()
        lo, hi = args.real_window
        if hi > 0 and chart.name not in stroke_charts and any(chart is c for c in train):
            return make_env(chart, False, rng, lo, hi, True)
        return make_env(chart, True, rng, 0, 0, True)

    max_objects = train[0].max_objects
    fpo = obs_features_per_obj(train[0].features_per_obj)
    actor = ActorNet(max_objects, fpo, use_map=args.map, use_view=use_view, hold_head=args.hold_head)
    critic = CriticNet(max_objects, fpo)
    planner = None
    if args.map:
        from nav_map import PACKED_MAP_BYTES, MapPlanner, ValueIteration

        planner = MapPlanner(ValueIteration().to(device), device, iters=args.map_iters)
    if args.init:
        ckpt = migrate_pre_split_checkpoint(torch.load(args.init, map_location="cpu", weights_only=False))
        # A checkpoint without the map branch loads into a map actor with the
        # branch at its zero-output init.
        missing, unexpected = actor.load_state_dict(ckpt["actor_state_dict"], strict=False)
        new_branches = ("map_proj.", "view_enc.", "view_to_trunk.", "view_to_attack.")
        if unexpected or any(not m.startswith(new_branches) for m in missing):
            raise SystemExit(f"[bc] {args.init}: missing {missing}, unexpected {unexpected}")
        critic.load_state_dict(ckpt["critic_state_dict"])
        if planner is not None and "vin_state_dict" in ckpt:
            planner.vin.load_state_dict(ckpt["vin_state_dict"])
        print(f"[bc] initialized from {args.init}")
    if planner is not None and args.vin_init:
        planner.vin.load_state_dict(torch.load(args.vin_init, map_location="cpu")["vin_state_dict"])
        print(f"[bc] value iteration from {args.vin_init}")
    actor.to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.lr)
    vin_optimizer = torch.optim.Adam(planner.vin.parameters(), lr=args.vin_lr) if planner else None

    from nav_map import PACKED_VIEW_BYTES, unpack_views

    def map_features(env, obs):
        """Extra act() inputs for one env's current tick: the map branch's
        value window (--map) and the local view (--local-view)."""
        extra = {}
        if planner is not None:
            own = obs[env._per_step_dim - rl_env.OWN_STATE_DIM : env._per_step_dim]
            extra["map_feat"] = planner.features_or_zero([env.current_map], own[None, 0:2])[0]
        if use_view:
            extra["view"] = unpack_views(torch.from_numpy(env.current_view[None]).to(device))
        return extra
    expert = ScriptedExpert(use_keys=not args.no_expert_keys)

    envs, env_obs = [], []
    for _ in range(args.num_envs):
        envs.append(new_env())
        env_obs.append(envs[-1].reset())
    own0 = envs[0]._per_step_dim - rl_env.OWN_STATE_DIM  # history slot 0 own state
    trail_held_col = own0 + 2
    buffer = Buffer(args.buffer_size, envs[0]._per_step_dim * len(rl_env.HISTORY_STRIDES),
                    map_bytes=PACKED_MAP_BYTES if args.map else 0,
                    view_bytes=PACKED_VIEW_BYTES if use_view else 0)

    stroke_driver: list[str | None] = [None] * len(envs)
    best, since_best = float("-inf"), 0
    started = time.time()
    for it in range(1, args.iterations + 1):
        beta = args.beta_decay ** (it - 1)
        student_steps = 0
        for i in range(len(envs)):
            for _ in range(args.steps_per_env):
                env, obs = envs[i], env_obs[i]  # envs[i] is replaced when an episode ends
                target_delta, target_action = expert.act(env, obs)
                press, toggle = decode_action(target_action)
                buffer.add(
                    obs,
                    np.clip(target_delta, -CURSOR_COMPONENT_LIMIT + 1e-4, CURSOR_COMPONENT_LIMIT - 1e-4),
                    press,
                    toggle,
                    env.guide is not None and env.guide.active,
                    env.current_map,
                    env.current_view,
                )
                if env.guide is not None and env.guide.active:
                    if stroke_driver[i] is None:
                        stroke_driver[i] = "expert" if rng.random() < args.stroke_expert_share else "student"
                    expert_drives = stroke_driver[i] == "expert"
                else:
                    stroke_driver[i] = None
                    env_beta = max(beta, args.stroke_beta_min) if env.stroke_guide.plan.strokes else beta
                    expert_drives = rng.random() < env_beta
                if expert_drives:
                    delta, action = target_delta, target_action
                else:
                    with torch.no_grad():
                        act = actor.act(torch.from_numpy(obs).float().to(device), deterministic=True,
                                        **map_features(env, obs))
                    delta, action = act["cursor_delta"], act["action_type"]
                    student_steps += 1
                env_obs[i], *_ = env.step(delta, action)
                if env.done:
                    envs[i] = new_env()
                    env_obs[i] = envs[i].reset()
                    stroke_driver[i] = None

        if args.lr_final is not None:
            progress = (it - 1) / max(1, args.iterations - 1)
            lr = args.lr_final + 0.5 * (args.lr - args.lr_final) * (1 + np.cos(np.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = lr
        stats = {"cursor": 0.0, "when": 0.0, "how": 0.0, "toggle": 0.0, "cursor_stroke": 0.0}
        vin_mae = 0.0
        if planner is not None and args.vin_updates > 0:
            # The value iteration learns to compute the flood-fill distance
            # from the map itself; the policy then learns to use its output.
            # (With a pretrained --vin-init, --vin-updates 0 keeps it frozen
            # and its value-map cache alive.)
            for _ in range(args.vin_updates):
                if not buffer.map_pool:
                    break
                pick = np.random.randint(0, len(buffer.map_pool), size=args.vin_batch)
                vin_loss, mae = planner.distance_loss(np.stack([buffer.map_pool[i] for i in pick]))
                vin_optimizer.zero_grad()
                vin_loss.backward()
                torch.nn.utils.clip_grad_norm_(planner.vin.parameters(), 1.0)
                vin_optimizer.step()
                vin_mae += mae / args.vin_updates
            planner.clear()
        for _ in range(args.updates_per_iter):
            obs_b, cur_b, press_b, toggle_b, maps_b, views_b, stroke_b = buffer.sample(args.batch_size, args.stroke_share,
                                                                             args.toggle_share)
            obs_b, cur_b, press_b, toggle_b = (t.to(device) for t in (obs_b, cur_b, press_b, toggle_b))
            feat_b = None
            if planner is not None:
                own0_b = obs_b[:, own0 : own0 + 2].cpu().numpy()
                feat_b = planner.features_or_zero(maps_b, own0_b)
            view_b = unpack_views(torch.from_numpy(views_b).to(device)) if views_b is not None else None
            loss, parts = bc_loss(actor, obs_b, cur_b, press_b, toggle_b, args.cursor_weight, args.toggle_weight,
                                  feat_b, view_b, trail_held_col, stroke_b.to(device),
                                  args.stroke_cursor_weight, how_weight=args.how_weight)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            for k in stats:
                stats[k] += parts[k] / args.updates_per_iter
        press_rate = float(np.mean(buffer.press[: buffer.n] != PRESS_NONE))
        print(
            f"[bc] iter {it} beta={beta:.3f} student_steps={student_steps} buffer={buffer.n} "
            f"lr={optimizer.param_groups[0]['lr']:.1e} press_rate={press_rate:.4f} cursor_mse={stats['cursor']:.5f} stroke_cursor_mse={stats['cursor_stroke']:.5f} when_ce={stats['when']:.4f} "
            f"toggle_bce={stats['toggle']:.4f} held={float(np.mean(buffer.obs[: buffer.n, trail_held_col] > 0.5)):.4f} "
            + (f"vin_mae={vin_mae:.2f}cells " if planner is not None else "")
            + f"elapsed={time.time() - started:.0f}s",
            flush=True,
        )

        if it % args.eval_every == 0 or it == args.iterations:
            hits, notes, grades, val_macro = evaluate_holdout(actor, holdout, feat_fn=map_features)
            print(f"[bc eval {it}] hits={hits}/{notes} macro={val_macro:.2f}% {grades}", flush=True)
            macro = val_macro
            if trail_eval:
                t_hits, t_notes, t_grades, t_macro = evaluate_holdout(actor, trail_eval, feat_fn=map_features)
                print(f"[bc eval {it}] trail charts hits={t_hits}/{t_notes} macro={t_macro:.2f}% {t_grades}",
                      flush=True)
                n_val, n_trail = len(holdout), len(trail_eval)
                macro = (val_macro * n_val + t_macro * n_trail) / (n_val + n_trail)
            ckpt = {
                "actor_state_dict": {k: v.cpu() for k, v in actor.state_dict().items()},
                "use_map": args.map,
                "use_view": use_view,
                "hold_head": args.hold_head,
                **({"vin_state_dict": {k: v.cpu() for k, v in planner.vin.state_dict().items()},
                    "map_iters": args.map_iters} if planner is not None else {}),
                "critic_state_dict": critic.state_dict(),
                "max_objects": max_objects,
                "features_per_obj": fpo,
                "own_state_dim": rl_env.OWN_STATE_DIM,
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
