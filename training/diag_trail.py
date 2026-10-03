"""Where a checkpoint's trail-chart Wrongs and Misses come from.

For each chart: every Wrong classified as a sweep (trail held, no click)
or a click, and whether it falls inside a planned stroke's window; per
planned stroke, whether the student held the trail at its start note and
how much of the stroke it kept held.

  python training/diag_trail.py training/rl_policy_bc12.pt output_synth_test2 [--charts output --names 迷宮🗣️🔥]
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import config  # noqa: E402
import rl_env  # noqa: E402
from data import ChartData  # noqa: E402
from train_rl import find_chart_pairs  # noqa: E402
from niceness import be_nice  # noqa: E402
from rl_env import TrailRLEnv, apply_obs_flags  # noqa: E402
from rl_policy import ActorNet, migrate_pre_split_checkpoint  # noqa: E402
from trail_plan import get_trail_plan  # noqa: E402


def load_actor(path: str, device: str) -> ActorNet:
    ckpt = migrate_pre_split_checkpoint(torch.load(path, map_location="cpu", weights_only=False))
    actor = ActorNet(ckpt["max_objects"], ckpt["features_per_obj"], hidden=ckpt["hidden"],
                     use_map=bool(ckpt.get("use_map")), use_view=bool(ckpt.get("use_view")),
                     hold_head=bool(ckpt.get("hold_head")))
    actor.load_state_dict(ckpt["actor_state_dict"])
    apply_obs_flags(ckpt.get("obs_flags", {}))
    return actor.to(device).eval()


def _tag_wrongs(env: TrailRLEnv) -> None:
    """Annotate each Wrong in the judge log with the object that caused it
    and how far that object's next note is (ms; None = no later note)."""
    judge = env.judge
    orig = judge._resolve_target_action
    note_times: dict[str, list[float]] = {}
    for ev in env.chart.events:
        note_times.setdefault(ev["id"], []).append(float(ev["time"]))

    def wrapped(t_ms, target):
        n = len(judge.log)
        r = orig(t_ms, target)
        for e in judge.log[n:]:
            if e["judgment"] == "Wrong":
                later = [x for x in note_times.get(target["id"], []) if x >= t_ms - config.HIT_WINDOW_MS]
                e["obj"] = target["id"]
                e["obj_type"] = target.get("type")
                e["to_note"] = round(min(later) - t_ms) if later else None
        return r

    judge._resolve_target_action = wrapped


def rollout(actor: ActorNet, charts: list[ChartData], device: str, teacher_gap: bool = False, ema: float = 0.0):
    """Batched deterministic rollout. -> per chart (env, held[T], clicked[T],
    gap[T]): gap = world-unit distance between the student's cursor move and
    the teacher's label for the same state on stroke ticks (NaN elsewhere,
    or everywhere without teacher_gap)."""
    from bc_expert import ScriptedExpert
    from nav_map import unpack_views

    expert = ScriptedExpert()
    envs = [TrailRLEnv(c, window=None) for c in charts]
    for env in envs:
        env.compute_guide = teacher_gap
    obs = [env.reset() for env in envs]
    for env in envs:
        _tag_wrongs(env)
    held = [[] for _ in envs]
    clicked = [[] for _ in envs]
    gaps = [[] for _ in envs]
    # ema > 0: while held, move by an exponential average of the policy's
    # moves (an inference-time test of whether stroke errors are jitter).
    smooth = [None] * len(envs)
    with torch.no_grad():
        while True:
            live = [i for i, e in enumerate(envs) if not e.done]
            if not live:
                break
            x = torch.from_numpy(np.stack([obs[i] for i in live])).float().to(device)
            extra = {}
            if actor.use_view:
                extra["view"] = unpack_views(torch.from_numpy(np.stack([envs[i].current_view for i in live]))
                                             .to(device))
            cursors, actions = actor.act_batch(x, **extra)
            for row, i in enumerate(live):
                env = envs[i]
                gap = float("nan")
                if teacher_gap and env.guide is not None and env.guide.active:
                    target, _ = expert.act(env, obs[i])
                    d = np.subtract(target, cursors[row]) * env.reach * env.chart.world_span
                    gap = float(np.hypot(*d))
                gaps[i].append(gap)
                move = np.asarray(cursors[row], dtype=np.float64)
                if ema > 0 and env.trail_held:
                    smooth[i] = move if smooth[i] is None else ema * smooth[i] + (1 - ema) * move
                    move = smooth[i]
                else:
                    smooth[i] = None
                obs[i], *_ = envs[i].step((float(move[0]), float(move[1])), int(actions[row]))
                held[i].append(envs[i].last_action["trail_held"])
                clicked[i].append(envs[i].last_action["attack_fired"] or bool(envs[i].last_action["keybind_fired"]))
    return [(env, np.array(h), np.array(c), np.array(g)) for env, h, c, g in zip(envs, held, clicked, gaps)]


def report(chart: ChartData, env: TrailRLEnv, held: np.ndarray, clicked: np.ndarray, gap: np.ndarray) -> dict:
    t = np.asarray(chart.t_ms[: len(held)], dtype=np.float64)
    plan = get_trail_plan(chart)
    windows = [(s.t_start - config.HIT_WINDOW_MS, s.t_end + config.HIT_WINDOW_MS) for s in plan.strokes]
    grade = {e["eventId"]: e["judgment"] for e in env.judge.log if "eventId" in e}
    out = {"sweep_in": 0, "sweep_out": 0, "click_in": 0, "click_out": 0, "other": 0,
           "miss": sum(e["judgment"] == "Miss" for e in env.judge.log), "w_early_note": 0, "w_obstacle": 0,
           "strokes": []}
    for e in env.judge.log:
        if e["judgment"] != "Wrong":
            continue
        k = int(np.clip(np.searchsorted(t, e["time"]), 0, len(t) - 1))
        inside = any(a <= e["time"] <= b for a, b in windows)
        kind = "click" if clicked[k] else ("sweep" if held[k] else "other")
        out[kind if kind == "other" else f"{kind}_{'in' if inside else 'out'}"] += 1
        # Entered a note's object before its window (early) or an object with
        # no note soon (an obstacle).
        soon = e.get("to_note") is not None and e["to_note"] < 1500
        out["w_early_note" if soon else "w_obstacle"] += 1
        if not soon and np.isfinite(gap[max(0, k - 20):k + 1]).any():
            out.setdefault("_pre_wrong_gap", []).append(float(np.nanmax(gap[max(0, k - 20):k + 1])))
    for s in plan.strokes:
        k0 = int(np.searchsorted(t, s.t_start))
        k1 = int(np.searchsorted(t, s.t_end))
        start_held = bool(held[max(0, k0 - 2): k0 + 4].any())
        coverage = float(held[k0:k1 + 1].mean()) if k1 > k0 else float(held[k0])
        hits = [grade.get(u, "?") for u in [s.start_uid] + s.hit_uids]
        # Each Wrong's place in the stroke, ms after its start.
        wrong_at = [round(e["time"] - s.t_start) for e in env.judge.log
                    if e["judgment"] == "Wrong" and s.t_start - config.HIT_WINDOW_MS <= e["time"]
                    <= s.t_end + config.HIT_WINDOW_MS]
        out["strokes"].append({"t": s.t_start, "start_held": start_held, "coverage": coverage,
                               "hits": sum(h in ("Perfect", "Good", "Bad") for h in hits), "notes": len(hits),
                               "len": round(s.t_end - s.t_start), "wrong_at": wrong_at})
    # Held while no stroke is planned (a stroke the plan never asked for).
    planned = np.zeros(len(held), dtype=bool)
    for a, b in windows:
        planned |= (t >= a) & (t <= b)
    out["held_unplanned_ms"] = float((held & ~planned).sum() * config.DT_MS)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("dirs", nargs="*", default=[])
    p.add_argument("--charts", default="output", help="real charts folder for --names")
    p.add_argument("--max-per-dir", type=int, default=None, help="first N charts of each folder")
    p.add_argument("--teacher-gap", action="store_true",
                   help="also run the teacher on the student's stroke states and report the action gap")
    p.add_argument("--names", default="只因為你那渴望自由的心臟🫀,迷宮🗣️🔥")
    p.add_argument("--match", default="", help="only charts from the folders whose file name contains this")
    p.add_argument("--ema", type=float, default=0.0, help="smooth held-stroke moves (inference-time test)")
    args = p.parse_args()
    be_nice(4)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    actor = load_actor(args.checkpoint, device)
    pairs = [pr for d in args.dirs for pr in sorted(find_chart_pairs(d))[: args.max_per_dir]
             if args.match in os.path.basename(pr[0])]
    names = set(filter(None, args.names.split(",")))
    pairs += [pr for pr in find_chart_pairs(args.charts) if os.path.basename(pr[0])[: -len(".frames.csv")] in names]
    charts = [ChartData(*pr) for pr in pairs]
    totals: dict[str, float] = {}
    n_strokes = start_ok = clean = 0
    all_gaps, pre_wrong = [], []
    for chart, (env, held, clicked, gap) in zip(charts, rollout(actor, charts, device, args.teacher_gap, args.ema)):
        r = report(chart, env, held, clicked, gap)
        strokes = r.pop("strokes")
        pre_wrong += r.pop("_pre_wrong_gap", [])
        all_gaps += list(gap[np.isfinite(gap)])
        for k, v in r.items():
            totals[k] = totals.get(k, 0) + v
        n_strokes += len(strokes)
        start_ok += sum(s["start_held"] for s in strokes)
        clean += sum(s["hits"] == s["notes"] for s in strokes)
        desc = " ".join(f"[{s['t']:.0f}ms start={'Y' if s['start_held'] else 'n'} cov={s['coverage']:.2f} "
                        f"hit={s['hits']}/{s['notes']} len={s['len']} W@{s['wrong_at']}]" for s in strokes)
        print(f"{chart.name}: " + " ".join(f"{k}={v:g}" for k, v in r.items()) + f" {desc}", flush=True)
    print(f"[total] {len(charts)} charts, strokes started {start_ok}/{n_strokes}, all-hit {clean}/{n_strokes}, "
          + " ".join(f"{k}={v:g}" for k, v in totals.items()))
    if all_gaps:
        g = np.array(all_gaps)
        print(f"[gap] stroke ticks {len(g)}: move gap to teacher (world units/tick) mean={g.mean():.2f} "
              f"p50={np.median(g):.2f} p90={np.percentile(g, 90):.2f} p99={np.percentile(g, 99):.2f}; "
              f"max over 100ms before an obstacle Wrong: mean={np.mean(pre_wrong) if pre_wrong else float('nan'):.2f} "
              f"(n={len(pre_wrong)})")


if __name__ == "__main__":
    main()
