"""Compute and cache trail_plan.get_trail_plan() for every chart in the
given folders, in parallel (each validation plays a whole expert episode).

With --require-strokes (for scripts/generate_trail_levels.py output), a
chart whose stroke did not validate is removed — its frames/events/etc.
move to <dir>/_rejected — so training never sees a generated level the
expert can't clear.

    python training/precompute_trail_plans.py output output_synth --require-strokes output_synth
"""

from __future__ import annotations

import argparse
import glob
import os
import shutil
from multiprocessing import Pool

from data import ChartData
from niceness import be_nice
from trail_plan import build_plan, get_trail_plan


def _init_worker():
    import rl_env

    be_nice(1)
    # The teacher as train_bc --student-obs clean runs it.
    rl_env.STROKE_GUIDE_FEATURE_ENABLED = False
    rl_env.SAFE_CLICK_HINT_ENABLED = False


def _expert_faults(chart) -> int:
    """Wrong + Miss of one identity-view expert episode."""
    from bc_expert import ScriptedExpert
    from rl_env import TrailRLEnv

    env = TrailRLEnv(chart)
    obs = env.reset()
    expert = ScriptedExpert()
    while not env.done:
        obs, *_ = env.step(*expert.act(env, obs))
    return sum(e["judgment"] in ("Wrong", "Miss") for e in env.judge.log)


def _one(job: tuple[str, bool]) -> tuple[str, int, int, int]:
    frames, check_expert = job
    chart = ChartData(frames, frames[: -len(".frames.csv")] + ".events.json")
    candidates = len(build_plan(chart).strokes)
    kept = len(get_trail_plan(chart).strokes)
    faults = _expert_faults(chart) if check_expert else 0
    return frames, candidates, kept, faults


def main():
    p = argparse.ArgumentParser()
    p.add_argument("dirs", nargs="+")
    p.add_argument("--require-strokes", nargs="*", default=[],
                   help="folders whose charts must keep every candidate stroke (else rejected)")
    p.add_argument("--require-expert-clean", nargs="*", default=[],
                   help="folders whose charts the expert must clear with no Wrong/Miss (else rejected)")
    p.add_argument("--workers", type=int, default=3, help="kept small: the machine is shared")
    args = p.parse_args()
    frames = sorted(f for d in args.dirs for f in glob.glob(os.path.join(d, "*.frames.csv")))
    strict = {os.path.normpath(d) for d in args.require_strokes}
    clean = {os.path.normpath(d) for d in args.require_expert_clean}
    jobs = [(f, os.path.normpath(os.path.dirname(f)) in clean) for f in frames]
    rejected = 0
    be_nice(1)
    with Pool(args.workers, initializer=_init_worker) as pool:
        for path, candidates, kept, faults in pool.imap_unordered(_one, jobs):
            folder = os.path.normpath(os.path.dirname(path))
            bad = (folder in strict and (candidates == 0 or kept < candidates)) or faults > 0
            if candidates or bad:
                print(f"{os.path.basename(path)}: candidates={candidates} kept={kept} faults={faults}"
                      f"{'  REJECTED' if bad else ''}", flush=True)
            if bad:
                rejected += 1
                stem = path[: -len(".frames.csv")]
                target = os.path.join(folder, "_rejected")
                os.makedirs(target, exist_ok=True)
                for f in glob.glob(glob.escape(stem) + ".*"):
                    shutil.move(f, os.path.join(target, os.path.basename(f)))
    print(f"[precompute] {len(frames)} charts, {rejected} rejected")


if __name__ == "__main__":
    main()
