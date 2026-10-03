"""Packages replays of several models on ONE chart into a compare bundle
(`ybnote-compare/1`, gzip JSON) that ybnote-web's admin AI Replay panel
overlays on the same chart — multi-model comparison, see TRAIN_DIARY.md
2026-09-26 "規劃：ybnote-web 多模型 AI replay 比較". The web side can't run
PyTorch, so every model is run here (export_replay.export_entries) and the
bundle carries each one's action log plus the Python Judge's judgments
(`judgmentSource: "python-sim"`).

Bundle layout:
    {
      "format": "ybnote-compare/1",
      "createdAt": ISO8601, "aiCommit": ybnote-ai HEAD sha or null,
      "chart": {"title", "levelId", "fingerprint", "eventCount", "bounds",
                "encoding": "square-bounds-v1"},
      "entries": [{
        "id", "label", "stage", "order", "policy", "checkpoint",
        "migratedFrom", "color", "meta",
        "log": {"t": [ms], "x": [world], "y": [world],
                "f": [bit0 attack | bit1 trailHeld],
                "keys": {"<tick index>": ["d", ...]}},   # only ticks with keys
        "judgments": [{"t", "blockId", "judgment", "offset", "x", "y"}],
        "judgmentSource": "python-sim",
        "simSummary": {"Perfect", "Good", "Bad", "Miss", "Wrong"},
        "neural": {...}          # optional, rl entries with "neural": true
      }]
    }
One log element per chart step (config.DT_MS). judgments are in chart-time
order and include Miss and Wrong; blockId = chart.events[eventId]["id"]
(null for Wrong), x/y = the model's WORLD cursor at that tick (for hit FX),
offset = ms late (+) / early (-), null for Miss/Wrong.

--models is a JSON list of {label, stage, order, policy, weights, color}
(see compare_models.example.json). `weights` is resolved relative to the
models file's folder first, then the cwd; null for engineered. A missing
file or an incompatible checkpoint is skipped with a warning. An rl spec
with `"neural": true` also records its neuron activity for the web replay's
neuron view (neural_trace.py; ~1-2 MB per 4-minute chart). `"obs_flags":
{"timing_feature": bool, "attack_clock_feature": bool}` sets the observation
a checkpoint was trained on when the checkpoint doesn't record it.

Usage (from training/); the bundle lands in ../replays/ and --render also turns it
into a video (videos/<chart>.mp4, see ../scripts/renderVideos.js):
    python export_compare_bundle.py --chart "FALL FROM THE SKY PT. 2" --models compare_models.example.json
    python export_compare_bundle.py --frames ../output/x.frames.csv --events ../output/x.events.json \
        --models my_models.json --out compare_x.json.gz
"""

import argparse
import bisect
import datetime
import gzip
import hashlib
import json
import math
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import config
from data import ChartData
from export_replay import JUDGMENT_NAMES, IncompatibleCheckpoint, export_entries, judgment_counts

FORMAT = "ybnote-compare/1"
# encodeFrames.js computeBounds(): one square box around every event /
# collidable / track path, which 0..1 normalized coordinates are relative to.
ENCODING = "square-bounds-v1"

ROOT = Path(__file__).resolve().parent.parent
FLAG_ATTACK = 1
FLAG_TRAIL = 2


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chart", default=None,
                   help="chart name: reads ../output/<name>.frames.csv / .events.json")
    p.add_argument("--frames", default=None)
    p.add_argument("--events", default=None)
    p.add_argument("--models", required=True, help="JSON list of {label, stage, order, policy, weights, color}")
    p.add_argument("--out", default=None, help="default ../replays/compare_<chart>.json.gz")
    p.add_argument(
        "--render",
        nargs="?",
        const="",
        default=None,
        metavar="ARGS",
        help="after writing the bundle, render it to videos/<chart>.mp4 via ybnote-web "
        "(scripts/renderVideos.js); optional quoted extra args, e.g. --render \"--camera free --max-seconds 30\"",
    )
    p.add_argument("--connectome", default=None, help="only for policy neural entries")
    p.add_argument("--roles", default=None, help="only for policy neural entries")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument(
        "--no-timing-feature",
        action="store_true",
        help="zero the env's hit-timing object column, matching checkpoints trained with it off "
        "(train_rl.py --no-timing-feature, train_bc.py)",
    )
    args = p.parse_args()
    if args.chart:
        args.frames = args.frames or str(ROOT / "output" / f"{args.chart}.frames.csv")
        args.events = args.events or str(ROOT / "output" / f"{args.chart}.events.json")
    if not (args.frames and args.events):
        p.error("give --chart, or both --frames and --events")
    return args


def chart_fingerprint(events: list[dict]) -> str:
    """Chart identity that ybnote-web recomputes from the game's resolved
    interactive events, to check a bundle belongs to the loaded level.

    Exactly: take every event of events.json's `events` list (one per
    interactive note: `time` in chart ms, `id` = target object id such as
    "noteblock-3zokm9s9"); build pairs [round(time), id] where round is
    half-up to the nearest integer, floor(time + 0.5) — the same as JS
    Math.round for these non-negative times (NOT Python's half-to-even
    round(); exact .5 ms times do occur, e.g. an eighth at 160 BPM =
    187.5ms); sort the pairs
    by (rounded time, id) ascending (id compared as a plain string, by code
    point); serialize with json.dumps(pairs, separators=(",", ":"),
    ensure_ascii=False) — i.e. no whitespace at all, non-ASCII ids kept as
    literal characters, e.g. `[[1404,"drumBlock-ypj5edif"],[1606,"..."]]`;
    SHA-1 of that string's UTF-8 bytes, as 40 lowercase hex characters."""
    pairs = sorted(([math.floor(ev["time"] + 0.5), ev["id"]] for ev in events), key=lambda p: (p[0], p[1]))
    text = json.dumps(pairs, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def level_identity(chart: ChartData, events_path: str) -> tuple[str, str | None]:
    """(title, levelId). Prefers events.json's `source` block (written by
    encodeFrames.js from now on), else the TITLE header of the source
    input/<name>.yblevel (read the same way as scripts/parseYblevel.js).
    A .yblevel has no level id — ids only exist in the ybnote-web database
    — so levelId is null unless the header ever gains one."""
    with open(events_path, "r", encoding="utf-8") as f:
        source = json.load(f).get("source") or {}
    title, level_id = source.get("title"), source.get("levelId")
    if title is None or level_id is None:
        header = _yblevel_header(ROOT / "input" / f"{chart.name}.yblevel")
        title = title or header.get("TITLE")
        level_id = level_id or header.get("LEVEL_ID") or header.get("ID")
    return (title or chart.name), (level_id or None)


def _yblevel_header(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        with zipfile.ZipFile(path) as zf:
            text = zf.read("level.txt").decode("utf-8")
    except (zipfile.BadZipFile, KeyError):
        return {}
    header_text = text.split("[JSON]\n", 1)[0]
    header = {}
    for line in header_text.split("\n"):
        if ":" in line and line.strip():
            key, value = line.split(":", 1)
            header[key] = value.strip("\r")
    return header


def git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")
    return slug or "model"


def _num(value: float):
    """Integral floats as ints — most tick times are whole ms, and this
    keeps the columnar arrays short."""
    value = float(value)
    return int(value) if value.is_integer() else round(value, 3)


def build_log(entries: list[dict]) -> dict:
    log = {"t": [], "x": [], "y": [], "f": [], "keys": {}}
    for i, e in enumerate(entries):
        log["t"].append(_num(e["t"]))
        log["x"].append(_num(round(e["cursorX"], 2)))
        log["y"].append(_num(round(e["cursorY"], 2)))
        log["f"].append((FLAG_ATTACK if e["attack"] else 0) | (FLAG_TRAIL if e["trailHeld"] else 0))
        if e["keybindsFired"]:
            log["keys"][str(i)] = list(e["keybindsFired"])
    return log


def build_judgments(chart: ChartData, entries: list[dict], judge_log: list[dict]) -> list[dict]:
    times = [float(t) for t in chart.t_ms]
    out = []
    for rec in sorted(judge_log, key=lambda r: r["time"]):  # stable: Judge order within a tick
        # Judge times are tick times, except end-of-chart Misses (after the
        # last tick) — those take the last tick's cursor.
        tick = max(0, bisect.bisect_right(times, float(rec["time"])) - 1)
        uid = rec.get("eventId")
        offset = rec.get("offset")
        out.append({
            "t": _num(rec["time"]),
            "blockId": chart.events[uid]["id"] if uid is not None else None,
            "judgment": rec["judgment"],
            "offset": round(float(offset), 2) if offset is not None else None,
            "x": entries[tick]["cursorX"],
            "y": entries[tick]["cursorY"],
        })
    return out


# Colors for expanded `runs` entries without their own color.
RUN_PALETTE = [
    "#ef4444", "#f97316", "#eab308", "#22c55e", "#14b8a6", "#3b82f6",
    "#8b5cf6", "#ec4899", "#a3e635", "#06b6d4", "#f43f5e", "#6366f1",
]


def expand_runs(models: list[dict], base_seed: int) -> list[dict]:
    """A model spec with `"runs": N` becomes N sampled entries of the same
    checkpoint ("<label> #1".."#N", seeds `seed`..`seed+N-1` defaulting to
    --seed, colors from RUN_PALETTE unless `colors` lists them). Deterministic
    playback is identical every time, so several takes of one model only
    differ when sampled."""
    out = []
    for spec in models:
        runs = int(spec.get("runs", 1))
        if runs <= 1:
            out.append(spec)
            continue
        seed0 = int(spec.get("seed", base_seed))
        colors = spec.get("colors") or RUN_PALETTE
        for k in range(runs):
            out.append({
                **{key: v for key, v in spec.items() if key not in ("runs", "colors")},
                "label": f"{spec['label']} #{k + 1}",
                "order": spec.get("order", 0) + k / 1000,
                "sample": True,
                "seed": seed0 + k,
                "color": colors[k % len(colors)],
            })
    return out


def resolve_weights(weights: str | None, models_dir: Path) -> Path | None:
    if not weights:
        return None
    for base in (models_dir, Path.cwd()):
        candidate = (base / weights).resolve()
        if candidate.exists():
            return candidate
    return models_dir / weights  # nonexistent; caller warns


def main():
    # Chart names/titles include emoji and CJK; a cp950 Windows console
    # would otherwise raise on print.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    args = parse_args()
    if args.no_timing_feature:
        import rl_env
        rl_env.TIMING_FEATURE_ENABLED = False
    chart = ChartData(args.frames, args.events)
    title, level_id = level_identity(chart, args.events)
    models_path = Path(args.models)
    models = json.loads(models_path.read_text(encoding="utf-8"))
    models = sorted(expand_runs(models, args.seed), key=lambda m: m.get("order", 0))

    bundle_entries, used_ids, skipped = [], set(), []
    for spec in models:
        label, policy = spec["label"], spec["policy"]
        weights_path = resolve_weights(spec.get("weights"), models_path.parent)
        if weights_path is not None and not weights_path.exists():
            print(f"[compare] WARNING {label}: {spec['weights']} not found, skipped")
            skipped.append((label, "file not found"))
            continue
        print(f"[compare] {label} ({policy}{', ' + weights_path.name if weights_path else ''})")
        try:
            entries, judge_log, meta = export_entries(
                policy, str(weights_path) if weights_path else None, chart,
                connectome=args.connectome, roles=args.roles, seed=spec.get("seed", args.seed),
                sample=bool(spec.get("sample", False)), trace=bool(spec.get("neural", False)), obs_flags=spec.get("obs_flags"), verbose=False,
            )
        except IncompatibleCheckpoint as err:
            print(f"[compare] WARNING {label}: incompatible, skipped - {err}")
            skipped.append((label, str(err)))
            continue

        entry_id = slugify(label)
        n = 2
        while entry_id in used_ids:
            entry_id, n = f"{slugify(label)}-{n}", n + 1
        used_ids.add(entry_id)

        judgments = build_judgments(chart, entries, judge_log)
        summary = judgment_counts(judge_log)
        entry_meta = {k: meta[k] for k in ("iteration", "best_holdout_score", "best_holdout_hits", "note")
                      if meta.get(k) is not None}
        if spec.get("sample"):
            entry_meta["sampled"] = True
            entry_meta["seed"] = spec.get("seed", args.seed)
        bundle_entries.append({
            "id": entry_id,
            "label": label,
            "stage": spec.get("stage"),
            "order": spec.get("order", 0),
            "policy": policy,
            "checkpoint": weights_path.name if weights_path else None,
            "migratedFrom": meta.get("migratedFrom"),
            "color": spec.get("color"),
            "meta": entry_meta,
            "log": build_log(entries),
            "judgments": judgments,
            "judgmentSource": "python-sim",
            "simSummary": {name: summary.get(name, 0) for name in JUDGMENT_NAMES},
        })
        if meta.get("neural"):
            bundle_entries[-1]["neural"] = meta["neural"]
        print(f"[compare]   {summary}")

    bundle = {
        "format": FORMAT,
        "createdAt": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "aiCommit": git_head(),
        "chart": {
            "title": title,
            "levelId": level_id,
            "fingerprint": chart_fingerprint(chart.events),
            "eventCount": len(chart.events),
            "bounds": chart.bounds,
            "encoding": ENCODING,
        },
        "entries": bundle_entries,
    }
    out_path = Path(args.out) if args.out else ROOT / "replays" / f"compare_{chart.name}.json.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, separators=(",", ":"))
    print(f"Wrote {out_path} ({out_path.stat().st_size / 1024:.0f} KB): {len(bundle_entries)} entries"
          + (f", skipped {len(skipped)}: " + "; ".join(label for label, _ in skipped) if skipped else ""))
    if args.render is not None:
        import shlex
        cmd = ["node", str(ROOT / "scripts" / "renderVideos.js"), "--bundle", str(out_path), *shlex.split(args.render)]
        print("[render] " + " ".join(cmd))
        sys.exit(subprocess.call(cmd, cwd=ROOT))


if __name__ == "__main__":
    main()
