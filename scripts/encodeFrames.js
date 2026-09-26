// Converts a .yblevel chart into a frame-by-frame feature matrix suitable for
// driving the input layer of the FlyWire-connectome SNN (LIF model).
//
// Usage:
//   node scripts/encodeFrames.js --input input/mysong.yblevel
//   node scripts/encodeFrames.js --input input            (batch: every .yblevel in the folder)
//   node scripts/encodeFrames.js --input input/mysong.yblevel --dt 5 --max-objects 8
//
// Output (written to output/<levelName>.*):
//   *.frames.json  frame-by-frame state, full precision, variable object list per frame
//   *.frames.csv   fixed-width flattened matrix, ready to feed a tensor loader
//   *.events.json  resolved per-note timing/spatial data + judgment windows,
//                  for the reward/STDP side (see docs/ENCODING_DESIGN.md)
//
// All timing constants below are copied from ybnote-web's
// src/config/gameTiming.ts and src/config/scoring.ts so this stays in sync
// with the actual judgment logic. If those files change, update here too.

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { parseYblevel } from "./parseYblevel.js";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// Shared with training/config.py — the fixed vocabulary of keys a note's
// keyBinding can one-hot encode into. A note whose key isn't in this list
// still gets its `keybind` presence flag set (so the model knows *something*
// needs pressing) but no one-hot channel fires — see TRAIN_DIARY.md
// 2026-09-24 "keybind support" entry.
const KEY_VOCAB = JSON.parse(fs.readFileSync(path.join(ROOT, "keyVocab.json"), "utf-8"));

// ---- ybnote-web timing/scoring constants (src/config/gameTiming.ts, src/config/scoring.ts) ----
const APPROACH_TIME_MS = 800;
const PERFECT_WINDOW_MS = 50;
const GOOD_WINDOW_MS = 100;
const HIT_WINDOW_MS = 200;
const JUDGMENT_POINTS = { Perfect: 300, Good: 200, Bad: 100, Miss: 0 };
const JUDGMENT_ACCURACY_WEIGHT = {
  Perfect: 1,
  Good: 1 - PERFECT_WINDOW_MS / HIT_WINDOW_MS, // 0.75
  Bad: 1 - GOOD_WINDOW_MS / HIT_WINDOW_MS, // 0.5
  Miss: 0,
};
const WRONG_PENALTY = 50;
const WRONG_ACCURACY_PENALTY = 0.25;

function parseArgs(argv) {
  const args = { dt: 5, maxObjects: 8, input: null, outDir: null };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--input") args.input = argv[++i];
    else if (a === "--dt") args.dt = Number(argv[++i]);
    else if (a === "--max-objects") args.maxObjects = Number(argv[++i]);
    else if (a === "--out") args.outDir = argv[++i];
  }
  if (!args.input) {
    throw new Error(
      "Missing --input <file.yblevel | folder>. See header comment for usage.",
    );
  }
  return args;
}

// ---- Track motion (ported subset of ybnote-web's src/utils/track/evaluateTrack.ts /
// spline.ts / channelModel.ts) — enough to place a track-CARRIED block/
// groupRect at its live position when its note is due, for AUTOPLAY tracks
// only. A non-autoplay (manually triggered) track's start time depends on
// the full trigger/collision timeline (src/utils/track/globalSimulation.ts,
// ~400 lines of runner/collision simulation) — out of scope here; such a
// carried object falls back to its rest x/y with a logged warning rather
// than silently using a wrong position. See TRAIN_DIARY.md 2026-09-24.

function getBezierPoint(t, p0, p1, p2, p3) {
  const cX = 3 * (p1.x - p0.x);
  const bX = 3 * (p2.x - p1.x) - cX;
  const aX = p3.x - p0.x - cX - bX;
  const cY = 3 * (p1.y - p0.y);
  const bY = 3 * (p2.y - p1.y) - cY;
  const aY = p3.y - p0.y - cY - bY;
  return {
    x: aX * t ** 3 + bX * t ** 2 + cX * t + p0.x,
    y: aY * t ** 3 + bY * t ** 2 + cY * t + p0.y,
  };
}

function getControlPoints(prev, curr, next) {
  const smoothFactor = 0.35;
  const d1 = Math.hypot(curr.x - prev.x, curr.y - prev.y);
  const d2 = Math.hypot(next.x - curr.x, next.y - curr.y);
  let vx = 0, vy = 0;
  if (d1 + d2 > 0) {
    vx = (next.x - prev.x) / (d1 + d2);
    vy = (next.y - prev.y) / (d1 + d2);
  }
  return {
    controlIn: { x: curr.x - vx * d1 * smoothFactor, y: curr.y - vy * d1 * smoothFactor },
    controlOut: { x: curr.x + vx * d2 * smoothFactor, y: curr.y + vy * d2 * smoothFactor },
  };
}

function pointsEqual(a, b, epsilon = 1e-6) {
  return Math.hypot(a.x - b.x, a.y - b.y) <= epsilon;
}

function findSurroundingKeyframes(keyframes, t) {
  const n = keyframes.length;
  if (n === 0) return null;
  if (n === 1 || t <= keyframes[0].t) return { prevIdx: 0, nextIdx: 0, localT: 0 };
  if (t >= keyframes[n - 1].t) return { prevIdx: n - 1, nextIdx: n - 1, localT: 0 };
  let lo = 0, hi = n - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (keyframes[mid].t <= t) lo = mid;
    else hi = mid - 1;
  }
  const prevIdx = lo;
  const nextIdx = Math.min(prevIdx + 1, n - 1);
  const span = keyframes[nextIdx].t - keyframes[prevIdx].t;
  const localT = span > 0 ? (t - keyframes[prevIdx].t) / span : 0;
  return { prevIdx, nextIdx, localT };
}

function sampleChannel(channel, t) {
  if (!channel) return undefined;
  const surround = findSurroundingKeyframes(channel.keyframes, t);
  if (!surround) return undefined;
  const { prevIdx, nextIdx, localT } = surround;
  const a = channel.keyframes[prevIdx].value;
  const b = channel.keyframes[nextIdx].value;
  return a + (b - a) * localT;
}

function channelDuration(channel) {
  if (!channel || channel.keyframes.length === 0) return 0;
  return channel.keyframes[channel.keyframes.length - 1].t;
}

/** t: elapsed seconds since the track started (for an autoplay track, that's
 *  chart time 0; for a non-autoplay track, since whichever trigger segment
 *  it's in — see computeTrackSegments/resolveLivePosition). Returns {x, y,
 *  finished} — finished mirrors the real evaluateTrackAtTime's non-looping
 *  "reached its own end" signal (needed by computeTrackSegments to know
 *  when a non-looping runner auto-stops). Drops live-drag override plumbing
 *  (irrelevant offline) that the real evaluateTrackAtTime also carries. */
function evaluateTrackAtTime(track, t) {
  const posChannel = track.channels.position;
  const n = posChannel.keyframes.length;
  if (n === 0) return { x: 0, y: 0, finished: true };
  if (n === 1) return { x: posChannel.keyframes[0].value.x, y: posChannel.keyframes[0].value.y, finished: false };

  const isCircular = track.loop === true;
  const isRestart = track.loop === "restart";
  const lastT = posChannel.keyframes[n - 1].t;
  const defaultSegDuration = 60 / (track.bpm || 120);
  const otherChannelsLastT = Math.max(
    channelDuration(track.channels.rotation),
    channelDuration(track.channels.scale),
    channelDuration(track.channels.alpha),
  );
  const totalDuration = isCircular ? lastT + defaultSegDuration : Math.max(lastT, otherChannelsLastT);

  let time = t;
  let finished = false;
  if (time >= totalDuration) {
    if (isCircular || isRestart) {
      time = totalDuration > 0 ? time % totalDuration : 0;
    } else {
      time = totalDuration;
      finished = true;
    }
  }
  if (time < 0) time = 0;

  let idxPrev, idxNext, localT, p1, p2;
  if (isCircular && time > lastT) {
    idxPrev = n - 1;
    idxNext = 0;
    localT = defaultSegDuration > 0 ? (time - lastT) / defaultSegDuration : 0;
    p1 = posChannel.keyframes[idxPrev].value;
    p2 = posChannel.keyframes[idxNext].value;
  } else {
    const surround = findSurroundingKeyframes(posChannel.keyframes, time);
    idxPrev = surround.prevIdx;
    idxNext = surround.nextIdx;
    localT = surround.localT;
    p1 = posChannel.keyframes[idxPrev].value;
    p2 = posChannel.keyframes[idxNext].value;
  }

  const visuallyClosed =
    n > 2 && !isCircular && pointsEqual(posChannel.keyframes[0].value, posChannel.keyframes[n - 1].value);
  const useClosedTangents = isCircular || visuallyClosed;
  const effectiveN = visuallyClosed ? n - 1 : n;
  const canon = (idx) => (visuallyClosed && idx === n - 1 ? 0 : idx);

  const prevNeighborIdx = useClosedTangents
    ? (canon(idxPrev) - 1 + effectiveN) % effectiveN
    : idxPrev > 0 ? idxPrev - 1 : idxPrev;
  const nextNeighborIdx = useClosedTangents
    ? (canon(idxNext) + 1) % effectiveN
    : idxNext < n - 1 ? idxNext + 1 : idxNext;
  const prevNeighbor = posChannel.keyframes[prevNeighborIdx].value;
  const nextNeighbor = posChannel.keyframes[nextNeighborIdx].value;

  const cp1 = getControlPoints(prevNeighbor, p1, p2).controlOut;
  const cp2 = getControlPoints(p1, p2, nextNeighbor).controlIn;
  const point = getBezierPoint(localT, p1, cp1, cp2, p2);
  return { x: point.x, y: point.y, finished };
}

// ---- Legacy track format migration (ported subset of ybnote-web's
// src/utils/track/trackMigration.ts) — .yblevel files saved before the
// channel-based keyframe system stored a track's path as nodes[]/
// segmentDurations[] instead of channels. Migrating these once at load time
// (matching what the real app does on open) means resolveLivePosition can
// treat every track uniformly instead of falling back to the rest position
// for anything a legacy track carries. See TRAIN_DIARY.md 2026-09-24 "old
// format track migration".

function isLegacyTrack(t) {
  if (!t || typeof t !== "object") return false;
  return !("channels" in t) || Array.isArray(t.nodes);
}

function makeKeyframe(t, value) {
  return { t, value };
}

// Mirrors channelModel.ts's capKeyframeTies: collapses a run of more than 2
// same-instant keyframes (a zero-duration segment) down to just its
// first/last member, matching the authored instant-jump representation.
const KEYFRAME_TIE_EPSILON_SECONDS = 0.001;
function capKeyframeTies(channel) {
  const kfs = channel.keyframes;
  let i = 0;
  while (i < kfs.length) {
    let j = i + 1;
    while (j < kfs.length && kfs[j].t - kfs[i].t < KEYFRAME_TIE_EPSILON_SECONDS) j++;
    const runLength = j - i;
    if (runLength > 2) {
      kfs.splice(i + 1, runLength - 2);
      j = i + 2;
    }
    i = j;
  }
}

function resolveLegacyCoord(value, fallback) {
  return typeof value === "number" && Number.isFinite(value) ? value : fallback;
}

/** Deterministically converts a legacy nodes[]+segmentDurations[] track into
 *  the canonical channel-based shape, mirroring trackMigration.ts's
 *  migrateTrackToChannels's segment-walk timing exactly (each node's t is
 *  the cumulative sum of segmentDurations[i-1] ?? 60/bpm) so a migrated
 *  track evaluates identically to how it used to play. */
function migrateTrackToChannels(legacy) {
  const bpm = typeof legacy.bpm === "number" && Number.isFinite(legacy.bpm) && legacy.bpm > 0 ? legacy.bpm : 120;
  const nodes = Array.isArray(legacy.nodes) ? legacy.nodes : [];
  const rawDurations = Array.isArray(legacy.segmentDurations) ? legacy.segmentDurations : [];

  const position = { property: "position", keyframes: [] };
  const rotation = { property: "rotation", keyframes: [] };
  const scale = { property: "scale", keyframes: [] };

  let cumulativeT = 0;
  let prevX = 0, prevY = 0, prevRotation = 0, prevScale = 1;

  nodes.forEach((node, i) => {
    if (i > 0) {
      const segDuration = rawDurations[i - 1] ?? 60 / bpm;
      const safeDuration =
        typeof segDuration === "number" && Number.isFinite(segDuration) && segDuration >= 0
          ? segDuration
          : 60 / bpm;
      cumulativeT += safeDuration;
    }

    const x = resolveLegacyCoord(node.x, prevX);
    const y = resolveLegacyCoord(node.y, prevY);
    position.keyframes.push(makeKeyframe(cumulativeT, { x, y }));
    prevX = x;
    prevY = y;

    const nodeRotation =
      typeof node.rotation === "number" && Number.isFinite(node.rotation) ? node.rotation : prevRotation;
    const nodeScale = typeof node.scale === "number" && Number.isFinite(node.scale) ? node.scale : prevScale;
    if (nodeRotation !== prevRotation) rotation.keyframes.push(makeKeyframe(cumulativeT, nodeRotation));
    if (nodeScale !== prevScale) scale.keyframes.push(makeKeyframe(cumulativeT, nodeScale));
    prevRotation = nodeRotation;
    prevScale = nodeScale;
  });

  capKeyframeTies(position);
  capKeyframeTies(rotation);
  capKeyframeTies(scale);

  // A channel with keyframes but none at t=0 has no left boundary to clamp
  // to — backfill an identity keyframe there, matching trackMigration.ts.
  if (rotation.keyframes.length > 0 && rotation.keyframes[0].t !== 0) rotation.keyframes.unshift(makeKeyframe(0, 0));
  if (scale.keyframes.length > 0 && scale.keyframes[0].t !== 0) scale.keyframes.unshift(makeKeyframe(0, 1));

  const channels = {
    position,
    ...(rotation.keyframes.length > 0 ? { rotation } : {}),
    ...(scale.keyframes.length > 0 ? { scale } : {}),
  };

  return {
    ...legacy,
    channels,
    timingMode: rawDurations.length > 0 ? "keyframe" : "bpm",
    bpm,
  };
}

/** Migrates every legacy-format track in level.tracks in place, once, before
 *  anything else in this file reads track.channels. */
function migrateLegacyTracks(level) {
  if (!Array.isArray(level.tracks)) return;
  level.tracks = level.tracks.map((t) => (isLegacyTrack(t) ? migrateTrackToChannels(t) : t));
}

// ---- Non-autoplay track trigger simulation (ported subset of ybnote-web's
// src/utils/track/globalSimulation.ts, src/utils/canvas/obb.ts,
// src/utils/canvas/trailSweep.ts's blockOverlapsRect/rectOverlapsRect, and
// src/utils/track/trackVisuals.ts's getTrackButtonBounds) — a non-autoplay
// track doesn't move until something TRIGGERS it (a chart note targeting it
// directly, or a cascade through a group rect/another track's control
// button), so unlike an autoplay track (always running from chart time 0),
// its carried objects have no fixed position without first simulating WHEN
// it starts. Before this, any such note silently fell back to the object's
// REST position — self-consistently wrong in both training data and
// offline eval, since target_xy()/Judge used the same wrong coordinates to
// grade themselves. See TRAIN_DIARY.md 2026-09-24 "non-autoplay track
// support". Assumes an idealized playthrough (every note "hit" on its
// charted time) to seed the cascade — the same assumption the real game's
// own offline tools (previewSoundEvents.ts, midiExport.ts) make.

function aabbIntersect(x1, y1, w1, h1, x2, y2, w2, h2) {
  return x1 < x2 + w2 && x1 + w1 > x2 && y1 < y2 + h2 && y1 + h1 > y2;
}
const IDENTITY_TRANSFORM = { rotationDeg: 0, scale: 1 };
function isIdentityTransform(t) {
  if (!t) return true;
  return ((t.rotationDeg % 360) + 360) % 360 === 0 && t.scale === 1;
}
function rectToOBB(x, y, w, h, t = IDENTITY_TRANSFORM) {
  return { cx: x + w / 2, cy: y + h / 2, hw: (w / 2) * t.scale, hh: (h / 2) * t.scale, rotationDeg: t.rotationDeg };
}
function axesOf(o) {
  const rad = (o.rotationDeg * Math.PI) / 180;
  const cos = Math.cos(rad), sin = Math.sin(rad);
  return [[cos, sin], [-sin, cos]];
}
function projectOBB(o, axis) {
  const [ux, uy] = axesOf(o)[0];
  const [vx, vy] = axesOf(o)[1];
  const c = o.cx * axis[0] + o.cy * axis[1];
  const extent = Math.abs(ux * o.hw * axis[0] + uy * o.hw * axis[1]) + Math.abs(vx * o.hh * axis[0] + vy * o.hh * axis[1]);
  return [c - extent, c + extent];
}
function obbIntersectsOBB(a, b) {
  for (const axis of [...axesOf(a), ...axesOf(b)]) {
    const [aMin, aMax] = projectOBB(a, axis);
    const [bMin, bMax] = projectOBB(b, axis);
    if (aMax < bMin || bMax < aMin) return false;
  }
  return true;
}
function rectIntersectRotatable(x1, y1, w1, h1, t1, x2, y2, w2, h2, t2) {
  if (isIdentityTransform(t1) && isIdentityTransform(t2)) return aabbIntersect(x1, y1, w1, h1, x2, y2, w2, h2);
  return obbIntersectsOBB(rectToOBB(x1, y1, w1, h1, t1), rectToOBB(x2, y2, w2, h2, t2));
}
function rectOverlapsRect(a, g) {
  const gT = { rotationDeg: g.rotationDeg ?? 0, scale: g.scale ?? 1 };
  return rectIntersectRotatable(a.x, a.y, a.w, a.h, IDENTITY_TRANSFORM, g.x, g.y, g.w, g.h, gT);
}
const DEFAULT_CONTROL_HANDLE_OFFSET = { x: 40, y: -100 };
const CONTROL_HANDLE_SIZE = 60;
function getTrackButtonBounds(track, size = CONTROL_HANDLE_SIZE) {
  const anchor = track.channels?.position?.keyframes?.[0];
  if (!anchor) return null;
  const offset = track.controlHandleOffset ?? DEFAULT_CONTROL_HANDLE_OFFSET;
  return { x: anchor.value.x + offset.x, y: anchor.value.y + offset.y, w: size, h: size };
}

const RUNNER_RADIUS = 10;

/** Returns Map<trackId, [{start, end}, ...]> in SECONDS — the time windows
 *  during which each non-autoplay track was actively running, given an
 *  idealized "every seed fires on time" playthrough. A carried object's
 *  live position at time t is evaluateTrackAtTime(track, t - segment.start)
 *  for whichever segment contains t (see resolveLivePosition below). */
function computeTrackSegments(level, seeds, endTimeSec, stepSeconds) {
  const tracks = level.tracks ?? [];
  const blocks = level.blocks ?? [];
  const groupRects = (level.groupRects ?? []).filter((g) => g.enabled !== false);
  const widgets = level.widgets ?? [];
  const trackById = new Map(tracks.map((t) => [t.id, t]));
  const groupRectById = new Map(groupRects.map((g) => [g.id, g]));
  const hasUsableChannels = (t) => (t.channels?.position?.keyframes?.length ?? 0) >= 1;

  const segmentsByTrack = new Map();
  const seedsSorted = seeds.slice().sort((a, b) => a.timeSec - b.timeSec);

  let runners = [];
  const carriedBlockGeom = new Map();
  const carriedGroupRectGeom = new Map();
  const activeRunnerByTrack = new Map();

  const closeSegment = (trackId, endSec) => {
    const segs = segmentsByTrack.get(trackId);
    if (segs && segs.length > 0) segs[segs.length - 1].end = endSec;
  };

  const triggerTrack = (trackId, atTimeSec) => {
    const track = trackById.get(trackId);
    if (!track || track.autoplay || track.enabled === false || !hasUsableChannels(track)) return;
    const existing = activeRunnerByTrack.get(trackId);
    if (existing && !existing.finished) {
      existing.finished = true;
      closeSegment(trackId, atTimeSec);
      return;
    }
    const runner = { trackId, startTimeSec: atTimeSec, memory: new Set(), finished: false };
    activeRunnerByTrack.set(trackId, runner);
    runners.push(runner);
    if (!segmentsByTrack.has(trackId)) segmentsByTrack.set(trackId, []);
    segmentsByTrack.get(trackId).push({ start: atTimeSec, end: null });
  };

  const cascadeGroupRect = (gr, atTimeSec, geom) => {
    const grForOverlap = { ...gr, ...geom };
    for (const t of tracks) {
      if (t.enabled === false || !hasUsableChannels(t) || t.controlHandleHidden === true) continue;
      const btn = getTrackButtonBounds(t);
      if (!btn) continue;
      if (rectOverlapsRect(btn, grForOverlap)) triggerTrack(t.id, atTimeSec);
    }
  };

  const spawnSeed = (seed) => {
    if (seed.targetType === "block") return; // a block seed can't itself cascade to a track
    if (seed.targetType === "groupRect") {
      const gr = groupRectById.get(seed.targetId);
      if (!gr) return;
      const live = carriedGroupRectGeom.get(gr.id);
      cascadeGroupRect(gr, seed.timeSec, {
        x: live?.x ?? gr.x, y: live?.y ?? gr.y, w: gr.w, h: gr.h,
        rotationDeg: live?.rotationDeg ?? 0, scale: live?.scale ?? 1,
      });
    } else {
      triggerTrack(seed.targetId, seed.timeSec);
    }
  };

  let nextSeedPtr = 0;
  for (let t = 0; t <= endTimeSec + 1e-9; t += stepSeconds) {
    while (nextSeedPtr < seedsSorted.length && seedsSorted[nextSeedPtr].timeSec <= t) {
      spawnSeed(seedsSorted[nextSeedPtr]);
      nextSeedPtr++;
    }
    if (runners.length === 0 && nextSeedPtr >= seedsSorted.length) break;

    const runnersThisTick = runners.slice();
    for (const runner of runnersThisTick) {
      if (runner.finished) continue;
      const track = trackById.get(runner.trackId);
      if (!track) { runner.finished = true; continue; }
      const elapsed = t - runner.startTimeSec;
      if (elapsed < 0) continue;

      const evaluation = evaluateTrackAtTime(track, elapsed);
      const posXY = evaluation;

      let hasCarriedObject = false;
      for (const block of blocks) {
        if (block.carriedByTrackId !== track.id) continue;
        hasCarriedObject = true;
        carriedBlockGeom.set(block.id, { x: posXY.x - 30, y: posXY.y - 30 });
      }
      for (const gr of groupRects) {
        if (gr.carriedByTrackId !== track.id) continue;
        hasCarriedObject = true;
        carriedGroupRectGeom.set(gr.id, { x: posXY.x - gr.w / 2, y: posXY.y - gr.h / 2 });
      }
      if (!hasCarriedObject) {
        for (const w of widgets) { if (w.carriedByTrackId === track.id) { hasCarriedObject = true; break; } }
      }

      if (!hasCarriedObject) {
        for (const other of tracks) {
          if (other.id === track.id) continue;
          if (other.enabled === false || !hasUsableChannels(other) || other.controlHandleHidden === true) continue;
          const btn = getTrackButtonBounds(other);
          if (!btn) continue;
          const isIntersecting = aabbIntersect(
            posXY.x - RUNNER_RADIUS, posXY.y - RUNNER_RADIUS, RUNNER_RADIUS * 2, RUNNER_RADIUS * 2,
            btn.x, btn.y, btn.w, btn.h,
          );
          const key = `track:${other.id}`;
          if (isIntersecting) {
            if (!runner.memory.has(key)) { triggerTrack(other.id, t); runner.memory.add(key); }
          } else if (runner.memory.has(key)) runner.memory.delete(key);
        }

        for (const gr of groupRects) {
          const live = carriedGroupRectGeom.get(gr.id);
          const gx = live?.x ?? gr.x, gy = live?.y ?? gr.y;
          const isIntersecting = aabbIntersect(
            posXY.x - RUNNER_RADIUS, posXY.y - RUNNER_RADIUS, RUNNER_RADIUS * 2, RUNNER_RADIUS * 2,
            gx, gy, gr.w, gr.h,
          );
          const key = `groupRect:${gr.id}`;
          if (isIntersecting) {
            if (!runner.memory.has(key)) {
              runner.memory.add(key);
              cascadeGroupRect(gr, t, { x: gx, y: gy, w: gr.w, h: gr.h, rotationDeg: 0, scale: 1 });
            }
          } else if (runner.memory.has(key)) runner.memory.delete(key);
        }
      }

      if (evaluation.finished) { runner.finished = true; closeSegment(track.id, t); }
    }
    if (runners.some((r) => r.finished)) runners = runners.filter((r) => !r.finished);
  }

  for (const segs of segmentsByTrack.values()) {
    for (const seg of segs) if (seg.end === null) seg.end = endTimeSec;
  }
  return segmentsByTrack;
}

const warnedNeverTriggeredTracks = new Set();

/** Resolves a target's position AT A SPECIFIC EVENT TIME, accounting for
 *  track carry — unlike the old static resolveTargets() map, this must be
 *  called per-event since a carried object's position is time-dependent.
 *  `trackSegments` (computed once per level by computeTrackSegments) gives
 *  non-autoplay tracks their trigger-dependent start times; an autoplay
 *  track always starts at chart time 0, so no segment lookup is needed. */
const warnedLegacyTracks = new Set();

function resolveLivePosition(level, restX, restY, carriedByTrackId, eventTimeMs, trackSegments) {
  if (!carriedByTrackId) return { x: restX, y: restY };
  const track = (level.tracks ?? []).find((t) => t.id === carriedByTrackId);
  if (!track) return { x: restX, y: restY };
  // migrateLegacyTracks() runs once at load time, so this should only ever
  // be malformed/empty data, not a legacy-format track — fall back rather
  // than crash: a stale position beats a hard failure on real user-uploaded
  // charts.
  const posKeyframes = track.channels?.position?.keyframes;
  if (!Array.isArray(posKeyframes) || posKeyframes.length === 0) {
    if (!warnedLegacyTracks.has(carriedByTrackId)) {
      warnedLegacyTracks.add(carriedByTrackId);
      console.warn(
        `  [track] "${carriedByTrackId}" has no position channel even after ` +
          `legacy migration — likely malformed track data. Falling back to the ` +
          `object's rest position for anything it carries.`,
      );
    }
    return { x: restX, y: restY };
  }
  if (track.autoplay) {
    // Autoplay starts at chart time 0, so elapsed seconds == event time.
    const { x, y } = evaluateTrackAtTime(track, eventTimeMs / 1000);
    return { x, y };
  }
  const eventTimeSec = eventTimeMs / 1000;
  const segs = trackSegments.get(carriedByTrackId) ?? [];
  const seg = segs.find((s) => eventTimeSec >= s.start && eventTimeSec <= s.end);
  if (!seg) {
    if (!warnedNeverTriggeredTracks.has(carriedByTrackId)) {
      warnedNeverTriggeredTracks.add(carriedByTrackId);
      console.warn(
        `  [track] "${carriedByTrackId}" is never triggered (by the simulated cascade) ` +
          `at a time one of its notes needs it — falling back to the object's rest ` +
          `position for that note.`,
      );
    }
    return { x: restX, y: restY };
  }
  const { x, y } = evaluateTrackAtTime(track, eventTimeSec - seg.start);
  return { x, y };
}

// Blocks render as a 60x60 box anchored at (x,y) as its TOP-LEFT corner —
// see PixiApproachCircleManager's `w: 60, h: 60` circle data and its
// `cx = circle.x + circle.w / 2` center calc. Using b.x/b.y directly (as
// this function did before) targets the block's top-left corner, not its
// center — reported symptom: replay's cursor lands "at the object's
// extreme top-left", missing most of the actual hitbox. groupRects below
// already did this correctly; blocks didn't.
const BLOCK_SIZE = 60;

function resolveTargets(level) {
  const byId = new Map();
  for (const b of level.blocks ?? []) {
    byId.set(b.id, {
      id: b.id,
      type: "block",
      // Rest position (center) — the fallback when this object isn't
      // track-carried, or is carried by a track resolveLivePosition can't
      // simulate (see its own comment).
      restX: b.x + BLOCK_SIZE / 2,
      restY: b.y + BLOCK_SIZE / 2,
      carriedByTrackId: b.carriedByTrackId ?? null,
      keyBinding: b.keyBinding ?? null,
      pitch: b.pitch ?? null,
      instrument: b.instrument ?? "piano",
    });
  }
  for (const g of level.groupRects ?? []) {
    if (g.enabled === false) continue;
    // groupRect's own footprint center — approach circle targets the shape,
    // frame encoding just needs a point, so use its center.
    byId.set(g.id, {
      id: g.id,
      type: "groupRect",
      restX: g.x + g.w / 2,
      restY: g.y + g.h / 2,
      carriedByTrackId: g.carriedByTrackId ?? null,
      keyBinding: g.keyBinding ?? null,
      pitch: null,
      instrument: null,
    });
  }
  for (const t of level.tracks ?? []) {
    // A track can itself be a direct note/keybind target (see
    // triggerKeyBoundPlayback's tracksWithKey in ybnote-web) — included here
    // so such an event becomes a SEED for computeTrackSegments, even though
    // it's not pushed into resolveInteractiveEvents' output below (a track
    // has no approach-circle hitbox of its own today).
    const handle = getTrackButtonBounds(t);
    byId.set(t.id, {
      id: t.id,
      type: "track",
      restX: handle ? handle.x + handle.w / 2 : 0,
      restY: handle ? handle.y + handle.h / 2 : 0,
      carriedByTrackId: null,
      keyBinding: t.keyBinding ?? null,
      trackEnabled: t.enabled !== false,
      controlHandle: handle,
    });
  }
  return byId;
}

/** Interactive events only: real blockId, background absent/false (see
 *  GameEvent doc in src/types/game.ts). Auto-trigger and pure-background
 *  events aren't scored and carry no approach circle, so the brain never
 *  needs to react to them. */
function resolveInteractiveEvents(level) {
  const targets = resolveTargets(level);

  // Seeds for computeTrackSegments: EVERY event that targets something real,
  // including background=true ones — unlike the scored/interactive events
  // below, a background event still fires in the real game (it's how an
  // author scripts a track to start without it being a judged note — see
  // TRAIN_DIARY.md 2026-09-24: this is exactly the "note delivered by a
  // flying-in track" pattern, where the track's own start is a silent
  // background event ~1-2s before the interactive note it carries becomes
  // hittable). Only `ev.blockId === "background"` (the literal sentinel for
  // "no target") is excluded — that one can never resolve to a real target
  // anyway.
  const seeds = [];
  let maxEventTimeMs = 0;
  for (const ev of level.events ?? []) {
    if (ev.blockId === "background") continue;
    const target = targets.get(ev.blockId);
    if (!target) continue;
    seeds.push({ timeSec: ev.time / 1000, targetType: target.type, targetId: target.id });
    if (ev.time > maxEventTimeMs) maxEventTimeMs = ev.time;
  }
  const endTimeSec = (maxEventTimeMs + HIT_WINDOW_MS) / 1000;
  const trackSegments = computeTrackSegments(level, seeds, endTimeSec, 0.01);

  const resolved = [];
  for (const ev of level.events ?? []) {
    if (ev.background) continue;
    if (ev.blockId === "background") continue;
    const target = targets.get(ev.blockId);
    if (!target) continue; // stale reference, e.g. deleted object
    if (
      target.type === "track" &&
      (!target.trackEnabled || !target.controlHandle)
    ) continue;
    const { x, y } = resolveLivePosition(
      level, target.restX, target.restY, target.carriedByTrackId, ev.time, trackSegments,
    );
    resolved.push({
      time: ev.time, // ms
      id: target.id,
      type: target.type,
      x,
      y,
      hasKeyBinding: target.keyBinding != null,
      keyBinding: target.keyBinding,
      pitch: ev.pitch ?? target.pitch ?? null,
      instrument: ev.instrument ?? target.instrument ?? "piano",
      windows: {
        perfect: [ev.time - PERFECT_WINDOW_MS, ev.time + PERFECT_WINDOW_MS],
        good: [ev.time - GOOD_WINDOW_MS, ev.time + GOOD_WINDOW_MS],
        bad: [ev.time - HIT_WINDOW_MS, ev.time + HIT_WINDOW_MS],
      },
    });
  }
  resolved.sort((a, b) => a.time - b.time);
  return { events: resolved, trackSegments };
}

// Sampling step for bounding a track-carried object's live path.
const BOUNDS_TRACK_SAMPLE_MS = 50;

/** The 0..1 normalized space every coordinate (notes, collidables, the RL
 *  cursor, which is clamped to it) is expressed in. It must cover everything
 *  the player can interact with, because the cursor cannot leave it — in
 *  the real game the cursor goes anywhere. It must also be SQUARE (one world
 *  scale for both axes): otherwise a normalized distance, the cursor speed
 *  cap and a D4 rotation mean different world lengths along x and y.
 *
 *  A note-positions-only box (the previous version) broke both: charts whose
 *  notes all sit on one row got a 2-world-unit-tall box (JAWNY - Honeypie
 *  720x2, a 60px block did not even fit), FALL FROM THE SKY PT. 2's single
 *  note position gave 2x2 with 7 of its 8 blocks unreachable, and float
 *  noise in 戀愛循環's row (y spread ~1e-14) slipped past the `|| 1` pad
 *  fallback into a ~0-tall box. See TRAIN_DIARY.md 2026-09-26 "bounds". */
function computeBounds(events, level, trackSegments) {
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  const addRect = (x, y, w, h) => {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x + w);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y + h);
  };
  const half = BLOCK_COLLIDER_SIZE / 2;
  for (const e of events) addRect(e.x - half, e.y - half, BLOCK_COLLIDER_SIZE, BLOCK_COLLIDER_SIZE);

  const endMs = events.reduce((m, e) => Math.max(m, e.time), 0) + HIT_WINDOW_MS;
  const addObject = (x, y, w, h, carriedByTrackId) => {
    addRect(x, y, w, h);
    if (!carriedByTrackId) return;
    // Live path: bound the rect's center along the track, with the rect
    // grown to its rotated extent (rotation is unknown per sample here).
    // Sampled only while the track is actually moving (same timing model as
    // resolveLivePosition, which would warn for every idle sample); a
    // stopped track leaves the object on its path or at rest, both covered.
    const track = (level.tracks ?? []).find((tr) => tr.id === carriedByTrackId);
    if (!track || !(track.channels?.position?.keyframes?.length > 0)) return;
    const r = Math.hypot(w, h) / 2;
    const windows = track.autoplay
      ? [{ start: 0, end: endMs / 1000 }]
      : trackSegments.get(carriedByTrackId) ?? [];
    for (const seg of windows) {
      for (let t = seg.start * 1000; t <= seg.end * 1000; t += BOUNDS_TRACK_SAMPLE_MS) {
        const c = evaluateTrackAtTime(track, t / 1000 - (track.autoplay ? 0 : seg.start));
        addRect(c.x - r, c.y - r, 2 * r, 2 * r);
      }
    }
  };
  // Same object set collectCollidables()/track handles export.
  for (const b of level.blocks ?? []) {
    if (b.enabled === false) continue;
    addObject(b.x, b.y, BLOCK_COLLIDER_SIZE, BLOCK_COLLIDER_SIZE, b.carriedByTrackId);
  }
  for (const g of level.groupRects ?? []) {
    if (g.enabled === false) continue;
    if (g.w > MAX_COLLIDABLE_WORLD_SIZE || g.h > MAX_COLLIDABLE_WORLD_SIZE) continue;
    addObject(g.x, g.y, g.w, g.h, g.carriedByTrackId);
  }
  for (const t of level.tracks ?? []) {
    if (t.enabled === false) continue;
    const handle = getTrackButtonBounds(t);
    if (handle) addRect(handle.x, handle.y, handle.w, handle.h);
  }

  if (!Number.isFinite(minX)) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  // One block of free margin on every side, so the cursor can go around an
  // edge object instead of being pinned against it; then square it up.
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const halfSpan = Math.max(maxX - minX, maxY - minY) / 2 + BLOCK_COLLIDER_SIZE;
  return { minX: cx - halfSpan, maxX: cx + halfSpan, minY: cy - halfSpan, maxY: cy + halfSpan };
}

// ---- Collidable geometry export (every enabled Block/GroupRect, not just
// ones a chart note ever targets) — the real game's trail sweep tests EVERY
// enabled Block/GroupRect for overlap, and judges Wrong on entry into one
// with no currently-due approach circle (see TRAIN_DIARY.md 2026-09-24
// "trail obstacle Wrong" — ported from PixiApproachCircleManager.ts's
// resolveGroupRectTrigger/scoreHit). A decorative "wall" groupRect (no note
// ever references its id) is exactly this case: geometrically identical to
// any other groupRect, "wrong" purely because nothing schedules a circle
// for it. This is what lets the offline Judge replicate that Wrong.
//
// SIMPLIFICATION: uses each object's REST rect (rotationDeg=0, scale=1) —
// correct for every real chart's non-carried obstacles (a Block/GroupRect's
// rotation is only ever non-zero while a Track carries it, and none of the
// obstacle-heavy charts seen so far use a moving wall). A track-carried
// collidable falls back to its rest rect rather than tracking live
// rotation/position, same documented tradeoff as resolveLivePosition's own
// legacy-track fallback.
const BLOCK_COLLIDER_SIZE = 60;

// UNVERIFIED SIMPLIFICATION: 迷宮🗣️🔥's chart has 10 enabled, silent
// (volume:0) groupRects sized ~1350x1350 WORLD units, each carrying a
// `playedAt` timestamp (i.e. author-scripted triggers, not a physical wall
// — nothing this size could be "walked around"). Treating these as
// trail-collidable obstacles made the maze's own occupancy grid 100%
// blocked (see TRAIN_DIARY.md 2026-09-24 "trail path label"). Whether the
// real game would actually judge Wrong on touching one of these is
// unconfirmed either way — dropping anything bigger than this ABSOLUTE
// world-unit threshold (comfortably above any real gameplay object seen so
// far — blocks are 60, the widest real wall segment found was ~390) is a
// pragmatic, documented guess, not a verified rule.
//
// This must be an ABSOLUTE size, not "a fraction of this chart's own
// bounds": bounds come from padding around this chart's NOTE positions,
// which can be tiny in one axis for e.g. a single-row keybind lane chart
// (one real chart's bounds were 360 wide x 2 tall) — a relative-to-bounds
// threshold wrongly excluded that chart's ordinary 80x80 lane objects,
// since 80*80 trivially exceeds even 50% of a 360x2 box. See TRAIN_DIARY.md
// 2026-09-24 "trail path label" for that regression.
const MAX_COLLIDABLE_WORLD_SIZE = 500;

// A tile-grid occupancy map for BFS path-finding labels (training/pathing.py)
// needs every collidable's cell bounds to land on EXACT integer cell
// boundaries — a maze's wall segments tile edge-to-edge with zero gap, so
// even a tiny sub-cell rounding drift between two adjacent walls can
// spuriously merge them and seal a doorway that's genuinely open. Computing
// cell bounds from NORMALIZED (0..1, divided by the padded chart bounds)
// coordinates was exactly this: normalization's extra division introduces
// float drift, and the padding itself has no relationship to the maze's own
// tile unit, so the grid's cell boundaries don't reliably land in phase
// with the walls at all (see TRAIN_DIARY.md 2026-09-24 "trail path label"
// for the debugging trail — this cost a lot of trial and error). Computing
// integer cell indices directly from RAW WORLD x/y/w/h here — before any
// normalization — is exact: every wall's own coordinates are already
// multiples of the level's true tile unit, so dividing by that same unit
// and rounding lands exactly on integers, every time.
function computeCollidableGrid(rawRects) {
  if (rawRects.length === 0) return null;
  const dims = [];
  for (const r of rawRects) {
    if (r.w > 1e-6) dims.push(r.w);
    if (r.h > 1e-6) dims.push(r.h);
  }
  if (dims.length === 0) return null;
  const tileUnit = Math.min(...dims);
  const minX = Math.min(...rawRects.map((r) => r.x));
  const minY = Math.min(...rawRects.map((r) => r.y));
  const maxX = Math.max(...rawRects.map((r) => r.x + r.w));
  const maxY = Math.max(...rawRects.map((r) => r.y + r.h));
  const resX = Math.round((maxX - minX) / tileUnit);
  const resY = Math.round((maxY - minY) / tileUnit);
  return { minX, minY, tileUnit, resX, resY };
}

function collectCollidables(level, bounds) {
  const spanX = bounds.maxX - bounds.minX;
  const spanY = bounds.maxY - bounds.minY;

  const raw = [];
  for (const b of level.blocks ?? []) {
    if (b.enabled === false) continue;
    raw.push({
      id: b.id, type: "block", x: b.x, y: b.y, w: BLOCK_COLLIDER_SIZE, h: BLOCK_COLLIDER_SIZE,
      carriedByTrackId: b.carriedByTrackId ?? null,
      keyBinding: b.keyBinding ?? null,
      pitch: b.pitch ?? null,
      instrument: b.instrument ?? "piano",
    });
  }
  for (const g of level.groupRects ?? []) {
    if (g.enabled === false) continue;
    if (g.w > MAX_COLLIDABLE_WORLD_SIZE || g.h > MAX_COLLIDABLE_WORLD_SIZE) continue;
    raw.push({
      id: g.id, type: g.type ?? "groupRect", x: g.x, y: g.y, w: g.w, h: g.h,
      carriedByTrackId: g.carriedByTrackId ?? null,
      keyBinding: g.keyBinding ?? null,
      pitch: null,
      instrument: null,
    });
  }

  const grid = computeCollidableGrid(raw);
  const out = raw.map((r) => {
    const entry = {
      id: r.id,
      type: r.type,
      x: (r.x - bounds.minX) / spanX,
      y: (r.y - bounds.minY) / spanY,
      w: r.w / spanX,
      h: r.h / spanY,
      carriedByTrackId: r.carriedByTrackId,
      keyBinding: r.keyBinding,
      pitch: r.pitch,
      instrument: r.instrument,
    };
    if (grid) {
      entry.gridCol0 = Math.round((r.x - grid.minX) / grid.tileUnit);
      entry.gridRow0 = Math.round((r.y - grid.minY) / grid.tileUnit);
      entry.gridCol1 = Math.round((r.x + r.w - grid.minX) / grid.tileUnit);
      entry.gridRow1 = Math.round((r.y + r.h - grid.minY) / grid.tileUnit);
    }
    return entry;
  });
  return { collidables: out, grid };
}

/** Exports what training/track_eval.py needs to resolve a track-carried
 *  collidable's LIVE position/rotation/scale at any instant, the same way
 *  resolveLivePosition already does for note events — collidables
 *  previously always used their REST rect even while actively being
 *  carried (documented simplification), which affects a real chunk of the
 *  corpus (14/32 charts have at least one carried object, several in the
 *  dozens). Only tracks that actually carry an exported collidable are
 *  included. `segments` (non-autoplay tracks' trigger windows, in
 *  SECONDS) is the same data resolveLivePosition uses — exporting it
 *  rather than re-deriving it in Python keeps one source of truth for
 *  "when does this track start moving." */
function exportTrackData(level, collidables, trackSegments, events) {
  const carriedTrackIds = new Set(
    collidables.map((c) => c.carriedByTrackId).filter((id) => id != null),
  );
  const simulatedTrackIds = new Set(carriedTrackIds);
  for (const event of events) {
    if (event.type === "track") simulatedTrackIds.add(event.id);
  }
  const tracks = (level.tracks ?? [])
    .filter((t) => getTrackButtonBounds(t))
    .map((t) => {
      simulatedTrackIds.add(t.id);
      return {
        id: t.id,
        channels: t.channels,
        loop: t.loop ?? false,
        bpm: t.bpm ?? 120,
        autoplay: t.autoplay === true,
        enabled: t.enabled !== false,
        keyBinding: t.keyBinding ?? null,
        controlHandleHidden: t.controlHandleHidden === true,
        controlHandle: getTrackButtonBounds(t),
      };
    });
  const segments = {};
  for (const id of simulatedTrackIds) {
    const segs = trackSegments.get(id);
    if (segs) segments[id] = segs;
  }
  return { tracks, segments };
}

/** Approach-circle progress as a 0..1 "urgency" signal, meant to drive spike
 *  rate / input current: 0 the instant an object spawns (APPROACH_TIME_MS
 *  before its hit time), rising linearly to 1 exactly at hit time — matching
 *  the circle's actual visual shrink — then held at 1 through the Bad grace
 *  window (hit time .. +HIT_WINDOW_MS), since the object is still legally
 *  hittable there. */
/** 0 -> 1 over the approach, then keeps rising 1 -> 2 across the late half
 *  of the hit window (+HIT_WINDOW_MS). It used to stay flat at 1 after the
 *  note time, so an observation could not tell "on time" from "150ms late"
 *  and the RL policy ended up systematically late (median +69ms, see
 *  TRAIN_DIARY.md 2026-09-26 "v5"). training/data.py's slot ordering uses
 *  the same formula. */
function approachProgress(t, eventTime) {
  if (t <= eventTime) {
    const elapsed = t - (eventTime - APPROACH_TIME_MS);
    return Math.max(0, Math.min(1, elapsed / APPROACH_TIME_MS));
  }
  return 1 + Math.min(1, (t - eventTime) / HIT_WINDOW_MS);
}

function buildFrames(events, bounds, dt, maxObjects) {
  if (events.length === 0) return [];
  const lastEventEnd = events[events.length - 1].time + HIT_WINDOW_MS;
  const startT = Math.min(0, events[0].time - APPROACH_TIME_MS);
  const frames = [];

  // Sliding window over events, avoids an O(frames * events) scan.
  let windowStart = 0;

  for (let t = startT; t <= lastEventEnd; t += dt) {
    while (
      windowStart < events.length &&
      events[windowStart].time + HIT_WINDOW_MS < t
    ) {
      windowStart++;
    }

    const active = [];
    for (let i = windowStart; i < events.length; i++) {
      const ev = events[i];
      if (ev.time - APPROACH_TIME_MS > t) break; // events sorted by time
      if (ev.time + HIT_WINDOW_MS < t) continue;
      active.push({
        id: ev.id,
        type: ev.type,
        proximity: approachProgress(t, ev.time),
        x: (ev.x - bounds.minX) / (bounds.maxX - bounds.minX),
        y: (ev.y - bounds.minY) / (bounds.maxY - bounds.minY),
        keybind: ev.hasKeyBinding ? 1 : 0,
        keyBinding: ev.keyBinding ?? null,
        eventTime: ev.time,
      });
    }

    active.sort((a, b) => b.proximity - a.proximity);
    frames.push({ t, objects: active.slice(0, maxObjects) });
  }

  return frames;
}

function framesToCsv(frames, maxObjects) {
  const header = ["t"];
  for (let i = 0; i < maxObjects; i++) {
    header.push(`obj${i}_proximity`, `obj${i}_x`, `obj${i}_y`, `obj${i}_keybind`);
    for (let k = 0; k < KEY_VOCAB.length; k++) header.push(`obj${i}_key${k}`);
  }
  const lines = [header.join(",")];

  for (const frame of frames) {
    const row = [frame.t];
    for (let i = 0; i < maxObjects; i++) {
      const o = frame.objects[i];
      if (o) {
        row.push(o.proximity.toFixed(4), o.x.toFixed(4), o.y.toFixed(4), o.keybind);
        const vocabIdx = o.keyBinding != null ? KEY_VOCAB.indexOf(o.keyBinding) : -1;
        for (let k = 0; k < KEY_VOCAB.length; k++) row.push(k === vocabIdx ? 1 : 0);
      } else {
        row.push(0, 0, 0, 0);
        for (let k = 0; k < KEY_VOCAB.length; k++) row.push(0);
      }
    }
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

function processOne(filePath, args, outDir) {
  const levelName = path.basename(filePath, path.extname(filePath));
  console.log(`[encode] ${filePath}`);

  const { header, level } = parseYblevel(filePath);
  migrateLegacyTracks(level);
  const { events, trackSegments } = resolveInteractiveEvents(level);
  const bounds = computeBounds(events, level, trackSegments);
  const frames = buildFrames(events, bounds, args.dt, args.maxObjects);
  const { collidables, grid: collidableGrid } = collectCollidables(level, bounds);
  const trackData = exportTrackData(level, collidables, trackSegments, events);

  fs.writeFileSync(
    path.join(outDir, `${levelName}.frames.json`),
    JSON.stringify({ dt: args.dt, maxObjects: args.maxObjects, bounds, frames }),
  );
  fs.writeFileSync(
    path.join(outDir, `${levelName}.collidables.json`),
    JSON.stringify({ bounds, collidables, grid: collidableGrid }),
  );
  fs.writeFileSync(
    path.join(outDir, `${levelName}.tracks.json`),
    JSON.stringify(trackData),
  );
  fs.writeFileSync(
    path.join(outDir, `${levelName}.frames.csv`),
    framesToCsv(frames, args.maxObjects),
  );
  fs.writeFileSync(
    path.join(outDir, `${levelName}.events.json`),
    JSON.stringify(
      {
        // Level identity for training/export_compare_bundle.py's chart
        // block. A .yblevel carries no database id (those only exist in
        // ybnote-web's levels table), so levelId stays null unless the
        // header ever gains one.
        source: {
          levelId: header.LEVEL_ID ?? header.ID ?? null,
          title: header.TITLE ?? levelName,
        },
        bounds,
        constants: {
          APPROACH_TIME_MS,
          PERFECT_WINDOW_MS,
          GOOD_WINDOW_MS,
          HIT_WINDOW_MS,
          JUDGMENT_POINTS,
          JUDGMENT_ACCURACY_WEIGHT,
          WRONG_PENALTY,
          WRONG_ACCURACY_PENALTY,
          matchByPitchInstrument: header.MATCH_BY_PITCH_INSTRUMENT === "1",
        },
        events,
      },
      null,
      2,
    ),
  );

  console.log(
    `  -> ${events.length} interactive notes, ${frames.length} frames @ ${args.dt}ms, ` +
      `${collidables.length} collidables, written to ${path.relative(ROOT, outDir)}/${levelName}.*`,
  );
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(ROOT, args.input);
  const outDir = path.resolve(ROOT, args.outDir ?? "output");
  fs.mkdirSync(outDir, { recursive: true });

  const stat = fs.statSync(inputPath);
  const files = stat.isDirectory()
    ? fs
        .readdirSync(inputPath)
        .filter((f) => f.endsWith(".yblevel"))
        .map((f) => path.join(inputPath, f))
    : [inputPath];

  if (files.length === 0) {
    console.warn(`No .yblevel files found in ${inputPath}`);
    return;
  }

  for (const file of files) {
    processOne(file, args, outDir);
  }
}

main();
