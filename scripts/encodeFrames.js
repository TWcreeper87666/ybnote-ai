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

function resolveTargets(level) {
  const byId = new Map();
  for (const b of level.blocks ?? []) {
    byId.set(b.id, { id: b.id, type: "block", x: b.x, y: b.y, keyBinding: b.keyBinding ?? null });
  }
  for (const g of level.groupRects ?? []) {
    if (g.enabled === false) continue;
    // groupRect's own footprint center — approach circle targets the shape,
    // frame encoding just needs a point, so use its center.
    byId.set(g.id, {
      id: g.id,
      type: "groupRect",
      x: g.x + g.w / 2,
      y: g.y + g.h / 2,
      keyBinding: g.keyBinding ?? null,
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
  const resolved = [];
  for (const ev of level.events ?? []) {
    if (ev.background) continue;
    if (ev.blockId === "background") continue;
    const target = targets.get(ev.blockId);
    if (!target) continue; // stale reference, e.g. deleted object
    resolved.push({
      time: ev.time, // ms
      id: target.id,
      type: target.type,
      x: target.x,
      y: target.y,
      hasKeyBinding: target.keyBinding != null,
      keyBinding: target.keyBinding,
      windows: {
        perfect: [ev.time - PERFECT_WINDOW_MS, ev.time + PERFECT_WINDOW_MS],
        good: [ev.time - GOOD_WINDOW_MS, ev.time + GOOD_WINDOW_MS],
        bad: [ev.time - HIT_WINDOW_MS, ev.time + HIT_WINDOW_MS],
      },
    });
  }
  resolved.sort((a, b) => a.time - b.time);
  return resolved;
}

function computeBounds(events) {
  if (events.length === 0) return { minX: 0, maxX: 1, minY: 0, maxY: 1 };
  let minX = Infinity,
    maxX = -Infinity,
    minY = Infinity,
    maxY = -Infinity;
  for (const e of events) {
    minX = Math.min(minX, e.x);
    maxX = Math.max(maxX, e.x);
    minY = Math.min(minY, e.y);
    maxY = Math.max(maxY, e.y);
  }
  // Pad 10% so edge objects don't sit exactly on 0/1 (keeps LIF input current
  // off the boundary, where a tiny camera-independent encoding error would
  // otherwise clip to a hard 0 or 1).
  const padX = (maxX - minX) * 0.1 || 1;
  const padY = (maxY - minY) * 0.1 || 1;
  return { minX: minX - padX, maxX: maxX + padX, minY: minY - padY, maxY: maxY + padY };
}

/** Approach-circle progress as a 0..1 "urgency" signal, meant to drive spike
 *  rate / input current: 0 the instant an object spawns (APPROACH_TIME_MS
 *  before its hit time), rising linearly to 1 exactly at hit time — matching
 *  the circle's actual visual shrink — then held at 1 through the Bad grace
 *  window (hit time .. +HIT_WINDOW_MS), since the object is still legally
 *  hittable there. */
function approachProgress(t, eventTime) {
  if (t <= eventTime) {
    const elapsed = t - (eventTime - APPROACH_TIME_MS);
    return Math.max(0, Math.min(1, elapsed / APPROACH_TIME_MS));
  }
  return 1;
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
  }
  const lines = [header.join(",")];

  for (const frame of frames) {
    const row = [frame.t];
    for (let i = 0; i < maxObjects; i++) {
      const o = frame.objects[i];
      if (o) row.push(o.proximity.toFixed(4), o.x.toFixed(4), o.y.toFixed(4), o.keybind);
      else row.push(0, 0, 0, 0);
    }
    lines.push(row.join(","));
  }
  return lines.join("\n");
}

function processOne(filePath, args, outDir) {
  const levelName = path.basename(filePath, path.extname(filePath));
  console.log(`[encode] ${filePath}`);

  const { level } = parseYblevel(filePath);
  const events = resolveInteractiveEvents(level);
  const bounds = computeBounds(events);
  const frames = buildFrames(events, bounds, args.dt, args.maxObjects);

  fs.writeFileSync(
    path.join(outDir, `${levelName}.frames.json`),
    JSON.stringify({ dt: args.dt, maxObjects: args.maxObjects, bounds, frames }),
  );
  fs.writeFileSync(
    path.join(outDir, `${levelName}.frames.csv`),
    framesToCsv(frames, args.maxObjects),
  );
  fs.writeFileSync(
    path.join(outDir, `${levelName}.events.json`),
    JSON.stringify(
      {
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
        },
        events,
      },
      null,
      2,
    ),
  );

  console.log(
    `  -> ${events.length} interactive notes, ${frames.length} frames @ ${args.dt}ms, written to ${path.relative(ROOT, outDir)}/${levelName}.*`,
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
