// Turns exported replay bundles into gameplay videos.
//
//   npm run render -- "Rhythm Hell"                 # replays/compare_Rhythm Hell.json.gz
//   npm run render -- --all                         # every bundle in replays/
//   npm run render -- --bundle replays/x.json.gz --level input/x.yblevel
//   npm run render -- "Rhythm Hell" --camera free --max-seconds 20
//
// Each replay bundle (training/export_compare_bundle.py) is paired with its
// .yblevel — looked up by name in input/ and input_*/ (a trailing `_suffix`
// of the bundle name is dropped until a level matches, so
// compare_STYX HELIX_all_versions.json.gz finds "STYX HELIX.yblevel"; pass
// --level to override) — and handed to ybnote-web's headless renderer
// (scripts/render-replay.mjs), which steps the real game engine on a virtual
// clock and writes videos/<name>.mp4. ybnote-web must be checked out next to
// this folder (or set YBNOTE_WEB / --web) and `npm install`-ed.
//
// Own options (everything else, e.g. --camera / --fps / --max-seconds, is
// forwarded to render-replay; list names BEFORE any option):
//   --all               render every bundle in replays/ that has a level
//   --bundle <file>     explicit replay file (repeatable)
//   --level <file>      level for a single --bundle
//   --out <dir>         output folder (default videos/)
//   --replays-dir <dir> where names/--all look (default replays/)
//   --skip-existing     leave a video alone if it is newer than its inputs
//   --dry-run           print the pairing, render nothing
//   --web <dir>         ybnote-web checkout (default ../ybnote-web)
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

const OWN_VALUED = new Set(["bundle", "level", "out", "replays-dir", "web"]);
const OWN_FLAGS = new Set(["all", "skip-existing", "dry-run", "help"]);

function parseArgs(argv) {
  const o = { names: [], bundles: [], forward: [] };
  let seenFlag = false;
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith("--")) {
      if (!seenFlag) o.names.push(a);
      else o.forward.push(a); // value of a forwarded option
      continue;
    }
    seenFlag = true;
    const key = a.slice(2);
    if (OWN_VALUED.has(key)) {
      const v = argv[++i];
      if (v === undefined) die(`--${key} needs a value`);
      if (key === "bundle") o.bundles.push(v);
      else o[key] = v;
    } else if (OWN_FLAGS.has(key)) {
      o[key] = true;
    } else {
      o.forward.push(a);
    }
  }
  return o;
}

function die(msg) {
  console.error(`renderVideos: ${msg}`);
  process.exit(1);
}

const stripExt = (f) => path.basename(f).replace(/\.json(\.gz)?$/i, "");

// name -> .yblevel path, over input/ then the input_* folders (first dir wins).
function indexLevels() {
  const dirs = fs
    .readdirSync(ROOT, { withFileTypes: true })
    .filter((e) => e.isDirectory() && (e.name === "input" || e.name.startsWith("input_")))
    .map((e) => e.name)
    .sort((a, b) => (a === "input" ? -1 : b === "input" ? 1 : a.localeCompare(b)));
  const map = new Map();
  for (const d of dirs) {
    for (const f of fs.readdirSync(path.join(ROOT, d))) {
      if (!f.toLowerCase().endsWith(".yblevel")) continue;
      const key = f.slice(0, -".yblevel".length);
      if (!map.has(key)) map.set(key, path.join(ROOT, d, f));
    }
  }
  return map;
}

function guessLevel(bundle, levels) {
  let stem = stripExt(bundle).replace(/^compare_/, "");
  for (;;) {
    if (levels.has(stem)) return levels.get(stem);
    const cut = stem.lastIndexOf("_");
    if (cut <= 0) return null;
    stem = stem.slice(0, cut);
  }
}

function resolveBundle(name, replaysDir) {
  if (fs.existsSync(name) && fs.statSync(name).isFile()) return path.resolve(name);
  for (const cand of [`compare_${name}.json.gz`, `${name}.json.gz`, `${name}.json`, `compare_${name}.json`]) {
    const p = path.join(replaysDir, cand);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function isReplayFile(f) {
  return /\.json(\.gz)?$/i.test(f) && !path.basename(f).startsWith("models_");
}

const args = parseArgs(process.argv.slice(2));
if (args.help) {
  console.log(fs.readFileSync(fileURLToPath(import.meta.url), "utf8").split("\n").filter((l) => l.startsWith("//")).map((l) => l.slice(3)).join("\n"));
  process.exit(0);
}

const replaysDir = path.resolve(args["replays-dir"] ?? path.join(ROOT, "replays"));
const outDir = path.resolve(args.out ?? path.join(ROOT, "videos"));
const web = path.resolve(args.web ?? process.env.YBNOTE_WEB ?? path.join(ROOT, "..", "ybnote-web"));
const renderer = path.join(web, "scripts", "render-replay.mjs");
if (!fs.existsSync(renderer)) die(`ybnote-web renderer not found at ${renderer} (set YBNOTE_WEB or --web)`);
if (!fs.existsSync(path.join(web, "node_modules", "playwright-core")))
  die(`run "npm install" in ${web} first (playwright-core is missing)`);

// ---- collect bundles ----
let bundles = args.bundles.map((b) => path.resolve(b));
for (const name of args.names) {
  const b = resolveBundle(name, replaysDir);
  if (!b) die(`no replay bundle for "${name}" in ${replaysDir}`);
  bundles.push(b);
}
if (args.all) {
  if (!fs.existsSync(replaysDir)) die(`no such folder: ${replaysDir}`);
  for (const f of fs.readdirSync(replaysDir).sort())
    if (isReplayFile(f) && fs.statSync(path.join(replaysDir, f)).isFile()) bundles.push(path.join(replaysDir, f));
}
bundles = [...new Set(bundles)];
if (bundles.length === 0) die('nothing to render — give a name, --bundle <file> or --all (see --help)');
if (args.level && bundles.length !== 1) die("--level only works with a single bundle");

// ---- pair with levels ----
const levels = indexLevels();
const jobs = [];
const unpaired = [];
const usedNames = new Set();
for (const bundle of bundles) {
  const level = args.level ? path.resolve(args.level) : guessLevel(bundle, levels);
  if (!level || !fs.existsSync(level)) {
    unpaired.push(bundle);
    continue;
  }
  let name = path.basename(level, ".yblevel");
  if (usedNames.has(name)) name = stripExt(bundle).replace(/^compare_/, "");
  usedNames.add(name);
  const out = path.join(outDir, `${name}.mp4`);
  if (args["skip-existing"] && fs.existsSync(out)) {
    const newest = Math.max(fs.statSync(bundle).mtimeMs, fs.statSync(level).mtimeMs);
    if (fs.statSync(out).mtimeMs > newest) {
      console.log(`skip (up to date): ${name}`);
      continue;
    }
  }
  jobs.push({ name, level, replays: [bundle] });
}
for (const b of unpaired) console.warn(`no level found for ${path.basename(b)} — pass --level (skipped)`);
if (jobs.length === 0) die("no renderable bundle/level pair");

console.log(`${jobs.length} video(s) -> ${outDir}`);
for (const j of jobs) console.log(`  ${j.name}\n    level  ${path.relative(ROOT, j.level)}\n    replay ${path.relative(ROOT, j.replays[0])}`);
if (args["dry-run"]) process.exit(0);

// ---- hand over to ybnote-web ----
const tmp = fs.mkdtempSync(path.join(os.tmpdir(), "ybjobs-"));
const manifest = path.join(tmp, "jobs.json");
fs.writeFileSync(manifest, JSON.stringify(jobs));
const r = spawnSync(process.execPath, [renderer, "--manifest", manifest, "--out", outDir, ...args.forward], {
  cwd: web,
  stdio: "inherit",
});
fs.rmSync(tmp, { recursive: true, force: true });
process.exit(r.status ?? 1);
