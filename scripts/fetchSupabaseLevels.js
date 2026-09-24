// Downloads every approved community .yblevel authored by a given credit
// name (matches src/services/levelsRepo.ts's getCommunityLevels() —
// search_levels RPC — the same call the real ybnote-web app makes when you
// search by author in the browser) into ybnote-ai/input/, so training has
// more than one chart to learn from — see ybnote-ai/training/TRAIN_DIARY.md
// 2026-09-24's generalization discussion.
//
// Read-only: only ever SELECTs approved levels and downloads their storage
// blob, using the same public anon key ybnote-web ships in its own client
// bundle (not a service-role key) — the exact same access any visitor
// browsing the site already has.
//
// Usage:
//   node scripts/fetchSupabaseLevels.js --author twc
//   node scripts/fetchSupabaseLevels.js --author twc --out input

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createClient } from "@supabase/supabase-js";
import ws from "ws";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

const SUPABASE_URL = "https://yndmzwwddfimrxoqitmc.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_VaOT4TMETXCneJuI4L0yiw_Y969CMRp";

function parseArgs(argv) {
  const args = { author: null, out: "input", pageSize: 50 };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a === "--author") args.author = argv[++i];
    else if (a === "--out") args.out = argv[++i];
  }
  if (!args.author) throw new Error("Missing --author <name>");
  return args;
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = path.resolve(ROOT, args.out);
  fs.mkdirSync(outDir, { recursive: true });

  const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY, {
    realtime: { transport: ws },
  });

  console.log(`Searching approved community levels with author "${args.author}"...`);
  let page = 0;
  const allRows = [];
  for (;;) {
    const { data, error } = await supabase.rpc("search_levels", {
      p_status: "approved",
      p_id: null,
      p_uploader_ids: null,
      p_query_variants: null,
      p_author_variants: [args.author],
      p_music_name_variants: null,
      p_music_author_variants: null,
      p_sort: "newest",
      p_limit: args.pageSize,
      p_offset: page * args.pageSize,
    });
    if (error) throw error;
    const rows = data ?? [];
    allRows.push(...rows);
    if (rows.length < args.pageSize) break;
    page++;
  }

  console.log(`Found ${allRows.length} level(s).`);
  if (allRows.length === 0) {
    console.log("Nothing to download.");
    return;
  }

  for (const row of allRows) {
    const safeName = (row.title || row.id).replace(/[\\/:*?"<>|]/g, "_");
    const outPath = path.join(outDir, `${safeName}.yblevel`);
    if (fs.existsSync(outPath)) {
      console.log(`  skip (already have): ${safeName}.yblevel`);
      continue;
    }
    const { data: blob, error: dlError } = await supabase.storage
      .from("levels")
      .download(row.file_path);
    if (dlError || !blob) {
      console.warn(`  FAILED to download "${row.title}" (${row.file_path}): ${dlError?.message}`);
      continue;
    }
    const buffer = Buffer.from(await blob.arrayBuffer());
    fs.writeFileSync(outPath, buffer);
    console.log(`  saved: ${safeName}.yblevel (${(buffer.length / 1024).toFixed(0)} KB)`);
  }

  console.log(`Done. Files are in ${outDir}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
