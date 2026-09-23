// Reads a .yblevel file (a zip containing level.txt) and returns the level's
// JSON payload — the same shape written by exportLevel() in
// src/utils/editor/levelUtils.ts of ybnote-web.
//
// level.txt layout:
//   KEY:VALUE
//   KEY:VALUE
//   ...
//   (blank line)
//   [JSON]
//   {...one-line JSON blob...}

import AdmZip from "adm-zip";

/**
 * @param {string} filePath absolute or relative path to a .yblevel file
 * @returns {{ header: Record<string,string>, level: object }}
 */
export function parseYblevel(filePath) {
  const zip = new AdmZip(filePath);
  const entry = zip.getEntry("level.txt");
  if (!entry) {
    throw new Error(`${filePath}: missing level.txt (not a valid .yblevel)`);
  }
  const text = entry.getData().toString("utf8");

  const jsonMarker = "[JSON]\n";
  const markerIndex = text.indexOf(jsonMarker);
  if (markerIndex === -1) {
    throw new Error(`${filePath}: missing [JSON] section`);
  }

  const headerText = text.slice(0, markerIndex);
  const jsonText = text.slice(markerIndex + jsonMarker.length);

  const header = {};
  for (const line of headerText.split("\n")) {
    if (!line.trim()) continue;
    const colonIndex = line.indexOf(":");
    if (colonIndex === -1) continue;
    header[line.slice(0, colonIndex)] = line.slice(colonIndex + 1);
  }

  const level = JSON.parse(jsonText);
  return { header, level };
}
