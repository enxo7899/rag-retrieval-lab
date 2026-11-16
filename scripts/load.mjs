import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { corpusFromJSON } from "../src/corpus.js";
import { parseQueries } from "../src/queries.js";

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (p) => fs.readFileSync(path.resolve(ROOT, p), "utf8");

/** Read the shipped data files in Node. The browser fetches the same files. */
export function loadAll() {
  return {
    corpus: corpusFromJSON(JSON.parse(read("data/corpus.json"))),
    queries: parseQueries(read("data/queries.jsonl")),
    lsa: JSON.parse(read("data/lsa.json")),
  };
}
