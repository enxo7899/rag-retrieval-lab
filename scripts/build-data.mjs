// Rebuilds data/corpus.json, data/queries.jsonl and data/lsa.json from the authoring files.
// Run with `npm run build`. Output is deterministic.
import fs from "node:fs";
import path from "node:path";
import { ROOT } from "./load.mjs";
import { parseDocument, buildCorpus, corpusToJSON } from "../src/corpus.js";
import { parseSource, assignSplit } from "../src/queries.js";
import { trainLSA } from "../src/lsa.js";

export const SPLIT_SEED = 4242;

/** Documents, queries and the LSA model, built from the files in data/. */
export function buildAll() {
  const dir = path.join(ROOT, "data/docs");
  const parsed = fs.readdirSync(dir).filter((f) => f.endsWith(".md")).sort().map((f) => parseDocument(fs.readFileSync(path.join(dir, f), "utf8")));
  const corpus = buildCorpus(parsed);
  const queries = assignSplit(parseSource(fs.readFileSync(path.join(ROOT, "data/queries.src.txt"), "utf8")), SPLIT_SEED);
  const units = corpus.docs.flatMap((d) => d.blocks.filter((b) => b.kind !== "heading").map((b) => d.body.slice(b.start, b.end)));
  const lsa = trainLSA(units, { k: 48, minDf: 2, seed: 7 });
  return { corpus, queries, lsa };
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const { corpus, queries, lsa } = buildAll();
  const missing = queries.flatMap((q) => q.gold.filter((g) => !corpus.passages.has(g)).map((g) => `${q.id}: ${g}`));
  if (missing.length) {
    console.error("unknown gold passages:\n" + missing.join("\n"));
    process.exit(1);
  }
  fs.writeFileSync(path.join(ROOT, "data/corpus.json"), JSON.stringify(corpusToJSON(corpus)));
  fs.writeFileSync(path.join(ROOT, "data/queries.jsonl"), queries.map((q) => JSON.stringify(q)).join("\n") + "\n");
  fs.writeFileSync(path.join(ROOT, "data/lsa.json"), JSON.stringify(lsa));
  const n = (s) => queries.filter((q) => q.split === s).length;
  console.log(`${corpus.docs.length} documents, ${corpus.passages.size} passages, ${queries.length} queries (${n("tuning")} tuning, ${n("test")} test), LSA ${lsa.vocab.length} terms x ${lsa.k} dims`);
}
