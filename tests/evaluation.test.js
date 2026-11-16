import test from "node:test";
import assert from "node:assert/strict";
import { data, lab } from "./helpers.js";
import { inSplit, assignSplit, parseSource } from "../src/queries.js";
import { recallAt, reciprocalRank, ndcgAt, tuneThreshold, abstainReport } from "../src/metrics.js";
import { bootstrapCI } from "../src/stats.js";
import { selectConfiguration, selectBest, defaultGrid } from "../src/grid.js";
import { buildAll } from "../scripts/build-data.mjs";
import fs from "node:fs";

test("there are at least 150 labelled queries and every gold passage exists", () => {
  assert.ok(data.queries.length >= 150);
  for (const q of data.queries) for (const g of q.gold) assert.ok(data.corpus.passages.has(g), `${q.id} ${g}`);
});

test("the corpus has at least 40 documents in English, Italian and Albanian with superseded revisions", () => {
  assert.ok(data.corpus.docs.length >= 40);
  assert.deepEqual(new Set(data.corpus.docs.map((d) => d.lang)), new Set(["en", "it", "sq"]));
  assert.ok(data.corpus.docs.some((d) => d.revision < d.latestRevision));
});

test("tuning and test queries never overlap and each query type appears in both", () => {
  const tu = new Set(inSplit(data.queries, "tuning").map((q) => q.id));
  const te = inSplit(data.queries, "test");
  assert.ok(te.every((q) => !tu.has(q.id)));
  assert.equal(tu.size + te.length, data.queries.length);
  for (const type of new Set(data.queries.map((q) => q.type))) {
    assert.ok(data.queries.some((q) => q.type === type && q.split === "test") && data.queries.some((q) => q.type === type && q.split === "tuning"), type);
  }
});

test("the shipped data files match what the build script produces", () => {
  const built = buildAll();
  assert.equal(fs.readFileSync(new URL("../data/queries.jsonl", import.meta.url), "utf8"), built.queries.map((q) => JSON.stringify(q)).join("\n") + "\n");
  assert.deepEqual(JSON.parse(fs.readFileSync(new URL("../data/lsa.json", import.meta.url), "utf8")), JSON.parse(JSON.stringify(built.lsa)));
});

test("the split is reproducible from its seed", () => {
  const items = parseSource(fs.readFileSync(new URL("../data/queries.src.txt", import.meta.url), "utf8"));
  assert.deepEqual(assignSplit(items, 4242), assignSplit(items, 4242));
  assert.notDeepEqual(assignSplit(items, 4242).map((q) => q.split), assignSplit(items, 99).map((q) => q.split));
});

test("recall, reciprocal rank and nDCG follow their definitions", () => {
  const ranked = ["a", "b", "c", "d"];
  assert.equal(recallAt(ranked, ["b", "z"], 3), 0.5);
  assert.equal(reciprocalRank(ranked, ["c"]), 1 / 3);
  assert.equal(reciprocalRank(ranked, ["z"]), 0);
  assert.equal(ndcgAt(ranked, ["a"], 10), 1);
  assert.ok(Math.abs(ndcgAt(ranked, ["b"], 10) - 1 / Math.log2(3)) < 1e-12);
});

test("the abstain threshold separates confident answerable queries from unanswerable ones", () => {
  const items = [
    ...[0.6, 0.7, 0.8, 0.9].map((c) => ({ confidence: c, answerable: true })),
    ...[0.05, 0.1, 0.2].map((c) => ({ confidence: c, answerable: false })),
  ];
  const t = tuneThreshold(items);
  assert.ok(t > 0.2 && t <= 0.6);
  const r = abstainReport(items, t);
  assert.equal(r.precision, 1);
  assert.equal(r.recall, 1);
  assert.equal(r.answerableKept, 1);
});

test("a bootstrap interval is seeded, contains the mean and narrows with more data", () => {
  const small = Array.from({ length: 20 }, (_, i) => (i % 2 ? 1 : 0));
  const big = Array.from({ length: 400 }, (_, i) => (i % 2 ? 1 : 0));
  const a = bootstrapCI(small, { seed: 1 });
  assert.deepEqual(a, bootstrapCI(small, { seed: 1 }));
  assert.ok(a.lo <= a.mean && a.mean <= a.hi);
  const c = bootstrapCI(big, { seed: 1 });
  assert.ok(c.hi - c.lo < a.hi - a.lo);
});

test("the chosen configuration is selected without reading the test set", () => {
  const grid = defaultGrid().filter((c) => c.chunker.type === "heading" && c.chunker.max === 700 && ["bm25", "tfidf"].includes(c.retriever));
  const tuning = inSplit(data.queries, "tuning");
  const first = selectConfiguration(lab, grid, tuning).best.label;
  // Scramble every test query. The selection function does not take them, so nothing can change.
  for (const q of inSplit(data.queries, "test")) (q.q = "nonsense", (q.gold = ["DE-FIN-520#1"]));
  const second = selectConfiguration(lab, grid, tuning).best.label;
  assert.equal(first, second);
  assert.equal(selectConfiguration.length, 3);
});

test("selecting on the test column can pick a different configuration than selecting on tuning", () => {
  const rows = [
    { label: "a", size: 1, tuning: 0.9, test: 0.7 },
    { label: "b", size: 1, tuning: 0.8, test: 0.75 },
  ];
  assert.equal(selectBest(rows, (r) => ({ ndcg10: r.tuning, mrr: 0, size: r.size, label: r.label })).label, "a");
  assert.equal(selectBest(rows, (r) => ({ ndcg10: r.test, mrr: 0, size: r.size, label: r.label })).label, "b");
});

test("on average, questions the documents answer get higher retrieval confidence than ones they do not", () => {
  const sys = lab.build({ chunker: { type: "heading", max: 700 }, retriever: "bm25", rerank: true, expand: true });
  const avg = (answerable) => {
    const qs = inSplit(data.queries, "tuning").filter((q) => (q.gold.length > 0) === answerable);
    return qs.reduce((s, q) => s + sys.search(q.q).confidence, 0) / qs.length;
  };
  assert.ok(avg(true) > avg(false));
});
