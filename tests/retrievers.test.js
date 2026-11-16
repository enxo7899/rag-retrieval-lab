import test from "node:test";
import assert from "node:assert/strict";
import { data, lab, baseConfig } from "./helpers.js";
import { chunkCorpus } from "../src/chunkers.js";
import { buildInvertedIndex, createBM25 } from "../src/bm25.js";
import { reciprocalRankFusion } from "../src/fusion.js";
import { powerIterationSVD, trainLSA } from "../src/lsa.js";
import { recency, proximity } from "../src/rerank.js";
import { mulberry32 } from "../src/rng.js";

test("BM25 gives zero score when no query term appears", () => {
  const chunks = chunkCorpus(data.corpus, baseConfig.chunker);
  const bm = createBM25(buildInvertedIndex(chunks, {}));
  const { scores } = bm.score([{ term: "zzzzqq", weight: 1, source: "query" }]);
  assert.ok(scores.every((s) => s === 0));
});

test("BM25 score rises with term frequency but saturates", () => {
  const chunks = [{ context: "", text: "fee" }, { context: "", text: "fee fee" }, { context: "", text: "fee fee fee fee fee fee fee fee" }, { context: "", text: "other words here" }];
  const bm = createBM25(buildInvertedIndex(chunks, {}), { k1: 1.2, b: 0 });
  const s = bm.score([{ term: "fee", weight: 1, source: "query" }]).scores;
  assert.ok(s[1] > s[0] && s[2] > s[1]);
  assert.ok(s[2] - s[1] < s[1] - s[0] + 0.5 * s[0]);
  assert.ok(s[2] < (1.2 + 1) * Math.log(1 + 3.5 / 1.5) + 1e-9);
});

test("a rarer query term contributes more to BM25 than a common one", () => {
  const chunks = [{ context: "", text: "fee rare" }, { context: "", text: "fee" }, { context: "", text: "fee bill" }, { context: "", text: "fee other" }];
  const bm = createBM25(buildInvertedIndex(chunks, {}));
  const terms = [{ term: "fee", weight: 1, source: "query" }, { term: "rare", weight: 1, source: "query" }];
  const idf = Object.fromEntries(bm.explain(terms, 0).map((r) => [r.term, r.idf]));
  assert.ok(idf.rare > idf.fee);
});

test("reciprocal rank fusion is order-independent between retrievers", () => {
  const rng = mulberry32(5);
  for (let t = 0; t < 50; t++) {
    const lists = [0, 1, 2].map(() => Array.from({ length: 12 }, (_, i) => i).sort(() => rng() - 0.5).slice(0, 8));
    const a = reciprocalRankFusion(lists);
    const b = reciprocalRankFusion([lists[2], lists[0], lists[1]]);
    assert.deepEqual(a.map((x) => [x.index, x.score]), b.map((x) => [x.index, x.score]));
  }
});

test("a passage ranked high by both retrievers beats one ranked first by only one", () => {
  const fused = reciprocalRankFusion([[1, 2, 3], [2, 4, 1]]);
  assert.equal(fused[0].index, 2);
});

test("power iteration recovers the singular values and vectors of a known matrix", () => {
  const rows = [{ idx: [0], val: [5] }, { idx: [1], val: [3] }, { idx: [2], val: [1] }];
  const { sigma, vectors } = powerIterationSVD(rows, 3, 3, { seed: 3 });
  assert.ok(sigma.every((s, i) => Math.abs(s - [5, 3, 1][i]) < 1e-6));
  assert.ok(Math.abs(Math.abs(vectors[0][0]) - 1) < 1e-6);
});

test("training the latent model twice with the same seed gives identical vectors", () => {
  const units = ["power cut fee bill", "bill payment fee late", "cable cross section current", "cable current rating table", "late fee interest bill"];
  const a = trainLSA(units, { k: 2, minDf: 2, seed: 9 });
  const b = trainLSA(units, { k: 2, minDf: 2, seed: 9 });
  assert.deepEqual(a, b);
});

test("dense retrieval puts a query next to chunks that use related words", () => {
  const sys = lab.build({ ...baseConfig, retriever: "dense" });
  const out = sys.search("late payment interest");
  assert.ok(out.results.length > 0);
  assert.ok(out.results.slice(0, 10).some((r) => r.passage.startsWith("DE-CUS-415")));
});

test("the reranker prefers the current revision when two revisions say nearly the same thing", () => {
  const q = "What type of tamper-evident seal does a technician carry?";
  const off = lab.build({ ...baseConfig, rerank: false }).search(q).results[0];
  const on = lab.build({ ...baseConfig, rerank: true }).search(q).results[0];
  assert.equal(on.chunk.docId, "DE-OPS-101-r3");
  assert.ok(off.chunk.docId.startsWith("DE-OPS-101"));
  assert.equal(recency({ revision: 3, latestRevision: 3 }), 1);
  assert.equal(recency({ revision: 2, latestRevision: 3 }), 0.5);
});

test("term proximity is higher when the matched words sit next to each other", () => {
  const near = ["late", "payment", "fee"].map((term) => ({ term }));
  const far = ["late", "x", "x", "x", "x", "payment", "x", "x", "fee"].map((term) => ({ term }));
  const q = new Set(["late", "payment", "fee"]);
  assert.ok(proximity(near, q) > proximity(far, q));
});
