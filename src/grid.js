import { CHUNKERS } from "./chunkers.js";
import { configLabel } from "./pipeline.js";
import { rankedPassages, recallAt, reciprocalRank, ndcgAt, tuneThreshold, abstainReport } from "./metrics.js";
import { bootstrapCI, percentiles } from "./stats.js";

/** Every configuration compared by the lab: chunker x retriever x reranker x synonym expansion. */
export function defaultGrid() {
  const retrievers = [
    { retriever: "bm25", k1: 1.2, b: 0.75 },
    { retriever: "bm25", k1: 0.9, b: 0.4 },
    { retriever: "bm25", k1: 1.6, b: 0.9 },
    { retriever: "tfidf" },
    { retriever: "dense" },
    { retriever: "hybrid", k1: 1.2, b: 0.75 },
    { retriever: "hybrid", k1: 0.9, b: 0.4 },
  ];
  const out = [];
  for (const chunker of CHUNKERS) for (const r of retrievers) for (const rr of [false, true]) for (const expand of [false, true]) out.push({ chunker, ...r, rerank: rr, expand });
  return out;
}

/**
 * Run every query through a system and keep what the metrics need.
 * @param {import("./pipeline.js").System} system
 * @param {import("./queries.js").LabelledQuery[]} queries
 */
export function runQueries(system, queries) {
  return queries.map((q) => {
    const t0 = performance.now();
    const out = system.search(q.q);
    const ms = performance.now() - t0;
    const ranked = rankedPassages(out.results);
    return { id: q.id, answerable: q.gold.length > 0, ranked, confidence: out.confidence, ms, gold: q.gold };
  });
}

/** Per-query metric values for answerable queries, plus the numbers for the abstain decision. */
export function scoreRuns(runs) {
  const ans = runs.filter((r) => r.answerable);
  return {
    ndcg10: ans.map((r) => ndcgAt(r.ranked, r.gold, 10)),
    mrr: ans.map((r) => reciprocalRank(r.ranked, r.gold)),
    recall5: ans.map((r) => recallAt(r.ranked, r.gold, 5)),
    recall10: ans.map((r) => recallAt(r.ranked, r.gold, 10)),
    items: runs.map((r) => ({ confidence: r.confidence, answerable: r.answerable })),
    ms: runs.map((r) => r.ms),
  };
}

const mean = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);

/**
 * Summarise a split. With `ci` set, adds seeded bootstrap intervals.
 * @param {ReturnType<typeof scoreRuns>} s
 * @param {{threshold: number, ci?: boolean, seed?: number}} o
 */
export function summarise(s, { threshold, ci = false, seed }) {
  const metric = (xs) => (ci ? bootstrapCI(xs, { seed }) : { mean: mean(xs) });
  return {
    n: s.items.length,
    ndcg10: metric(s.ndcg10),
    mrr: metric(s.mrr),
    recall5: metric(s.recall5),
    recall10: metric(s.recall10),
    abstain: abstainReport(s.items, threshold),
    latencyMs: percentiles(s.ms),
  };
}

/**
 * Choose a configuration. This function receives tuning-split results only: it has no way to
 * read the test set. Highest tuning nDCG@10 wins; ties go to higher MRR, then smaller index, then name.
 * @template T
 * @param {T[]} rows
 * @param {(row: T) => {ndcg10: number, mrr: number, size: number, label: string}} key
 * @returns {T}
 */
export function selectBest(rows, key) {
  return [...rows].sort((a, b) => {
    const x = key(a);
    const y = key(b);
    return y.ndcg10 - x.ndcg10 || y.mrr - x.mrr || x.size - y.size || (x.label < y.label ? -1 : 1);
  })[0];
}

/**
 * Evaluate the grid on the tuning set and choose the winner and its abstain threshold from that
 * set alone. The test queries are not an argument, so nothing here can depend on them.
 * @param {ReturnType<typeof import("./pipeline.js").createLab>} lab
 * @param {import("./pipeline.js").Config[]} grid
 * @param {import("./queries.js").LabelledQuery[]} tuningQueries
 */
export function selectConfiguration(lab, grid, tuningQueries) {
  const rows = grid.map((config) => {
    const system = lab.build(config);
    const scores = scoreRuns(runQueries(system, tuningQueries));
    const threshold = tuneThreshold(scores.items);
    const summary = summarise(scores, { threshold });
    return { config, label: configLabel(config), system, threshold, summary };
  });
  const best = selectBest(rows, (r) => ({ ndcg10: r.summary.ndcg10.mean, mrr: r.summary.mrr.mean, size: r.system.sizeBytes, label: r.label }));
  return { rows, best };
}
