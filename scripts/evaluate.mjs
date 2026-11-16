// Runs the whole grid and writes docs/results.json.
// Configurations and the abstain threshold are chosen on the tuning split only; the test split is
// then scored for every configuration so the page can also show what happens when you cheat.
import fs from "node:fs";
import path from "node:path";
import { loadAll, ROOT } from "./load.mjs";
import { createLab, configLabel } from "../src/pipeline.js";
import { defaultGrid, selectConfiguration, selectBest, runQueries, scoreRuns, summarise } from "../src/grid.js";
import { tuneThreshold } from "../src/metrics.js";
import { inSplit, QUERY_TYPES } from "../src/queries.js";
import { rankedPassages, ndcgAt, recallAt } from "../src/metrics.js";
import { diagnose } from "../src/diagnose.js";
import { mean } from "./util.mjs";

const SEED = 20261116;
const { corpus, queries, lsa } = loadAll();
const tuning = inSplit(queries, "tuning");
const test = inSplit(queries, "test");
const lab = createLab(corpus, lsa);
const grid = defaultGrid();

const sel = selectConfiguration(lab, grid, tuning);

/** Score one built system on the test split: metrics with intervals, abstain with tuned and leaked thresholds. */
function testRow(row) {
  let runs = runQueries(row.system, test);
  // timing: median of three passes
  const passes = [runs, runQueries(row.system, test), runQueries(row.system, test)];
  runs = passes[0];
  runs.forEach((r, i) => (r.ms = [passes[0][i].ms, passes[1][i].ms, passes[2][i].ms].sort((a, b) => a - b)[1]));
  const scores = scoreRuns(runs);
  return {
    runs,
    scores,
    honest: summarise(scores, { threshold: row.threshold, ci: true, seed: SEED }),
    leakedThreshold: summarise(scores, { threshold: tuneThreshold(scores.items), ci: false }),
  };
}

const rows = sel.rows.map((r) => {
  const t = testRow(r);
  return { ...r, test: t };
});

const compact = (m) => ({ mean: m.mean, lo: m.lo, hi: m.hi });
const out = rows.map((r, i) => ({
  id: i,
  label: r.label,
  chunker: r.config.chunker,
  retriever: r.config.retriever,
  k1: r.config.k1 ?? null,
  b: r.config.b ?? null,
  rerank: r.config.rerank,
  expand: r.config.expand,
  chunks: r.system.chunks.length,
  indexBytes: r.system.sizeBytes,
  buildMs: r.system.buildMs,
  threshold: r.threshold,
  tuning: { ndcg10: r.summary.ndcg10.mean, mrr: r.summary.mrr.mean, recall5: r.summary.recall5.mean, recall10: r.summary.recall10.mean, abstain: r.summary.abstain },
  test: {
    ndcg10: compact(r.test.honest.ndcg10), mrr: compact(r.test.honest.mrr), recall5: compact(r.test.honest.recall5), recall10: compact(r.test.honest.recall10),
    abstain: r.test.honest.abstain, abstainLeaked: r.test.leakedThreshold.abstain,
  },
  latencyMs: r.test.honest.latencyMs,
}));

const key = (r) => ({ ndcg10: r.tuning.ndcg10, mrr: r.tuning.mrr, size: r.indexBytes, label: r.label });
const chosen = selectBest(out, key);
const leaked = selectBest(out, (r) => ({ ndcg10: r.test.ndcg10.mean, mrr: r.test.mrr.mean, size: r.indexBytes, label: r.label }));
const baselineIdx = out.findIndex((r) => r.retriever === "bm25" && r.k1 === 1.2 && r.chunker.type === "fixed" && r.chunker.size === 400 && !r.rerank && !r.expand);

// Ablations: the chosen configuration with one thing switched off, re-thresholded on tuning.
const chosenRow = rows[chosen.id];
function variant(label, patch) {
  const config = { ...chosenRow.config, ...patch };
  const system = lab.build(config);
  const tScores = scoreRuns(runQueries(system, tuning));
  const threshold = tuneThreshold(tScores.items);
  const s = scoreRuns(runQueries(system, test));
  const sum = summarise(s, { threshold, ci: true, seed: SEED });
  return { label, config: configLabel(config), ndcg10: compact(sum.ndcg10), mrr: compact(sum.mrr), recall10: compact(sum.recall10) };
}
const ablations = [
  variant("chosen configuration", {}),
  variant("without typo matching", { typo: false }),
  variant("without accent folding", { fold: false }),
  variant(chosenRow.config.expand ? "without synonym expansion" : "with synonym expansion", { expand: !chosenRow.config.expand }),
  variant(chosenRow.config.rerank ? "without reranker" : "with reranker", { rerank: !chosenRow.config.rerank }),
];

// Breakdown by query type for the chosen configuration (test split).
const byType = QUERY_TYPES.map((type) => {
  const idx = test.map((q, i) => (q.type === type ? i : -1)).filter((i) => i >= 0);
  const runs = idx.map((i) => chosenRow.test.runs[i]);
  const ans = runs.filter((r) => r.answerable);
  return {
    type,
    n: runs.length,
    ndcg10: ans.length ? mean(ans.map((r) => ndcgAt(r.ranked, r.gold, 10))) : null,
    recall10: ans.length ? mean(ans.map((r) => recallAt(r.ranked, r.gold, 10))) : null,
    abstained: type === "unanswerable" ? runs.filter((r) => r.confidence < chosenRow.threshold).length / runs.length : null,
  };
});

// Failure gallery: answerable test queries the chosen configuration still misses.
const failures = [];
test.forEach((q, i) => {
  if (!q.gold.length) return;
  const run = chosenRow.test.runs[i];
  if (recallAt(run.ranked, q.gold, 5) === 1 && run.confidence >= chosenRow.threshold) return;
  const o = chosenRow.system.search(q.q);
  const d = diagnose(q, o, corpus, { threshold: chosenRow.threshold });
  failures.push({
    id: q.id, q: q.q, type: q.type, lang: q.lang, gold: q.gold,
    goldTitles: q.gold.map((g) => `${corpus.passages.get(g).doc.title} (${g})`),
    top: rankedPassages(o.results).slice(0, 3).map((p) => `${corpus.passages.get(p).doc.title} (${p})`),
    confidence: o.confidence, ...d,
  });
});

// Outdated-revision effect: how often the top passage comes from a superseded revision.
function supersededTop1(config, qs) {
  const sys = lab.build(config);
  const hits = qs.filter((q) => {
    const r = rankedPassages(sys.search(q.q).results)[0];
    const doc = r && corpus.passages.get(r).doc;
    return doc && doc.revision < doc.latestRevision;
  });
  return hits.length;
}
const outdatedQs = queries.filter((q) => q.type === "outdated");
const outdated = {
  queries: outdatedQs.length,
  superseded_top1_without_reranker: supersededTop1({ ...chosenRow.config, rerank: false }, outdatedQs),
  superseded_top1_with_reranker: supersededTop1({ ...chosenRow.config, rerank: true }, outdatedQs),
};

const results = {
  generated_by: "npm run evaluate",
  seed: SEED,
  bootstrap_resamples: 2000,
  corpus: { documents: corpus.docs.length, passages: corpus.passages.size, languages: [...new Set(corpus.docs.map((d) => d.lang))], families_with_revisions: new Set(corpus.docs.filter((d) => corpus.docs.filter((e) => e.family === d.family).length > 1).map((d) => d.family)).size },
  queries: { total: queries.length, tuning: tuning.length, test: test.length, unanswerable_test: test.filter((q) => !q.gold.length).length, unanswerable_tuning: tuning.filter((q) => !q.gold.length).length },
  grid_size: out.length,
  selection_rule: "highest nDCG@10 on the tuning split; ties by MRR, then smaller index, then name",
  chosen_id: chosen.id,
  leaked_id: leaked.id,
  baseline_id: baselineIdx,
  rows: out,
  ablations,
  by_type: byType,
  outdated,
  failures,
};
fs.writeFileSync(path.join(ROOT, "docs/results.json"), JSON.stringify(results));
const f = (x) => x.toFixed(3);
console.log(`grid ${out.length} configs; chosen on tuning: ${chosen.label}`);
console.log(`  tuning nDCG@10 ${f(chosen.tuning.ndcg10)}; test nDCG@10 ${f(chosen.test.ndcg10.mean)} (${f(chosen.test.ndcg10.lo)} to ${f(chosen.test.ndcg10.hi)})`);
console.log(`  leaked choice (selected on test): ${leaked.label}, test nDCG@10 ${f(leaked.test.ndcg10.mean)}`);
console.log(`  baseline ${out[baselineIdx].label}: test nDCG@10 ${f(out[baselineIdx].test.ndcg10.mean)}`);
console.log(`  failures ${failures.length}; outdated top-1 superseded ${outdated.superseded_top1_without_reranker} -> ${outdated.superseded_top1_with_reranker} of ${outdated.queries}`);
