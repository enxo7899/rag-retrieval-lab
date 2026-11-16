import { chunkCorpus, chunkerLabel } from "./chunkers.js";
import { buildInvertedIndex, createBM25 } from "./bm25.js";
import { createTfidf } from "./tfidf.js";
import { createDense } from "./lsa.js";
import { createHybrid, topIndices } from "./fusion.js";
import { analyzeQuery, compileSynonyms } from "./query.js";
import { rerank } from "./rerank.js";
import { SYNONYM_GROUPS } from "./synonyms.js";
import { ConfigError } from "./errors.js";

/**
 * @typedef {Object} Config
 * @property {import("./chunkers.js").ChunkerSpec} chunker
 * @property {"bm25"|"tfidf"|"dense"|"hybrid"} retriever
 * @property {number} [k1] BM25 term saturation (default 1.2)
 * @property {number} [b] BM25 length normalisation (default 0.75)
 * @property {boolean} rerank
 * @property {boolean} expand synonym query expansion
 * @property {boolean} [typo] typo-tolerant matching (default true)
 * @property {boolean} [fold] accent folding (default true)
 */

/** Short readable name of a configuration. */
export function configLabel(c) {
  const bm = c.retriever === "bm25" || c.retriever === "hybrid" ? ` k1=${c.k1 ?? 1.2} b=${c.b ?? 0.75}` : "";
  return `${chunkerLabel(c.chunker)} / ${c.retriever}${bm}${c.rerank ? " + rerank" : ""}${c.expand ? " + synonyms" : ""}`;
}

/** Depth of the list handed to the reranker and returned to callers. */
export const TOP_K = 20;

/**
 * Shared, memoised building blocks so the grid does not rebuild an index for every configuration.
 * @param {import("./corpus.js").Corpus} corpus
 * @param {import("./lsa.js").LsaModel} lsa
 */
export function createLab(corpus, lsa) {
  const cache = new Map();
  const memo = (key, make) => {
    if (!cache.has(key)) {
      const t0 = performance.now();
      const value = make();
      cache.set(key, { value, ms: performance.now() - t0 });
    }
    return cache.get(key);
  };
  const synonymCache = new Map();

  /** @param {Config} config */
  function build(config) {
    const analyzer = { fold: config.fold !== false, stem: true, stop: true };
    const akey = `${analyzer.fold}`;
    const ckey = `${chunkerLabel(config.chunker)}|${akey}`;
    const chunksE = memo(`chunks|${chunkerLabel(config.chunker)}`, () => chunkCorpus(corpus, config.chunker));
    const chunks = chunksE.value;
    const invE = memo(`inv|${ckey}`, () => buildInvertedIndex(chunks, analyzer));
    const k1 = config.k1 ?? 1.2;
    const b = config.b ?? 0.75;
    const bmE = memo(`bm25|${ckey}|${k1}|${b}`, () => createBM25(invE.value, { k1, b }));
    const tfE = memo(`tfidf|${ckey}`, () => createTfidf(chunks, invE.value, analyzer));
    const dnE = memo(`dense|${ckey}`, () => createDense(chunks, lsa, analyzer));
    const parts = { bm25: [bmE], tfidf: [tfE], dense: [dnE], hybrid: [bmE, dnE] }[config.retriever];
    if (!parts) throw new ConfigError(`unknown retriever ${config.retriever}`);
    const retriever = config.retriever === "hybrid" ? createHybrid(parts.map((p) => p.value)) : parts[0].value;
    if (!synonymCache.has(akey)) synonymCache.set(akey, compileSynonyms(SYNONYM_GROUPS, analyzer));
    const synonyms = synonymCache.get(akey);
    const vocab = invE.value.vocab;
    const buildMs = chunksE.ms + invE.ms + parts.reduce((s, p) => s + p.ms, 0);

    return {
      config,
      label: configLabel(config),
      chunks,
      retriever,
      buildMs,
      sizeBytes: retriever.sizeBytes(),
      /**
       * Run one question.
       * @param {string} text
       */
      search(text) {
        const analysis = analyzeQuery(text, { vocab, analyzer, typo: config.typo !== false, expand: config.expand, synonyms });
        const run = retriever.score(analysis.terms);
        const order = topIndices(run.scores, TOP_K);
        let ranked = order.map((index) => ({ index, score: run.scores[index] }));
        const originals = new Set(analysis.terms.filter((t) => t.source !== "synonym").map((t) => t.term));
        if (config.rerank && ranked.length) ranked = rerank(ranked, { chunks, corpus, queryTerms: originals, analyzer });
        const results = ranked.map((r, pos) => ({
          rank: pos + 1,
          index: r.index,
          chunk: chunks[r.index],
          passage: chunks[r.index].passage,
          score: r.score,
          confidence: run.confidence(r.index),
          features: r.features,
          rerankScore: r.rerankScore,
          before: r.before,
        }));
        return { analysis, results, confidence: results.length ? results[0].confidence : 0, run };
      },
    };
  }
  return { build, cache };
}

/** @typedef {ReturnType<ReturnType<typeof createLab>["build"]>} System */
