import { tokenSpans } from "./text.js";

/** Weights are fixed by hand and not tuned on the data. */
export const RERANK_WEIGHTS = { proximity: 0.15, title: 0.3, recency: 0.25 };

/**
 * Smallest window of consecutive tokens that holds every matched query term, as a share of
 * how tight the match is: matched terms / window length, scaled by how many query terms matched.
 * @param {{term: string}[]} tokens chunk tokens in order
 * @param {Set<string>} queryTerms
 * @returns {number} 0 to 1
 */
export function proximity(tokens, queryTerms) {
  if (queryTerms.size === 0) return 0;
  const present = new Set(tokens.map((t) => t.term).filter((t) => queryTerms.has(t)));
  const m = present.size;
  if (m === 0) return 0;
  let best = Infinity;
  const counts = new Map();
  let have = 0;
  let left = 0;
  for (let right = 0; right < tokens.length; right++) {
    const t = tokens[right].term;
    if (present.has(t)) {
      counts.set(t, (counts.get(t) || 0) + 1);
      if (counts.get(t) === 1) have++;
    }
    while (have === m) {
      best = Math.min(best, right - left + 1);
      const l = tokens[left].term;
      if (present.has(l)) {
        counts.set(l, counts.get(l) - 1);
        if (counts.get(l) === 0) have--;
      }
      left++;
    }
  }
  return (m / queryTerms.size) * (m / best);
}

/**
 * 1 for the newest revision of a document, falling by 0.5 for every revision behind it.
 * @param {{revision: number, latestRevision: number}} doc
 */
export function recency(doc) {
  return Math.max(0, 1 - 0.5 * (doc.latestRevision - doc.revision));
}

/**
 * Re-order the top results using three cheap signals on top of the retriever score.
 * @param {{index: number, score: number}[]} ranked top results, best first
 * @param {{chunks: import("./chunkers.js").Chunk[], corpus: import("./corpus.js").Corpus,
 *   queryTerms: Set<string>, analyzer: import("./text.js").AnalyzerOptions, weights?: typeof RERANK_WEIGHTS}} ctx
 */
export function rerank(ranked, { chunks, corpus, queryTerms, analyzer, weights = RERANK_WEIGHTS }) {
  const top = Math.max(...ranked.map((r) => r.score), 0) || 1;
  const rows = ranked.map((r) => {
    const chunk = chunks[r.index];
    const doc = corpus.byId.get(chunk.docId);
    const sec = doc.sections.find((s) => s.passage === chunk.passage);
    const heading = new Set(tokenSpans(`${doc.title} ${sec ? sec.heading : ""}`, analyzer).map((t) => t.term));
    const hits = [...queryTerms].filter((t) => heading.has(t)).length;
    const features = {
      base: r.score / top,
      proximity: proximity(tokenSpans(chunk.text, analyzer), queryTerms),
      title: queryTerms.size ? hits / queryTerms.size : 0,
      recency: recency(doc),
    };
    const rerankScore = features.base + weights.proximity * features.proximity + weights.title * features.title + weights.recency * features.recency;
    return { ...r, rerankScore, features, before: 0 };
  });
  rows.forEach((r, i) => (r.before = i + 1));
  return rows.sort((a, b) => b.rerankScore - a.rerankScore || a.index - b.index);
}
