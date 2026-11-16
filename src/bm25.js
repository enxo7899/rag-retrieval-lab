import { tokenize, termCounts } from "./text.js";

/**
 * Inverted index over chunk text (context heading plus body).
 * @param {import("./chunkers.js").Chunk[]} chunks
 * @param {import("./text.js").AnalyzerOptions} analyzer
 */
export function buildInvertedIndex(chunks, analyzer) {
  /** @type {Map<string, number[]>} term -> flat [chunk, tf, chunk, tf, ...] */
  const postings = new Map();
  const lengths = new Float64Array(chunks.length);
  chunks.forEach((c, i) => {
    const terms = tokenize(`${c.context} ${c.text}`, analyzer);
    lengths[i] = terms.length;
    for (const [t, tf] of termCounts(terms)) {
      let p = postings.get(t);
      if (!p) postings.set(t, (p = []));
      p.push(i, tf);
    }
  });
  const vocab = new Map([...postings].map(([t, p]) => [t, p.length / 2]));
  const avgLength = lengths.reduce((a, b) => a + b, 0) / Math.max(1, chunks.length);
  return { postings, lengths, vocab, avgLength, n: chunks.length };
}

/**
 * BM25 retriever written from scratch.
 * score(d, q) = sum over query terms of w * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len / avgLen))
 * with idf = ln(1 + (N - df + 0.5) / (df + 0.5)), so a term that is in every chunk still scores above zero.
 * @param {ReturnType<typeof buildInvertedIndex>} inv
 * @param {{k1?: number, b?: number}} [params]
 */
export function createBM25(inv, { k1 = 1.2, b = 0.75 } = {}) {
  const idf = (df) => Math.log(1 + (inv.n - df + 0.5) / (df + 0.5));
  const norm = (i) => k1 * (1 - b + (b * inv.lengths[i]) / inv.avgLength);
  return {
    name: "bm25",
    k1,
    b,
    /** @param {import("./query.js").QueryTerm[]} terms */
    score(terms) {
      const scores = new Float64Array(inv.n);
      let ceiling = 0;
      for (const { term, weight } of terms) {
        const p = inv.postings.get(term);
        if (!p) continue;
        const w = weight * idf(p.length / 2);
        ceiling += w * (k1 + 1);
        for (let j = 0; j < p.length; j += 2) {
          const i = p[j];
          const tf = p[j + 1];
          scores[i] += (w * tf * (k1 + 1)) / (tf + norm(i));
        }
      }
      /** Share of the best score a chunk could reach, so thresholds do not depend on query length. */
      return { scores, confidence: (i) => (ceiling > 0 ? scores[i] / ceiling : 0) };
    },
    /** Per-term contributions to one chunk's score. */
    explain(terms, i) {
      const rows = [];
      for (const { term, weight, source } of terms) {
        const p = inv.postings.get(term);
        if (!p) continue;
        for (let j = 0; j < p.length; j += 2) {
          if (p[j] !== i) continue;
          const tf = p[j + 1];
          const w = weight * idf(p.length / 2);
          rows.push({ term, source, tf, idf: idf(p.length / 2), contribution: (w * tf * (k1 + 1)) / (tf + norm(i)) });
        }
      }
      return rows.sort((a, c) => c.contribution - a.contribution);
    },
    sizeBytes: () => JSON.stringify({ k1, b, lengths: Array.from(inv.lengths), postings: Object.fromEntries(inv.postings) }).length,
  };
}
