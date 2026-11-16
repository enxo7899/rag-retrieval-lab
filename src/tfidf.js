import { tokenize, termCounts } from "./text.js";

/**
 * TF-IDF cosine retriever. Weight = (1 + ln tf) * (ln((N + 1) / (df + 1)) + 1), vectors are length-normalised.
 * @param {import("./chunkers.js").Chunk[]} chunks
 * @param {ReturnType<typeof import("./bm25.js").buildInvertedIndex>} inv
 * @param {import("./text.js").AnalyzerOptions} analyzer
 */
export function createTfidf(chunks, inv, analyzer) {
  const idf = (t) => Math.log((inv.n + 1) / ((inv.vocab.get(t) || 0) + 1)) + 1;
  /** @type {Map<string, number[]>} term -> [chunk, weight, ...] */
  const postings = new Map();
  chunks.forEach((c, i) => {
    const counts = termCounts(tokenize(`${c.context} ${c.text}`, analyzer));
    const w = new Map([...counts].map(([t, tf]) => [t, (1 + Math.log(tf)) * idf(t)]));
    const len = Math.sqrt([...w.values()].reduce((a, x) => a + x * x, 0)) || 1;
    for (const [t, x] of w) {
      let p = postings.get(t);
      if (!p) postings.set(t, (p = []));
      p.push(i, x / len);
    }
  });
  return {
    name: "tfidf",
    /** @param {import("./query.js").QueryTerm[]} terms */
    score(terms) {
      const scores = new Float64Array(inv.n);
      let qlen = 0;
      const q = terms.filter((t) => inv.vocab.has(t.term)).map((t) => ({ term: t.term, x: t.weight * idf(t.term) }));
      for (const { x } of q) qlen += x * x;
      qlen = Math.sqrt(qlen) || 1;
      for (const { term, x } of q) {
        const p = postings.get(term);
        for (let j = 0; j < p.length; j += 2) scores[p[j]] += (x / qlen) * p[j + 1];
      }
      return { scores, confidence: (i) => scores[i] };
    },
    sizeBytes: () => JSON.stringify(Object.fromEntries([...postings].map(([t, p]) => [t, p.map((v, j) => (j % 2 ? Math.round(v * 1e4) / 1e4 : v))]))).length,
  };
}
