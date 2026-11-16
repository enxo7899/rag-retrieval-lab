/**
 * Reciprocal rank fusion: a chunk's score is the sum over rankings of 1 / (k + rank).
 * Contributions are added in sorted order and ties are broken by id, so the result does not
 * depend on the order in which the rankings are given.
 * @param {number[][]} rankings each a list of chunk indices, best first
 * @param {number} [k] damping constant
 * @returns {{index: number, score: number, ranks: (number|null)[]}[]} best first
 */
export function reciprocalRankFusion(rankings, k = 60) {
  /** @type {Map<number, number[]>} */
  const parts = new Map();
  for (const list of rankings) {
    list.forEach((idx, pos) => {
      if (!parts.has(idx)) parts.set(idx, []);
      parts.get(idx).push(1 / (k + pos + 1));
    });
  }
  const out = [...parts].map(([index, p]) => ({
    index,
    score: p.sort((a, b) => a - b).reduce((s, x) => s + x, 0),
    ranks: rankings.map((list) => {
      const pos = list.indexOf(index);
      return pos < 0 ? null : pos + 1;
    }),
  }));
  return out.sort((a, b) => b.score - a.score || a.index - b.index);
}

/**
 * Hybrid retriever: reciprocal rank fusion of the top `depth` results of each component.
 * @param {{name: string, score: (t: any) => {scores: Float64Array, confidence: (i: number) => number}}[]} parts
 * @param {number} depth how many results of each component enter the fusion
 */
export function createHybrid(parts, depth = 50) {
  return {
    name: "hybrid",
    parts,
    /** @param {import("./query.js").QueryTerm[]} terms */
    score(terms) {
      const runs = parts.map((p) => p.score(terms));
      const rankings = runs.map((r) => topIndices(r.scores, depth));
      const fused = reciprocalRankFusion(rankings);
      const scores = new Float64Array(runs[0].scores.length);
      for (const f of fused) scores[f.index] = f.score;
      const byIndex = new Map(fused.map((f) => [f.index, f]));
      return {
        scores,
        confidence: (i) => runs.reduce((s, r) => s + r.confidence(i), 0) / runs.length,
        fusion: (i) => byIndex.get(i),
        components: runs,
      };
    },
    sizeBytes: () => parts.reduce((s, p) => s + p.sizeBytes(), 0),
  };
}

/**
 * Indices of the highest positive scores, best first, ties by index.
 * @param {Float64Array} scores
 * @param {number} n
 */
export function topIndices(scores, n) {
  const idx = [];
  for (let i = 0; i < scores.length; i++) if (scores[i] > 0) idx.push(i);
  idx.sort((a, b) => scores[b] - scores[a] || a - b);
  return idx.slice(0, n);
}
