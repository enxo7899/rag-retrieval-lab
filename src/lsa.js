import { mulberry32 } from "./rng.js";
import { tokenize, termCounts } from "./text.js";

/**
 * Truncated SVD by power iteration with deflation, no libraries.
 * Finds the top k right singular vectors of a sparse matrix X (rows are documents) by repeatedly
 * applying X^T X to a seeded random vector and removing the directions already found.
 * @param {{idx: number[], val: number[]}[]} rows sparse rows
 * @param {number} nCols
 * @param {number} k
 * @param {{seed?: number, iterations?: number, tolerance?: number}} [o]
 * @returns {{sigma: number[], vectors: Float64Array[]}} singular values and right singular vectors
 */
export function powerIterationSVD(rows, nCols, k, { seed = 1, iterations = 300, tolerance = 1e-10 } = {}) {
  const rng = mulberry32(seed);
  const vectors = [];
  const sigma = [];
  const mulX = (v) => rows.map((r) => r.idx.reduce((s, c, j) => s + r.val[j] * v[c], 0));
  const mulXt = (u) => {
    const out = new Float64Array(nCols);
    rows.forEach((r, i) => r.idx.forEach((c, j) => (out[c] += r.val[j] * u[i])));
    return out;
  };
  const orthogonalise = (v) => {
    for (const p of vectors) {
      let d = 0;
      for (let i = 0; i < nCols; i++) d += v[i] * p[i];
      for (let i = 0; i < nCols; i++) v[i] -= d * p[i];
    }
  };
  const normalise = (v) => {
    const n = Math.sqrt(v.reduce((s, x) => s + x * x, 0));
    if (n > 0) for (let i = 0; i < nCols; i++) v[i] /= n;
    return n;
  };
  for (let f = 0; f < Math.min(k, nCols); f++) {
    let v = Float64Array.from({ length: nCols }, () => rng() - 0.5);
    orthogonalise(v);
    normalise(v);
    for (let it = 0; it < iterations; it++) {
      const w = mulXt(mulX(v));
      orthogonalise(w);
      if (normalise(w) === 0) break;
      let dot = 0;
      for (let i = 0; i < nCols; i++) dot += w[i] * v[i];
      v = w;
      if (1 - Math.abs(dot) < tolerance) break;
    }
    const s = Math.sqrt(mulX(v).reduce((a, x) => a + x * x, 0));
    if (s < 1e-9) break;
    vectors.push(v);
    sigma.push(s);
  }
  return { sigma, vectors };
}

/**
 * @typedef {Object} LsaModel
 * @property {number} k latent dimensions
 * @property {string[]} vocab
 * @property {number[]} idf
 * @property {number[]} sigma
 * @property {number[][]} T term vectors, one row of k numbers per vocabulary term
 */

/**
 * Train latent semantic analysis on a list of text units (paragraphs).
 * Terms seen in fewer than minDf units are dropped. Each unit is a length-normalised TF-IDF row.
 * @param {string[]} units
 * @param {{k?: number, minDf?: number, seed?: number, analyzer?: import("./text.js").AnalyzerOptions}} [o]
 * @returns {LsaModel}
 */
export function trainLSA(units, { k = 48, minDf = 2, seed = 7, analyzer = {} } = {}) {
  const counted = units.map((u) => termCounts(tokenize(u, analyzer)));
  const df = new Map();
  for (const c of counted) for (const t of c.keys()) df.set(t, (df.get(t) || 0) + 1);
  const vocab = [...df].filter(([, d]) => d >= minDf).map(([t]) => t).sort();
  const index = new Map(vocab.map((t, i) => [t, i]));
  const idf = vocab.map((t) => Math.log((units.length + 1) / (df.get(t) + 1)) + 1);
  const rows = counted.map((c) => {
    const idx = [];
    const val = [];
    for (const [t, tf] of c) {
      const i = index.get(t);
      if (i !== undefined) idx.push(i), val.push((1 + Math.log(tf)) * idf[i]);
    }
    const len = Math.sqrt(val.reduce((a, x) => a + x * x, 0)) || 1;
    return { idx, val: val.map((x) => x / len) };
  });
  const { sigma, vectors } = powerIterationSVD(rows, vocab.length, k, { seed });
  const r = (x) => Math.round(x * 1e4) / 1e4;
  return {
    k: sigma.length,
    vocab,
    idf: idf.map(r),
    sigma: sigma.map(r),
    T: vocab.map((_, t) => vectors.map((v) => r(v[t]))),
  };
}

/**
 * Dense retriever: queries and chunks are projected into the latent space and compared by cosine.
 * @param {import("./chunkers.js").Chunk[]} chunks
 * @param {LsaModel} model
 * @param {import("./text.js").AnalyzerOptions} analyzer
 */
export function createDense(chunks, model, analyzer) {
  const index = new Map(model.vocab.map((t, i) => [t, i]));
  /** Project weighted terms into the latent space and normalise. */
  const project = (weighted) => {
    const z = new Float64Array(model.k);
    for (const [t, x] of weighted) {
      const i = index.get(t);
      if (i === undefined) continue;
      const row = model.T[i];
      for (let j = 0; j < model.k; j++) z[j] += x * model.idf[i] * row[j];
    }
    const n = Math.sqrt(z.reduce((s, v) => s + v * v, 0));
    if (n > 0) for (let j = 0; j < model.k; j++) z[j] /= n;
    return { z, norm: n };
  };
  const vectors = chunks.map((c) => {
    const counts = termCounts(tokenize(`${c.context} ${c.text}`, analyzer));
    return project([...counts].map(([t, tf]) => [t, 1 + Math.log(tf)])).z;
  });
  return {
    name: "dense",
    /** @param {import("./query.js").QueryTerm[]} terms */
    score(terms) {
      const { z, norm } = project(terms.map((t) => [t.term, t.weight]));
      const scores = new Float64Array(chunks.length);
      if (norm > 0) {
        vectors.forEach((v, i) => {
          let d = 0;
          for (let j = 0; j < model.k; j++) d += z[j] * v[j];
          scores[i] = Math.max(0, d);
        });
      }
      return { scores, confidence: (i) => scores[i] };
    },
    sizeBytes: () => JSON.stringify(vectors.map((v) => Array.from(v, (x) => Math.round(x * 1e4) / 1e4))).length,
  };
}
