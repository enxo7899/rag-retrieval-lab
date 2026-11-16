import { normalizeWord, tokenize } from "./text.js";

/**
 * @typedef {Object} QueryTerm
 * @property {string} term index term
 * @property {number} weight 1 for a word the user typed, less for guesses
 * @property {"query"|"typo"|"morph"|"synonym"} source how the term got into the query
 * @property {string} [from] the word it was derived from
 */

/**
 * @typedef {Object} QueryAnalysis
 * @property {string} raw
 * @property {QueryTerm[]} terms
 * @property {string[]} unknown words that matched nothing in the index
 */

/**
 * Optimal string alignment distance (insert, delete, substitute, swap neighbours),
 * giving up early when it must exceed `max`.
 * @param {string} a
 * @param {string} b
 * @param {number} max
 * @returns {number} the distance, or max + 1 when larger
 */
export function editDistance(a, b, max) {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const prev2 = new Array(b.length + 1).fill(0);
  let prev = Array.from({ length: b.length + 1 }, (_, j) => j);
  for (let i = 1; i <= a.length; i++) {
    const cur = [i];
    let rowMin = i;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      let v = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) v = Math.min(v, prev2[j - 2] + 1);
      cur[j] = v;
      rowMin = Math.min(rowMin, v);
    }
    if (rowMin > max) return max + 1;
    for (let j = 0; j <= b.length; j++) prev2[j] = prev[j];
    prev = cur;
  }
  return prev[b.length];
}

/**
 * Typo-tolerant lookup: the closest index term within a length-dependent distance,
 * ties broken by document frequency, then alphabetically.
 * @param {string} word
 * @param {Map<string, number>} vocab term to document frequency
 * @returns {string|null}
 */
export function fuzzyMatch(word, vocab) {
  const max = word.length >= 8 ? 2 : word.length >= 4 ? 1 : 0;
  if (max === 0) return null;
  let best = null;
  let bestD = max + 1;
  let bestDf = 0;
  for (const [term, df] of vocab) {
    const d = editDistance(word, term, Math.min(max, bestD));
    if (d > max) continue;
    if (d < bestD || (d === bestD && (df > bestDf || (df === bestDf && best !== null && term < best)))) {
      best = term;
      bestD = d;
      bestDf = df;
    }
  }
  return best;
}

const SUFFIXES = ["it", "in", "ve", "ut", "et", "en", "es", "ne", "a", "e", "i", "u"];

/** Strip a common Albanian or Italian ending when the stripped form is known. */
function morphMatch(word, known) {
  for (const s of SUFFIXES) {
    if (word.length - s.length >= 5 && word.endsWith(s) && known(word.slice(0, -s.length))) return word.slice(0, -s.length);
  }
  return null;
}

/**
 * Turn synonym groups into a lookup from term to the other terms of its groups.
 * @param {string[][]} groups
 * @param {import("./text.js").AnalyzerOptions} [analyzer]
 * @returns {Map<string, Set<string>>}
 */
export function compileSynonyms(groups, analyzer = {}) {
  const map = new Map();
  for (const g of groups) {
    const terms = [...new Set(g.map((w) => normalizeWord(w, { ...analyzer, stop: false })).filter(Boolean))];
    for (const t of terms) {
      if (!map.has(t)) map.set(t, new Set());
      for (const o of terms) if (o !== t) map.get(t).add(o);
    }
  }
  return map;
}

/**
 * Analyse a question into weighted index terms.
 * @param {string} text
 * @param {{vocab: Map<string, number>, analyzer?: import("./text.js").AnalyzerOptions,
 *   typo?: boolean, expand?: boolean, synonyms?: Map<string, Set<string>>}} o
 * @returns {QueryAnalysis}
 */
export function analyzeQuery(text, o) {
  const { vocab, analyzer = {}, typo = true, expand = false, synonyms = new Map() } = o;
  /** @type {Map<string, QueryTerm>} */
  const terms = new Map();
  const unknown = [];
  const add = (t) => {
    const old = terms.get(t.term);
    if (!old || old.weight < t.weight) terms.set(t.term, t);
  };
  const tokens = tokenize(text, analyzer);
  for (const tok of tokens) {
    let key = tok;
    if (vocab.has(tok)) add({ term: tok, weight: 1, source: "query" });
    else {
      const morph = typo ? morphMatch(tok, (c) => vocab.has(c) || synonyms.has(c)) : null;
      const fuzzy = typo ? fuzzyMatch(tok, vocab) : null;
      if (morph && vocab.has(morph)) {
        add({ term: morph, weight: 0.9, source: "morph", from: tok });
        key = morph;
      } else if (fuzzy) {
        add({ term: fuzzy, weight: 0.9, source: "typo", from: tok });
        key = fuzzy;
      } else {
        unknown.push(tok);
        if (morph) key = morph;
      }
    }
    if (expand) {
      for (const k of new Set([tok, key])) {
        for (const s of synonyms.get(k) || []) if (vocab.has(s)) add({ term: s, weight: 0.5, source: "synonym", from: tok });
      }
    }
  }
  return { raw: text, terms: [...terms.values()], unknown };
}
