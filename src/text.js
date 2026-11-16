/**
 * Text analysis shared by every retriever and by query handling.
 * Lowercases, folds accents (so Albanian e-diaeresis and c-cedilla match plain e and c),
 * drops stop words in English, Italian and Albanian, and applies a very light suffix stripper.
 */

const STOP_WORDS = `
a an the of to in on for and or is are was were be by with at from as it its that this these those which what how when who where why
do does did can may must should shall will would i my me we you your our they their there if not no than then any each per has have had been
into over under up about after before between also only more most other some such so but
il lo la gli le un una uno di da con su tra fra e che chi come quando quanto quanti quanta quale quali dove non si al del della dei delle nel nella
entro piu sono ha hanno puo possono posso mi ti ci vi suo sua
te se nje ne me per nga si sa ka eshte jane nuk qe kur cili cila cilat cilin cfare kush une ju ata ato mund duhet nese ose por
`;

/** @type {Set<string>} */
export const STOP = new Set(STOP_WORDS.split(/\s+/).filter(Boolean));

/**
 * Lowercase and optionally strip accents.
 * @param {string} s
 * @param {boolean} [accents] true to fold accents (default)
 */
export function fold(s, accents = true) {
  const lower = s.toLowerCase();
  return accents ? lower.normalize("NFD").replace(/\p{M}/gu, "") : lower;
}

/**
 * A deliberately small English suffix stripper. It is applied to every language, which is
 * harmless because the same function runs on documents and queries.
 * @param {string} w folded word
 * @returns {string}
 */
export function stem(w) {
  if (w.length <= 3 || /\d/.test(w)) return w;
  if (w.endsWith("ies") && w.length > 4) return w.slice(0, -3) + "y";
  if (/(ss|us|is)$/.test(w)) return w;
  if (/(ch|sh|x|z|ss)es$/.test(w)) return w.slice(0, -2);
  if (w.endsWith("s")) w = w.slice(0, -1);
  if (w.endsWith("ing") && w.length > 6) w = w.slice(0, -3);
  else if (w.endsWith("ed") && w.length > 5) w = w.slice(0, -2);
  if (w.endsWith("e") && w.length > 4) w = w.slice(0, -1);
  return w;
}

/**
 * @typedef {Object} AnalyzerOptions
 * @property {boolean} [fold] fold accents (default true)
 * @property {boolean} [stem] strip suffixes (default true)
 * @property {boolean} [stop] drop stop words (default true)
 */

/**
 * Normalise one word with the analyzer.
 * @param {string} word
 * @param {AnalyzerOptions} [o]
 * @returns {string|null} the term, or null when it is a stop word
 */
export function normalizeWord(word, o = {}) {
  let t = fold(word, o.fold !== false);
  if (o.stop !== false && STOP.has(fold(word, true))) return null;
  if (o.stem !== false) t = stem(t);
  return t;
}

/**
 * Tokenise text and keep the character span of every term in the original string.
 * @param {string} text
 * @param {AnalyzerOptions} [o]
 * @returns {{term: string, start: number, end: number}[]}
 */
export function tokenSpans(text, o = {}) {
  const out = [];
  for (const m of text.matchAll(/[\p{L}\p{N}]+/gu)) {
    const term = normalizeWord(m[0], o);
    if (term) out.push({ term, start: m.index, end: m.index + m[0].length });
  }
  return out;
}

/**
 * @param {string} text
 * @param {AnalyzerOptions} [o]
 * @returns {string[]}
 */
export function tokenize(text, o = {}) {
  return tokenSpans(text, o).map((t) => t.term);
}

/**
 * Count term frequencies.
 * @param {string[]} terms
 * @returns {Map<string, number>}
 */
export function termCounts(terms) {
  const m = new Map();
  for (const t of terms) m.set(t, (m.get(t) || 0) + 1);
  return m;
}
