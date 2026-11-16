import { ConfigError } from "./errors.js";
import { shuffled } from "./rng.js";

/**
 * @typedef {Object} LabelledQuery
 * @property {string} id
 * @property {string} q the question as a user would type it
 * @property {"direct"|"paraphrase"|"keyword-poor"|"typo"|"multilingual"|"outdated"|"unanswerable"} type
 * @property {string} lang
 * @property {string[]} gold passage ids that answer it, empty when unanswerable
 * @property {"tuning"|"test"} split
 */

export const QUERY_TYPES = ["direct", "paraphrase", "keyword-poor", "typo", "multilingual", "outdated", "unanswerable"];

/**
 * Parse the authoring file: `type | lang | gold,gold | question` per line.
 * @param {string} text
 * @returns {Omit<LabelledQuery, "id"|"split">[]}
 */
export function parseSource(text) {
  const out = [];
  for (const [n, line] of text.split("\n").entries()) {
    if (!line.trim() || line.startsWith("#")) continue;
    const parts = line.split(" | ");
    if (parts.length < 4) throw new ConfigError(`queries source line ${n + 1} needs 4 fields`);
    const [type, lang, gold, ...rest] = parts;
    if (!QUERY_TYPES.includes(type.trim())) throw new ConfigError(`line ${n + 1}: unknown type ${type}`);
    out.push({ type: /** @type {any} */ (type.trim()), lang: lang.trim(), gold: gold.trim() ? gold.trim().split(",").map((g) => g.trim()) : [], q: rest.join(" | ").trim() });
  }
  return out;
}

/**
 * Assign every query to the tuning or test split. The split is stratified by query type
 * and driven by a seed, so the files can be rebuilt exactly.
 * @param {Omit<LabelledQuery, "id"|"split">[]} items
 * @param {number} seed
 * @param {number} testShare fraction of each type that goes to the test set
 * @returns {LabelledQuery[]}
 */
export function assignSplit(items, seed, testShare = 0.4) {
  const withId = items.map((it, i) => ({ ...it, id: `Q${String(i + 1).padStart(3, "0")}` }));
  /** @type {Map<string, typeof withId>} */
  const byType = new Map();
  for (const it of withId) byType.set(it.type, [...(byType.get(it.type) || []), it]);
  const testIds = new Set();
  let k = 0;
  for (const type of QUERY_TYPES) {
    const group = shuffled(byType.get(type) || [], seed + 101 * ++k);
    const nTest = Math.round(group.length * testShare);
    for (const it of group.slice(0, nTest)) testIds.add(it.id);
  }
  return withId.map((it) => ({ ...it, split: testIds.has(it.id) ? "test" : "tuning" }));
}

/**
 * @param {string} jsonl
 * @returns {LabelledQuery[]}
 */
export function parseQueries(jsonl) {
  return jsonl.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l));
}

/** @param {LabelledQuery[]} qs @param {"tuning"|"test"} split */
export const inSplit = (qs, split) => qs.filter((q) => q.split === split);
