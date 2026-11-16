/**
 * Distinct passage ids in rank order. Several chunks can come from one passage; only the best counts.
 * @param {{passage: string}[]} results
 * @returns {string[]}
 */
export function rankedPassages(results) {
  const seen = new Set();
  const out = [];
  for (const r of results) if (!seen.has(r.passage)) (seen.add(r.passage), out.push(r.passage));
  return out;
}

/** Share of the gold passages found in the top k. */
export function recallAt(ranked, gold, k) {
  if (gold.length === 0) return 0;
  const top = new Set(ranked.slice(0, k));
  return gold.filter((g) => top.has(g)).length / gold.length;
}

/** 1 / rank of the first gold passage, 0 when none is retrieved. */
export function reciprocalRank(ranked, gold) {
  const i = ranked.findIndex((p) => gold.includes(p));
  return i < 0 ? 0 : 1 / (i + 1);
}

/** Binary-relevance nDCG at k. */
export function ndcgAt(ranked, gold, k = 10) {
  if (gold.length === 0) return 0;
  let dcg = 0;
  ranked.slice(0, k).forEach((p, i) => {
    if (gold.includes(p)) dcg += 1 / Math.log2(i + 2);
  });
  let ideal = 0;
  for (let i = 0; i < Math.min(gold.length, k); i++) ideal += 1 / Math.log2(i + 2);
  return dcg / ideal;
}

/**
 * Pick the abstain threshold that maximises balanced accuracy on the given items:
 * (answerable kept + unanswerable abstained) / 2. A query is abstained when confidence < threshold.
 * @param {{confidence: number, answerable: boolean}[]} items
 * @returns {number}
 */
export function tuneThreshold(items) {
  const values = [...new Set(items.map((i) => i.confidence))].sort((a, b) => a - b);
  const na = items.filter((i) => i.answerable).length || 1;
  const nu = items.filter((i) => !i.answerable).length || 1;
  const candidates = [0, ...values.map((v, i) => (i + 1 < values.length ? (v + values[i + 1]) / 2 : v + 1e-6))];
  let best = 0;
  let bestScore = -1;
  for (const t of candidates) {
    const kept = items.filter((i) => i.answerable && i.confidence >= t).length / na;
    const caught = items.filter((i) => !i.answerable && i.confidence < t).length / nu;
    const s = (kept + caught) / 2;
    if (s > bestScore + 1e-12) (bestScore = s, (best = t));
  }
  return best;
}

/**
 * Precision and recall of the abstain decision on unanswerable queries.
 * Precision is 0 when nothing was abstained (no positive predictions).
 * @param {{confidence: number, answerable: boolean}[]} items
 * @param {number} threshold
 */
export function abstainReport(items, threshold) {
  const abstained = items.filter((i) => i.confidence < threshold);
  const truePos = abstained.filter((i) => !i.answerable).length;
  const unanswerable = items.filter((i) => !i.answerable).length;
  const answerable = items.filter((i) => i.answerable);
  return {
    threshold,
    precision: abstained.length ? truePos / abstained.length : 0,
    recall: unanswerable ? truePos / unanswerable : 0,
    answerableKept: answerable.length ? answerable.filter((i) => i.confidence >= threshold).length / answerable.length : 0,
  };
}
