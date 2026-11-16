import { tokenize } from "./text.js";
import { rankedPassages } from "./metrics.js";

/**
 * Explain, from the data alone, why an answerable query was missed.
 * @param {import("./queries.js").LabelledQuery} q
 * @param {ReturnType<import("./pipeline.js").System["search"]>} out
 * @param {import("./corpus.js").Corpus} corpus
 * @param {{threshold: number}} abstain
 * @returns {{kind: string, reason: string, goldRank: number|null}}
 */
export function diagnose(q, out, corpus, abstain) {
  const ranked = rankedPassages(out.results);
  const ranks = q.gold.map((g) => ranked.indexOf(g) + 1 || null);
  const goldRank = ranks.filter(Boolean).sort((a, b) => a - b)[0] ?? null;
  const topPassage = corpus.passages.get(ranked[0]);
  const goldPassages = q.gold.map((g) => corpus.passages.get(g));
  if (out.confidence < abstain.threshold) {
    return { kind: "abstained", goldRank, reason: `Confidence ${out.confidence.toFixed(2)} is below the abstain threshold ${abstain.threshold.toFixed(2)}, so a question that has an answer was refused.` };
  }
  const queryTerms = new Set(out.analysis.terms.filter((t) => t.source !== "synonym").map((t) => t.term));
  const goldTerms = new Set(goldPassages.flatMap((p) => tokenize(p.doc.body.slice(p.section.start, p.section.end))));
  const shared = [...queryTerms].filter((t) => goldTerms.has(t));
  const langGap = goldPassages.some((p) => p.doc.lang !== q.lang);
  if (shared.length === 0) {
    return { kind: langGap ? "cross-language" : "vocabulary gap", goldRank, reason: langGap ? `The question is in ${q.lang} and the answer is in a ${goldPassages[0].doc.lang} document: no query term occurs in the gold passage.` : "No query term occurs in the gold passage after folding and typo matching, so no lexical method can rank it." };
  }
  if (topPassage && goldPassages.some((p) => p.doc.family === topPassage.doc.family && p.doc.id !== topPassage.doc.id)) {
    return { kind: "outdated revision", goldRank, reason: `The top result is ${topPassage.doc.id}, another revision of the same document.` };
  }
  const top = topPassage ? `${topPassage.doc.title}, section ${topPassage.section.num}` : "nothing";
  const topShared = topPassage ? [...queryTerms].filter((t) => tokenize(topPassage.doc.body.slice(topPassage.section.start, topPassage.section.end)).includes(t)) : [];
  return {
    kind: goldRank ? "ranked too low" : "outranked",
    goldRank,
    reason: `${goldRank ? `The answer is at rank ${goldRank}, ` : "The answer is outside the top 20, "}behind "${top}", which matches ${topShared.length} of ${queryTerms.size} query terms (${topShared.slice(0, 4).join(", ")}); the gold passage matches ${shared.length} (${shared.slice(0, 4).join(", ")}).`,
  };
}
