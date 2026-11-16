import { corpusFromJSON } from "./src/corpus.js";
import { parseQueries, inSplit } from "./src/queries.js";
import { createLab, configLabel } from "./src/pipeline.js";
import { CHUNKERS, chunkerLabel } from "./src/chunkers.js";
import { tokenSpans } from "./src/text.js";
import { runQueries, scoreRuns, selectBest } from "./src/grid.js";
import { tuneThreshold } from "./src/metrics.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const f2 = (x) => x.toFixed(2);
const f3 = (x) => x.toFixed(3);

const [corpusJson, queriesText, lsa, results] = await Promise.all([
  fetch("data/corpus.json").then((r) => r.json()),
  fetch("data/queries.jsonl").then((r) => r.text()),
  fetch("data/lsa.json").then((r) => r.json()),
  fetch("docs/results.json").then((r) => r.json()),
]);
const corpus = corpusFromJSON(corpusJson);
const queries = parseQueries(queriesText);
const tuning = inSplit(queries, "tuning");
const lab = createLab(corpus, lsa);
const rows = results.rows;
const chosen = rows[results.chosen_id];
const baseline = rows[results.baseline_id];
const params = new URLSearchParams(location.search);

$("meta").innerHTML = `<b>${results.corpus.documents} documents</b> in English, Italian and Albanian, ${results.corpus.passages} sections, ${results.corpus.families_with_revisions} documents with older revisions. <b>${results.queries.total} labelled questions</b>: ${results.queries.tuning} for tuning, ${results.queries.test} held out for the test. ${results.grid_size} configurations compared.`;

// ---------- configuration state per side ----------
const chunkerIndex = (spec) => CHUNKERS.findIndex((c) => chunkerLabel(c) === chunkerLabel(spec));
const fromRow = (r) => ({ chunker: chunkerIndex(r.chunker), retriever: r.retriever, k1: r.k1 ?? 1.2, b: r.b ?? 0.75, rerank: r.rerank, expand: r.expand });
const sides = { a: fromRow(baseline), b: fromRow(chosen) };
const toConfig = (s) => ({ chunker: CHUNKERS[s.chunker], retriever: s.retriever, k1: s.k1, b: s.b, rerank: s.rerank, expand: s.expand });
const thresholds = new Map();
function thresholdFor(config) {
  const key = configLabel(config);
  const row = rows.find((r) => r.label === key);
  if (row) return row.threshold;
  if (!thresholds.has(key)) thresholds.set(key, tuneThreshold(scoreRuns(runQueries(lab.build(config), tuning)).items));
  return thresholds.get(key);
}

let question = "";
const goldFor = (q) => queries.find((x) => x.q.toLowerCase() === q.trim().toLowerCase())?.gold || null;

// ---------- the bench ----------
function highlight(text, terms, analyzer) {
  let out = "";
  let at = 0;
  for (const t of tokenSpans(text, analyzer)) {
    if (!terms.has(t.term)) continue;
    out += esc(text.slice(at, t.start)) + `<span class="q">${esc(text.slice(t.start, t.end))}</span>`;
    at = t.end;
  }
  return out + esc(text.slice(at));
}

function docView(chunk, terms, analyzer) {
  const body = corpus.byId.get(chunk.docId).body;
  return `${highlight(body.slice(0, chunk.start), terms, analyzer)}<mark class="chunk">${highlight(body.slice(chunk.start, chunk.end), terms, analyzer)}</mark>${highlight(body.slice(chunk.end), terms, analyzer)}`;
}

function breakdown(sys, out, r) {
  const cfg = sys.config;
  const terms = out.analysis.terms;
  const bm = (retriever) => retriever.explain(terms, r.index);
  let h = "";
  const termTable = (list) =>
    list.length
      ? `<table class="break-table"><tr><th>BM25 term</th><th>how found</th><th class="n">tf</th><th class="n">idf</th><th class="n">score</th></tr>${list.map((x) => `<tr><td>${esc(x.term)}</td><td>${x.source}</td><td class="n">${x.tf}</td><td class="n">${f2(x.idf)}</td><td class="n">${f3(x.contribution)}</td></tr>`).join("")}</table>`
      : `<p class="note">No query term occurs in this chunk.</p>`;
  if (cfg.retriever === "bm25") h += termTable(bm(sys.retriever));
  if (cfg.retriever === "tfidf") h += `<p class="mono">TF-IDF cosine ${f3(r.score)}</p>`;
  if (cfg.retriever === "dense") h += `<p class="mono">Latent-space cosine ${f3(r.score)}</p>`;
  if (cfg.retriever === "hybrid") {
    const fus = out.run.fusion(r.index);
    const rk = (x) => (x ? `rank ${x}` : "not retrieved");
    h += termTable(bm(sys.retriever.parts[0]));
    h += `<p class="mono">Dense cosine ${f3(out.run.components[1].scores[r.index])}. Fusion: BM25 ${rk(fus.ranks[0])}, dense ${rk(fus.ranks[1])}, reciprocal rank score ${f3(fus.score)}.</p>`;
  }
  if (r.features) {
    const x = r.features;
    h += `<table class="break-table"><tr><th>Rerank feature</th><th class="n">value</th></tr><tr><td>retriever score, relative</td><td class="n">${f2(x.base)}</td></tr><tr><td>term proximity</td><td class="n">${f2(x.proximity)}</td></tr><tr><td>section title match</td><td class="n">${f2(x.title)}</td></tr><tr><td>revision recency</td><td class="n">${f2(x.recency)}</td></tr><tr><td>reranked score (was position ${r.before}, now ${r.rank})</td><td class="n">${f2(r.rerankScore)}</td></tr></table>`;
  }
  return h;
}

function renderSide(key) {
  const s = sides[key];
  const config = toConfig(s);
  const sys = lab.build(config);
  const out = sys.search(question);
  const threshold = thresholdFor(config);
  const abstain = out.confidence < threshold;
  const gold = goldFor(question);
  const terms = new Set(out.analysis.terms.filter((t) => t.source !== "synonym").map((t) => t.term));
  const seen = new Set();
  const shown = out.results.filter((r) => (seen.has(r.passage) ? false : seen.add(r.passage))).slice(0, 5);
  const verdict = !out.results.length
    ? `<p class="verdict abstain">Abstained. No indexed term matches this question.</p>`
    : abstain
      ? `<p class="verdict abstain">Abstained: confidence ${f2(out.confidence)} is below the threshold ${f2(threshold)}. The passages below are what it would have shown.</p>`
      : `<p class="verdict answer">Answering with <b>confidence ${f2(out.confidence)}</b>, above the threshold ${f2(threshold)}.</p>`;
  const how = out.analysis.terms.filter((t) => t.source !== "query");
  const resolved = how.length ? `<p class="note">Query handling: ${how.map((t) => `${esc(t.from)} to ${esc(t.term)} (${t.source})`).join(", ")}.${out.analysis.unknown.length ? ` Not in the index: ${out.analysis.unknown.map(esc).join(", ")}.` : ""}</p>` : out.analysis.unknown.length ? `<p class="note">Not in the index: ${out.analysis.unknown.map(esc).join(", ")}.</p>` : "";
  const list = shown
    .map((r, i) => {
      const doc = corpus.byId.get(r.chunk.docId);
      const old = doc.revision < doc.latestRevision;
      const isGold = gold && gold.includes(r.passage);
      const analyzer = { fold: config.fold !== false };
      return `<li class="res${isGold ? " gold" : ""}"><div class="rhead"><span class="rk">${i + 1}</span><div><span class="title">${esc(doc.title)}</span> <span class="rev${old ? " old" : ""}">${doc.id}, revision ${doc.revision}${old ? " (superseded)" : doc.latestRevision > 1 ? " (current)" : ""}</span><div class="path">${esc(r.chunk.sectionPath)}</div></div><span class="sc">${f3(r.score)}</span></div>
      ${isGold ? `<p class="tag" style="margin:2px 0 0 34px">This passage answers the question.</p>` : ""}
      <p class="snip">${highlight((r.chunk.text.length > 280 ? r.chunk.text.slice(0, 280) + "…" : r.chunk.text).replace(/^#+\s*/gm, "").replace(/\n+/g, " "), terms, analyzer)}</p>
      <details${i === 0 ? " open" : ""}><summary>Score breakdown</summary>${breakdown(sys, out, r)}</details>
      <details${i === 0 && params.get("run") ? " open" : ""}><summary>Show in document</summary><pre class="doc">${docView(r.chunk, terms, analyzer)}</pre></details></li>`;
    })
    .join("");
  return `${verdict}${resolved}<ol class="results${abstain ? " dim" : ""}">${list}</ol>`;
}

function controls(key) {
  const s = sides[key];
  const opt = (arr, cur, fmt) => arr.map((v, i) => `<option value="${i}"${i === cur ? " selected" : ""}>${fmt(v)}</option>`).join("");
  const bm = s.retriever === "bm25" || s.retriever === "hybrid";
  return `<div class="cfg">
    <label>Chunker <select data-s="${key}" data-k="chunker" aria-label="Chunker, side ${key}">${opt(CHUNKERS, s.chunker, chunkerLabel)}</select></label>
    <label>Retriever <select data-s="${key}" data-k="retriever" aria-label="Retriever, side ${key}">${["bm25", "tfidf", "dense", "hybrid"].map((r) => `<option${r === s.retriever ? " selected" : ""}>${r}</option>`).join("")}</select></label>
    ${bm ? `<label>k1 <input type="number" step="0.1" min="0.2" max="3" value="${s.k1}" data-s="${key}" data-k="k1" aria-label="BM25 k1, side ${key}"></label><label>b <input type="number" step="0.05" min="0" max="1" value="${s.b}" data-s="${key}" data-k="b" aria-label="BM25 b, side ${key}"></label>` : ""}
    <label><input type="checkbox" data-s="${key}" data-k="rerank" ${s.rerank ? "checked" : ""}> Reranker</label>
    <label><input type="checkbox" data-s="${key}" data-k="expand" ${s.expand ? "checked" : ""}> Synonyms</label>
  </div>
  <div class="cfg"><button class="link" type="button" data-preset="${key}:baseline">Use the baseline</button><button class="link" type="button" data-preset="${key}:chosen">Use the chosen configuration</button></div>`;
}

function renderBench() {
  const head = (key, title) => `<div class="side-head"><h3>${title}</h3>${controls(key)}</div>`;
  if (!question.trim()) {
    $("sides").innerHTML = `<div class="c12" style="grid-column:1/-1"><div class="empty"><p>Nothing asked yet.</p><p>Pick one of the questions above, or type your own, to see two configurations answer it. A good first try is the outdated revision example: the baseline returns the superseded document and the chosen configuration does not.</p></div></div>` .replace('class="c12" style="grid-column:1/-1"', "");
    $("sides").style.display = "block";
    return;
  }
  $("sides").style.display = "";
  $("sides").innerHTML = `<div>${head("a", "Configuration A")}<div id="resA"></div></div><div>${head("b", "Configuration B")}<div id="resB"></div></div>`;
  $("resA").innerHTML = renderSide("a");
  $("resB").innerHTML = renderSide("b");
}

function ask(q) {
  question = q;
  $("q").value = q;
  renderBench();
}

$("go").onclick = () => ask($("q").value);
$("q").addEventListener("keydown", (e) => e.key === "Enter" && ask($("q").value));
$("sides").addEventListener("change", (e) => {
  const t = e.target;
  if (!t.dataset.s) return;
  const s = sides[t.dataset.s];
  const k = t.dataset.k;
  s[k] = k === "rerank" || k === "expand" ? t.checked : k === "retriever" ? t.value : Number(t.value);
  const y = scrollY;
  renderBench();
  scrollTo(0, y);
});
$("sides").addEventListener("click", (e) => {
  const p = e.target.dataset?.preset;
  if (!p) return;
  const [k, which] = p.split(":");
  sides[k] = fromRow(which === "chosen" ? chosen : baseline);
  renderBench();
});

const firstOf = (type) => tuning.find((q) => q.type === type);
const chipQs = [firstOf("direct"), firstOf("paraphrase"), firstOf("typo"), firstOf("multilingual"), tuning.find((q) => q.lang === "sq"), firstOf("unanswerable")].filter(Boolean);
$("chips").innerHTML = `<span class="label">Labelled questions to try</span>` + chipQs.map((q) => `<button class="link" type="button" data-q="${esc(q.q)}">${esc(q.q)}</button>`).join("");
$("chips").addEventListener("click", (e) => e.target.dataset.q && ask(e.target.dataset.q));

const SCENARIOS = {
  unanswerable: () => {
    sides.a = fromRow(baseline);
    sides.b = fromRow(chosen);
    ask("What is the fine for stealing electricity?");
  },
  outdated: () => {
    sides.b = fromRow(chosen);
    sides.a = { ...fromRow(chosen), rerank: false };
    ask("What type of tamper-evident seal does a technician carry?");
  },
  leak: () => {
    document.querySelector('input[value="test"]').checked = true;
    mode = "test";
    renderGrid();
    $("grid").scrollIntoView();
  },
};
document.querySelectorAll("[data-try]").forEach((b) => (b.onclick = () => SCENARIOS[b.dataset.try]()));

// ---------- the grid ----------
let mode = "tuning";
let sortKey = "tuning";
let sortDir = -1;
let filter = "all";
const SORTS = {
  config: (r) => r.label,
  tuning: (r) => r.tuning.ndcg10,
  test: (r) => r.test.ndcg10.mean,
  mrr: (r) => r.test.mrr.mean,
  r5: (r) => r.test.recall5.mean,
  r10: (r) => r.test.recall10.mean,
  ap: (r) => r.test.abstain.precision,
  ar: (r) => r.test.abstain.recall,
  ms: (r) => r.latencyMs.p50,
  size: (r) => r.indexBytes,
};
const pickKey = (m) => (r) => (m === "test" ? { ndcg10: r.test.ndcg10.mean, mrr: r.test.mrr.mean, size: r.indexBytes, label: r.label } : { ndcg10: r.tuning.ndcg10, mrr: r.tuning.mrr, size: r.indexBytes, label: r.label });
const pct = (x) => (x * 100).toFixed(1);

function renderGrid() {
  const pick = selectBest(rows, pickKey(mode));
  const honest = chosen;
  $("gridnote").innerHTML = `Every configuration is built and scored here. Selection looks at the <b>tuning</b> questions only (${results.queries.tuning}); the <b>test</b> questions (${results.queries.test}) are scored once and never used to choose. Intervals are seeded bootstrap 95% over ${results.bootstrap_resamples} resamples.`;
  const ci = (m) => `${f3(m.mean)} <span class="ci">${f3(m.lo)} to ${f3(m.hi)}</span>`;
  $("state").innerHTML =
    mode === "tuning"
      ? `Chosen on the tuning set, reported on the test set: <b>${esc(pick.label)}</b>. Test nDCG@10 ${ci(pick.test.ndcg10)}.`
      : `Chosen by looking at the test set (leakage): <b>${esc(pick.label)}</b>. Its test nDCG@10 is ${ci(pick.test.ndcg10)}, which is no longer an honest estimate.`;
  const gap = pick.test.ndcg10.mean - honest.test.ndcg10.mean;
  const lk = rows[results.leaked_id];
  $("leak").innerHTML = `<div class="wrap"><table class="kv"><tr><th>Reported on the test set</th><th class="n">Chosen on tuning</th><th class="n">Chosen on test</th><th class="n">Difference</th></tr>
    <tr><td>nDCG@10</td><td class="n">${f3(honest.test.ndcg10.mean)}</td><td class="n">${f3(lk.test.ndcg10.mean)}</td><td class="n">${(lk.test.ndcg10.mean - honest.test.ndcg10.mean >= 0 ? "+" : "") + f3(lk.test.ndcg10.mean - honest.test.ndcg10.mean)}</td></tr>
    <tr><td>Unanswerable questions caught (recall), threshold from tuning vs from test</td><td class="n">${pct(honest.test.abstain.recall)}%</td><td class="n">${pct(honest.test.abstainLeaked.recall)}%</td><td class="n">+${pct(honest.test.abstainLeaked.recall - honest.test.abstain.recall)} points</td></tr>
    <tr><td>Chosen configuration: score on the tuning set vs on the test set</td><td class="n">${f3(honest.tuning.ndcg10)}</td><td class="n">${f3(honest.test.ndcg10.mean)}</td><td class="n">${(honest.test.ndcg10.mean - honest.tuning.ndcg10).toFixed(3)}</td></tr></table></div>
    <p class="note">${mode === "test" ? `Picking the best of ${rows.length} configurations on the test set moved the headline by ${gap >= 0 ? "+" : ""}${f3(gap)} nDCG@10 here. That is small because the best configurations are close; the number is still optimistic by construction, and the abstain threshold shows the effect more clearly.` : "Switch to the test set to see the leakage version of this table."}</p>`;
  const opts = ["all", "bm25", "tfidf", "dense", "hybrid"];
  $("filters").innerHTML = `<span class="label">Show</span>` + opts.map((o) => `<button class="link" type="button" data-f="${o}"${o === filter ? ' style="font-weight:700;text-decoration:none"' : ""}>${o === "all" ? "all retrievers" : o}</button>`).join("");
  const cols = [["config", "Configuration", ""], ["tuning", "Tuning nDCG", "n"], ["test", "Test nDCG@10 (95%)", "n"], ["mrr", "MRR", "n"], ["r5", "R@5", "n"], ["r10", "R@10", "n"], ["ap", "Abst. P", "n"], ["ar", "Abst. R", "n"], ["ms", "ms", "n"], ["size", "KB", "n"]];
  const view = rows.filter((r) => filter === "all" || r.retriever === filter).sort((a, b) => {
    const x = SORTS[sortKey](a);
    const y = SORTS[sortKey](b);
    return (x < y ? -1 : x > y ? 1 : 0) * sortDir;
  });
  $("table").innerHTML = `<thead><tr>${cols.map(([k, t, c]) => `<th class="${c}" aria-sort="${k === sortKey ? (sortDir > 0 ? "ascending" : "descending") : "none"}"><button type="button" data-sort="${k}">${t}</button></th>`).join("")}</tr></thead><tbody>${view
    .map(
      (r) => `<tr class="${r.id === pick.id ? "chosen" : r.id === results.baseline_id ? "base" : ""}"><td class="cfgname">${esc(r.label)}${r.id === pick.id ? (mode === "tuning" ? " (chosen on tuning set)" : " (chosen on test set)") : r.id === results.baseline_id ? " (baseline)" : ""}</td><td class="n">${f3(r.tuning.ndcg10)}</td><td class="n">${f3(r.test.ndcg10.mean)} <span class="ci">${f3(r.test.ndcg10.lo)} to ${f3(r.test.ndcg10.hi)}</span></td><td class="n">${f3(r.test.mrr.mean)}</td><td class="n">${f3(r.test.recall5.mean)}</td><td class="n">${f3(r.test.recall10.mean)}</td><td class="n">${f2(r.test.abstain.precision)}</td><td class="n">${f2(r.test.abstain.recall)}</td><td class="n">${r.latencyMs.p50.toFixed(2)}</td><td class="n">${(r.indexBytes / 1024).toFixed(0)}</td></tr>`,
    )
    .join("")}</tbody>`;
}
$("table").addEventListener("click", (e) => {
  const k = e.target.dataset?.sort;
  if (!k) return;
  sortDir = sortKey === k ? -sortDir : k === "config" || k === "ms" || k === "size" ? 1 : -1;
  sortKey = k;
  renderGrid();
});
$("filters").addEventListener("click", (e) => {
  if (e.target.dataset?.f) (filter = e.target.dataset.f, renderGrid());
});
document.querySelectorAll('input[name="mode"]').forEach((i) => i.addEventListener("change", () => ((mode = i.value), renderGrid())));

$("ablate").innerHTML = `<tr><th>Variant of the chosen configuration</th><th class="n">nDCG@10 (95% interval)</th><th class="n">MRR</th><th class="n">Recall@10</th></tr>${results.ablations
  .map((a) => `<tr><td>${esc(a.label)}</td><td class="n">${f3(a.ndcg10.mean)} <span class="ci">${f3(a.ndcg10.lo)} to ${f3(a.ndcg10.hi)}</span></td><td class="n">${f3(a.mrr.mean)}</td><td class="n">${f3(a.recall10.mean)}</td></tr>`)
  .join("")}`;
$("bytype").innerHTML = `<tr><th>Kind of question (test)</th><th class="n">Questions</th><th class="n">nDCG@10</th><th class="n">Recall@10</th><th class="n">Abstained</th></tr>${results.by_type
  .map((t) => `<tr><td>${t.type}</td><td class="n">${t.n}</td><td class="n">${t.ndcg10 == null ? "n/a" : f3(t.ndcg10)}</td><td class="n">${t.recall10 == null ? "n/a" : f3(t.recall10)}</td><td class="n">${t.abstained == null ? "n/a" : pct(t.abstained) + "%"}</td></tr>`)
  .join("")}`;

// ---------- failure gallery ----------
let fkind = "all";
function renderFailures() {
  const kinds = ["all", ...new Set(results.failures.map((x) => x.kind))];
  $("ffilters").innerHTML = `<span class="label">${results.failures.length} of ${results.queries.test - results.queries.unanswerable_test} answerable test questions</span>` + kinds.map((k) => `<button class="link" type="button" data-k="${k}"${k === fkind ? ' style="font-weight:700;text-decoration:none"' : ""}>${k}</button>`).join("");
  $("fails").innerHTML = results.failures
    .filter((x) => fkind === "all" || x.kind === fkind)
    .map((x) => `<div class="fail"><p class="qq">${esc(x.q)}</p><dl><dt>Kind</dt><dd><span class="kind">${x.kind}</span>, ${x.type} question in ${x.lang}</dd><dt>Why</dt><dd>${esc(x.reason)}</dd><dt>Should have found</dt><dd>${x.goldTitles.map(esc).join("; ")}</dd><dt>Returned first</dt><dd>${x.top.map(esc).join("; ")}</dd><dt>Try it</dt><dd><button class="link" type="button" data-ask="${esc(x.q)}">Open in the bench</button></dd></dl></div>`)
    .join("");
}
$("ffilters").addEventListener("click", (e) => e.target.dataset?.k && ((fkind = e.target.dataset.k), renderFailures()));
$("fails").addEventListener("click", (e) => {
  if (!e.target.dataset?.ask) return;
  ask(e.target.dataset.ask);
  $("bench").scrollIntoView();
});

renderGrid();
renderFailures();
renderBench();
const run = params.get("run");
if (run === "unanswerable" || run === "outdated") SCENARIOS[run]();
else if (run === "leak") SCENARIOS.leak();
else if (run === "ask") ask("How much does a household pay for reconnection within 24 hours?");
else if (params.get("q")) ask(params.get("q"));
document.documentElement.dataset.ready = "1";
