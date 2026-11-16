import { CorpusError } from "./errors.js";

/**
 * @typedef {Object} Section
 * @property {string} num section number as written in the heading ("0" is the preamble)
 * @property {string} heading heading text without the number
 * @property {number} start character offset in the body
 * @property {number} end exclusive end offset
 * @property {string} passage passage id, `${docId}#${num}`
 */

/**
 * @typedef {Object} Block
 * @property {"heading"|"paragraph"|"list"|"table"} kind
 * @property {number} start
 * @property {number} end
 * @property {number} section index into doc.sections
 * @property {{start: number, end: number}[]} units sentences, list items or table rows
 */

/**
 * @typedef {Object} Doc
 * @property {string} id
 * @property {string} family documents with the same family are revisions of one another
 * @property {number} revision
 * @property {string} date ISO date of the revision
 * @property {string} status "current" or "superseded" (display only, never read by retrieval)
 * @property {string} lang "en", "it" or "sq"
 * @property {string} title
 * @property {string} kind
 * @property {string} body markdown text, every offset in the lab refers to this string
 * @property {Section[]} sections
 * @property {Block[]} blocks
 * @property {number} latestRevision highest revision number in the family
 */

const SECTION_RE = /^##\s+(\d+)\.\s+(.*)$/;

/**
 * Split a body into sentence-like spans.
 * @param {string} text
 * @param {number} offset
 * @returns {{start: number, end: number}[]}
 */
export function sentenceSpans(text, offset = 0) {
  const spans = [];
  let from = 0;
  const re = /[.!?]+(?=\s+[A-Z0-9À-Ý])/g;
  let m;
  while ((m = re.exec(text))) {
    spans.push({ start: offset + from, end: offset + m.index + m[0].length });
    from = m.index + m[0].length;
    while (from < text.length && /\s/.test(text[from])) from++;
  }
  if (from < text.length) spans.push({ start: offset + from, end: offset + text.length });
  return spans;
}

/**
 * Parse the body into sections and blocks.
 * @param {string} body
 * @returns {{sections: Section[], blocks: Block[]}}
 */
export function analyseBody(body) {
  const lines = [];
  let pos = 0;
  for (const text of body.split("\n")) {
    lines.push({ text, start: pos, end: pos + text.length });
    pos += text.length + 1;
  }
  /** @type {Section[]} */
  const sections = [{ num: "0", heading: "", start: 0, end: body.length, passage: "" }];
  /** @type {Block[]} */
  const blocks = [];
  let cur = null;
  const flush = () => {
    if (!cur) return;
    const last = cur.lines[cur.lines.length - 1];
    cur.end = last.end;
    cur.units =
      cur.kind === "paragraph"
        ? sentenceSpans(body.slice(cur.start, cur.end), cur.start)
        : cur.lines.map((l) => ({ start: l.start, end: l.end }));
    delete cur.lines;
    blocks.push(cur);
    cur = null;
  };
  for (const ln of lines) {
    const t = ln.text;
    const sec = SECTION_RE.exec(t);
    if (sec) {
      flush();
      sections[sections.length - 1].end = ln.start;
      sections.push({ num: sec[1], heading: sec[2].trim(), start: ln.start, end: body.length, passage: "" });
      blocks.push({ kind: "heading", start: ln.start, end: ln.end, section: sections.length - 1, units: [{ start: ln.start, end: ln.end }] });
      continue;
    }
    if (/^#{1,3}\s/.test(t)) {
      flush();
      blocks.push({ kind: "heading", start: ln.start, end: ln.end, section: sections.length - 1, units: [{ start: ln.start, end: ln.end }] });
      continue;
    }
    if (t.trim() === "") {
      flush();
      continue;
    }
    const kind = t.trimStart().startsWith("|") ? "table" : /^\d+\.\s/.test(t) ? "list" : "paragraph";
    if (cur && cur.kind !== kind) flush();
    if (!cur) cur = { kind, start: ln.start, end: ln.end, section: sections.length - 1, units: [], lines: [] };
    cur.lines.push(ln);
  }
  flush();
  return { sections, blocks };
}

/**
 * Parse one markdown document with a small front matter block.
 * @param {string} raw
 * @returns {Omit<Doc, "latestRevision">}
 */
export function parseDocument(raw) {
  const m = /^---\n([\s\S]*?)\n---\n([\s\S]*)$/.exec(raw.replace(/\r\n/g, "\n"));
  if (!m) throw new CorpusError("document has no front matter");
  /** @type {Record<string, string>} */
  const meta = {};
  for (const line of m[1].split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) meta[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  for (const key of ["id", "family", "revision", "date", "status", "lang", "title", "kind"]) {
    if (!meta[key]) throw new CorpusError(`document is missing "${key}" in its front matter`);
  }
  return { ...meta, revision: Number(meta.revision), body: m[2].trim() + "\n", ...hydrateParts(meta.id, m[2].trim() + "\n") };
}

/** @param {string} id @param {string} body */
function hydrateParts(id, body) {
  const { sections, blocks } = analyseBody(body);
  for (const s of sections) s.passage = `${id}#${s.num}`;
  return { sections, blocks };
}

/**
 * Build a corpus from parsed documents. Adds the latest revision number of each family.
 * @param {Omit<Doc, "latestRevision">[]} parsed
 */
export function buildCorpus(parsed) {
  const ids = new Set();
  const latest = new Map();
  for (const d of parsed) {
    if (ids.has(d.id)) throw new CorpusError(`duplicate document id ${d.id}`);
    ids.add(d.id);
    latest.set(d.family, Math.max(latest.get(d.family) || 0, d.revision));
  }
  /** @type {Doc[]} */
  const docs = parsed.map((d) => ({ ...d, latestRevision: latest.get(d.family) }));
  docs.sort((a, b) => (a.id < b.id ? -1 : 1));
  const byId = new Map(docs.map((d) => [d.id, d]));
  /** @type {Map<string, {doc: Doc, section: Section}>} */
  const passages = new Map();
  for (const d of docs) for (const s of d.sections) passages.set(s.passage, { doc: d, section: s });
  return { docs, byId, passages };
}

/**
 * Serialisable form: front matter fields and the body only.
 * @param {ReturnType<typeof buildCorpus>} corpus
 */
export function corpusToJSON(corpus) {
  return {
    docs: corpus.docs.map((d) => ({
      id: d.id, family: d.family, revision: d.revision, date: d.date, status: d.status,
      lang: d.lang, title: d.title, kind: d.kind, body: d.body,
    })),
  };
}

/**
 * Rebuild a corpus from `corpusToJSON` output (what the browser fetches).
 * @param {{docs: any[]}} json
 */
export function corpusFromJSON(json) {
  if (!json || !Array.isArray(json.docs)) throw new CorpusError("corpus JSON has no docs array");
  return buildCorpus(json.docs.map((d) => ({ ...d, ...hydrateParts(d.id, d.body) })));
}

/** @typedef {ReturnType<typeof buildCorpus>} Corpus */
