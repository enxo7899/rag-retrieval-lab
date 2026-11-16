import { ConfigError } from "./errors.js";

/**
 * @typedef {Object} Chunk
 * @property {string} id unique within one chunking run
 * @property {string} docId
 * @property {number} start character offset in the document body
 * @property {number} end exclusive end offset
 * @property {string} text body.slice(start, end), exactly
 * @property {string} sectionPath "Document title > 3. Procedure"
 * @property {string} passage id of the section that holds most of the chunk, `${docId}#${num}`
 * @property {string} context heading text that is indexed together with the chunk (empty for chunkers that do not add it)
 */

/**
 * @typedef {{type: "fixed", size: number, overlap: number}
 *   | {type: "sentence", window: number, stride: number}
 *   | {type: "heading", max: number}} ChunkerSpec
 */

/** @param {ChunkerSpec} s */
export function chunkerLabel(s) {
  if (s.type === "fixed") return `fixed-${s.size}-${s.overlap}`;
  if (s.type === "sentence") return `sentence-${s.window}-${s.stride}`;
  return `heading-${s.max}`;
}

/** The chunkers compared in the grid. */
export const CHUNKERS = /** @type {ChunkerSpec[]} */ ([
  { type: "fixed", size: 400, overlap: 80 },
  { type: "fixed", size: 900, overlap: 180 },
  { type: "sentence", window: 3, stride: 2 },
  { type: "heading", max: 250 },
  { type: "heading", max: 700 },
]);

/** @param {string} body @param {number} pos start and end offsets of the line holding pos */
function lineAt(body, pos) {
  const start = body.lastIndexOf("\n", pos - 1) + 1;
  let end = body.indexOf("\n", pos);
  if (end < 0) end = body.length;
  return { start, end };
}

/** @param {string} body @param {number} pos true when pos sits inside a table row */
function inTableRow(body, pos) {
  const { start, end } = lineAt(body, Math.min(pos, body.length - 1));
  return pos > start && pos < end && body.slice(start, end).trimStart().startsWith("|");
}

/** Section that holds most of the span. */
function dominantSection(doc, start, end) {
  let best = doc.sections[0];
  let bestOverlap = -1;
  for (const s of doc.sections) {
    const o = Math.min(end, s.end) - Math.max(start, s.start);
    if (o > bestOverlap) {
      best = s;
      bestOverlap = o;
    }
  }
  return best;
}

/**
 * @param {import("./corpus.js").Doc} doc
 * @param {number} n counter within the document
 * @param {number} start
 * @param {number} end
 * @param {boolean} withContext
 * @returns {Chunk}
 */
function makeChunk(doc, n, start, end, withContext) {
  while (start < end && /\s/.test(doc.body[start])) start++;
  while (end > start && /\s/.test(doc.body[end - 1])) end--;
  const sec = dominantSection(doc, start, end);
  const heading = sec.heading ? `${sec.num}. ${sec.heading}` : "";
  return {
    id: `${doc.id}:${n}`,
    docId: doc.id,
    start,
    end,
    text: doc.body.slice(start, end),
    sectionPath: heading ? `${doc.title} > ${heading}` : doc.title,
    passage: sec.passage,
    context: withContext ? `${doc.title} ${sec.heading}`.trim() : "",
  };
}

/**
 * Fixed-size windows with overlap. Boundaries move to word edges, and a boundary that
 * falls inside a table row moves to the row edge, so a row is never cut in two.
 * @param {import("./corpus.js").Doc} doc
 * @param {number} size
 * @param {number} overlap
 */
function fixedChunks(doc, size, overlap) {
  const body = doc.body;
  const out = [];
  let start = 0;
  let n = 0;
  while (start < body.length) {
    let end = Math.min(body.length, start + size);
    if (end < body.length) {
      const row = lineAt(body, end - 1);
      if (body.slice(row.start, row.end).trimStart().startsWith("|")) end = row.end;
      else {
        const back = body.slice(start, end).search(/\s\S*$/);
        if (back > size * 0.5) end = start + back;
      }
    }
    const chunk = makeChunk(doc, n, start, end, false);
    if (chunk.text) out.push(chunk), n++;
    if (end >= body.length) break;
    let next = Math.max(start + 1, end - overlap);
    while (next < end && !/\s/.test(body[next - 1] || " ")) next++;
    if (inTableRow(body, next)) next = lineAt(body, next).start;
    start = next > start ? next : end;
  }
  return out;
}

/**
 * Sliding windows over sentences, list items and table rows within a section.
 * @param {import("./corpus.js").Doc} doc
 * @param {number} window
 * @param {number} stride
 */
function sentenceChunks(doc, window, stride) {
  const out = [];
  let n = 0;
  doc.sections.forEach((_, si) => {
    const units = doc.blocks.filter((b) => b.section === si && b.kind !== "heading").flatMap((b) => b.units);
    for (let i = 0; i < units.length; i += stride) {
      const slice = units.slice(i, i + window);
      out.push(makeChunk(doc, n++, slice[0].start, slice[slice.length - 1].end, false));
      if (i + window >= units.length) break;
    }
  });
  return out;
}

/**
 * Heading-aware: blocks are packed into chunks up to `max` characters, never across a section
 * boundary. Tables and numbered lists stay whole. The document title and section heading
 * are indexed with the chunk.
 * @param {import("./corpus.js").Doc} doc
 * @param {number} max
 */
function headingChunks(doc, max) {
  const out = [];
  let n = 0;
  doc.sections.forEach((_, si) => {
    const blocks = doc.blocks.filter((b) => b.section === si);
    /** @type {{start: number, end: number}[]} */
    const pieces = [];
    for (const b of blocks) {
      if (b.kind === "paragraph" && b.end - b.start > max) {
        let from = b.units[0].start;
        let to = from;
        for (const u of b.units) {
          if (u.end - from > max && to > from) {
            pieces.push({ start: from, end: to });
            from = u.start;
          }
          to = u.end;
        }
        pieces.push({ start: from, end: to });
      } else pieces.push({ start: b.start, end: b.end });
    }
    let cur = null;
    for (const p of pieces) {
      if (cur && p.end - cur.start > max) {
        out.push(makeChunk(doc, n++, cur.start, cur.end, true));
        cur = null;
      }
      cur = cur ? { start: cur.start, end: p.end } : { ...p };
    }
    if (cur) out.push(makeChunk(doc, n++, cur.start, cur.end, true));
  });
  return out;
}

/**
 * Chunk every document in a corpus.
 * @param {import("./corpus.js").Corpus} corpus
 * @param {ChunkerSpec} spec
 * @returns {Chunk[]}
 */
export function chunkCorpus(corpus, spec) {
  const run = (doc) => {
    if (spec.type === "fixed") return fixedChunks(doc, spec.size, spec.overlap);
    if (spec.type === "sentence") return sentenceChunks(doc, spec.window, spec.stride);
    if (spec.type === "heading") return headingChunks(doc, spec.max);
    throw new ConfigError(`unknown chunker type ${/** @type {any} */ (spec).type}`);
  };
  return corpus.docs.flatMap(run).filter((c) => c.text.length > 0);
}
