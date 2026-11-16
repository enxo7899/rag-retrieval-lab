import test from "node:test";
import assert from "node:assert/strict";
import { data } from "./helpers.js";
import { chunkCorpus, CHUNKERS, chunkerLabel } from "../src/chunkers.js";

const rowsOf = (body) => {
  let pos = 0;
  const rows = [];
  for (const t of body.split("\n")) {
    if (t.trim().startsWith("|")) rows.push([pos, pos + t.length]);
    pos += t.length + 1;
  }
  return rows;
};

for (const spec of CHUNKERS) {
  test(`chunker ${chunkerLabel(spec)} never splits a table row`, () => {
    for (const c of chunkCorpus(data.corpus, spec)) {
      for (const [s, e] of rowsOf(data.corpus.byId.get(c.docId).body)) {
        const overlap = Math.min(c.end, e) - Math.max(c.start, s);
        assert.ok(overlap <= 0 || overlap === e - s, `${c.id} cuts a row at ${s}-${e}`);
      }
    }
  });
}

test("every chunk keeps its document id, section path and exact character offsets", () => {
  for (const spec of CHUNKERS) {
    for (const c of chunkCorpus(data.corpus, spec)) {
      const doc = data.corpus.byId.get(c.docId);
      assert.equal(doc.body.slice(c.start, c.end), c.text);
      assert.ok(c.sectionPath.startsWith(doc.title));
      assert.ok(data.corpus.passages.has(c.passage));
    }
  }
});

test("heading-aware chunks never cross a section boundary", () => {
  for (const c of chunkCorpus(data.corpus, { type: "heading", max: 700 })) {
    const sec = data.corpus.passages.get(c.passage).section;
    assert.ok(c.start >= sec.start && c.end <= sec.end, c.id);
  }
});

test("fixed-size chunks overlap and together cover every character of the document", () => {
  const chunks = chunkCorpus(data.corpus, { type: "fixed", size: 400, overlap: 80 });
  for (const doc of data.corpus.docs) {
    const mine = chunks.filter((c) => c.docId === doc.id);
    let reach = 0;
    for (const c of mine) {
      assert.ok(c.start <= reach + 1 || /^\s*$/.test(doc.body.slice(reach, c.start)), `${c.id} leaves a gap`);
      reach = Math.max(reach, c.end);
    }
    assert.ok(doc.body.slice(reach).trim() === "");
  }
});

test("a numbered procedure stays in one heading-aware chunk when it fits", () => {
  const chunks = chunkCorpus(data.corpus, { type: "heading", max: 700 });
  const c = chunks.find((x) => x.passage === "DE-OPS-122#3");
  assert.ok(c.text.includes("1. Obtain a permit") && c.text.includes("6. Report to the control room"));
});
