import test from "node:test";
import assert from "node:assert/strict";
import { tokenize, fold } from "../src/text.js";
import { editDistance, fuzzyMatch, analyzeQuery, compileSynonyms } from "../src/query.js";

test("Albanian e-diaeresis and plain e match after folding", () => {
  assert.deepEqual(tokenize("matësi"), tokenize("matesi"));
  assert.equal(fold("Çfarë"), "cfare");
});

test("without folding an accented word and its plain spelling differ", () => {
  assert.notDeepEqual(tokenize("matësi", { fold: false }), tokenize("matesi", { fold: false }));
});

test("stop words in English, Italian and Albanian are dropped", () => {
  assert.deepEqual(tokenize("the fee of the bill"), ["fee", "bill"]);
  assert.deepEqual(tokenize("sa është një faturë"), ["fatur"]);
});

test("one transposed or missing letter is still matched to the vocabulary", () => {
  const vocab = new Map([["reconnection", 3], ["interruption", 5]]);
  assert.equal(editDistance("acknowledge", "acknowlege", 2), 1);
  assert.equal(fuzzyMatch("reconection", vocab), "reconnection");
  assert.equal(fuzzyMatch("interuption", vocab), "interruption");
});

test("a word far from every term is left unknown instead of being guessed", () => {
  const vocab = new Map([["reconnection", 3]]);
  const a = analyzeQuery("dividend shareholders", { vocab });
  assert.equal(a.terms.length, 0);
  assert.equal(a.unknown.length, 2);
});

test("synonym expansion adds lower-weight terms only when the index knows them", () => {
  const vocab = new Map([["interruption", 2], ["bill", 4]]);
  const synonyms = compileSynonyms([["outage", "interruption", "blackout"]]);
  const a = analyzeQuery("outage bill", { vocab, expand: true, synonyms });
  const syn = a.terms.find((t) => t.term === "interruption");
  assert.equal(syn.source, "synonym");
  assert.equal(syn.weight, 0.5);
  assert.equal(a.terms.some((t) => t.term === "blackout"), false);
});
