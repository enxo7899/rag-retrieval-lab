# Design

## The problem

A team builds chat over its documents and picks a chunk size, an embedding model and a reranker by feel. The answer quality depends mostly on whether the right passage is retrieved, and that can be measured without any language model. This lab measures it: which design puts the passage that answers a question near the top, how sure we are of that number, and when the system should say "I do not know".

## Architecture

```
 data/docs/*.md ──> corpus.js ──> chunkers.js ──> chunks (doc id, section path, offsets)
 (43 documents)      sections,     fixed | sentence-window | heading-aware
                     blocks             │
                                        ├─> bm25.js    inverted index, k1 and b
                                        ├─> tfidf.js   cosine
                                        ├─> lsa.js     truncated SVD (power iteration) saved in data/lsa.json
                                        └─> fusion.js  reciprocal rank fusion of BM25 and dense
 question ──> query.js (fold, stop words, typo and suffix matching, synonyms) ──> retriever ──> top 20
                                                                                      │
                                                    rerank.js  proximity + section title + revision recency
                                                                                      │
 data/queries.jsonl ──> grid.js ──> metrics.js (recall, MRR, nDCG, abstain) + stats.js (seeded bootstrap)
 (211, tuning | test)        │
                             └─> scripts/evaluate.mjs ──> docs/results.json ──> index.html and README
```

Everything in `src/` is plain ES modules with no file access, network or clock. Node reads the files and the browser fetches them, and both call the same functions, so the page is not a mock-up of the tests.

## Decisions and trade-offs

- **Gold labels are sections, not chunks.** A label is a passage id such as `DE-CUS-415#3` (document id and section number). Any chunker can be scored against the same labels: a chunk counts as the section that holds most of its characters, and results are de-duplicated by section so a chunker that returns overlapping windows gets no extra credit.
- **Dense retrieval is latent semantic analysis.** Truncated SVD of a TF-IDF matrix, computed by power iteration with deflation in `src/lsa.js` and saved to `data/lsa.json` by `npm run build`. It is trained on paragraphs only, never on questions, and shared by all chunkers. It captures co-occurrence, not meaning, so it is a weaker stand-in for a neural embedding model. The retriever interface (`score(terms)`) is the same, so a real embedding model drops in.
- **Abstain on confidence, not on raw score.** BM25 scores grow with query length, so BM25 confidence is the score divided by the best score a chunk could reach for that query. Cosine retrievers use the cosine. The threshold is chosen per configuration on the tuning set by balanced accuracy.
- **Selection cannot see the test set.** `selectConfiguration` receives the tuning queries and nothing else. The test set is scored for every configuration afterwards so the page can show what leakage does, but the chosen row comes from the tuning column only. A test rewrites every test question and shows the choice does not move.
- **Reranker weights are fixed by hand.** Three cheap features on the top 20: term proximity, section-title match and revision recency (1 for the newest revision, 0.5 less for each older one). They are not tuned, because tuning three weights on 127 questions would mostly fit noise.
- **Seeds everywhere.** The query split, LSA initialisation and bootstrap all use one seeded generator, so every number can be rebuilt.

## What I would do next for production

- Replace LSA with a neural embedding model behind the same interface and keep BM25 as the other half of the hybrid.
- Learn the reranker weights, or use a cross-encoder, with a separate validation split.
- Calibrate the abstain score (for example with a small classifier on the top-k features) instead of one global threshold.
- Handle tables as structured data and resolve revisions from metadata at query time rather than by a ranking nudge.
- Track the same metrics on real user questions, and re-run `npm run evaluate` in CI when documents change.
