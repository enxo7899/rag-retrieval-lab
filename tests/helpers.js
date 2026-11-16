import { loadAll } from "../scripts/load.mjs";
import { createLab } from "../src/pipeline.js";

export const data = loadAll();
export const lab = createLab(data.corpus, data.lsa);
export const baseConfig = { chunker: { type: "heading", max: 700 }, retriever: "bm25", rerank: false, expand: false };
