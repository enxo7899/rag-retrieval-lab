// Reproducible README screenshots. Run `npm run dev` in another terminal, then `npm run screenshots`.
import { createRequire } from "node:module";
const { chromium, devices } = createRequire(import.meta.url)("playwright");
const base = process.env.BASE || "http://127.0.0.1:5188/";
const out = process.env.OUT || "docs";
const browser = await chromium.launch();
const errors = [];

/** Open a scenario and save either the viewport or one element (height capped). */
async function shot(name, url, { device, viewport = { width: 1360, height: 1500 }, panel, maxHeight = 1300, before } = {}) {
  const ctx = await browser.newContext(device ? { ...devices[device] } : { viewport, deviceScaleFactor: 2 });
  const page = await ctx.newPage();
  page.on("console", (m) => m.type() === "error" && !/fonts\.(googleapis|gstatic)/.test(m.location().url) && errors.push(`${name}: ${m.text()}`));
  page.on("pageerror", (e) => errors.push(`${name}: ${e.message}`));
  await page.goto(base + url, { waitUntil: "networkidle" });
  await page.waitForSelector("html[data-ready]");
  await page.waitForTimeout(300);
  if (before) await before(page);
  const h = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth);
  if (h > 0) errors.push(`${name}: horizontal scroll of ${h}px`);
  if (panel) {
    const box = await page.locator(panel).boundingBox();
    const top = box.y + (await page.evaluate(() => scrollY));
    await page.screenshot({ path: `${out}/${name}.png`, fullPage: true, clip: { x: 0, y: top, width: viewport.width, height: Math.min(box.height, maxHeight) } });
  } else await page.screenshot({ path: `${out}/${name}.png` });
  await ctx.close();
}

await shot("overview", "?run=outdated", { viewport: { width: 1360, height: 1500 } });
await shot("bench-abstain", "?run=unanswerable", { panel: "#bench", maxHeight: 1250 });
await shot("results-grid", "?run=leak", { panel: "#grid", maxHeight: 1350 });
await shot("failure-gallery", "", { panel: "#failures", maxHeight: 1150 });
await shot("mobile", "?run=unanswerable", { device: "iPhone 13", before: (p) => p.evaluate(() => document.getElementById("sides").scrollIntoView()) });
await browser.close();
if (errors.length) {
  console.error(errors.join("\n"));
  process.exit(1);
}
console.log("no console errors");
