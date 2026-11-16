import { mulberry32 } from "./rng.js";

/**
 * Seeded percentile bootstrap of the mean.
 * @param {number[]} values
 * @param {{seed?: number, resamples?: number, alpha?: number}} [o]
 * @returns {{mean: number, lo: number, hi: number}}
 */
export function bootstrapCI(values, { seed = 20261116, resamples = 2000, alpha = 0.05 } = {}) {
  const n = values.length;
  const mean = values.reduce((a, b) => a + b, 0) / (n || 1);
  if (n === 0) return { mean: 0, lo: 0, hi: 0 };
  const rng = mulberry32(seed);
  const means = new Float64Array(resamples);
  for (let r = 0; r < resamples; r++) {
    let s = 0;
    for (let i = 0; i < n; i++) s += values[Math.floor(rng() * n)];
    means[r] = s / n;
  }
  means.sort();
  return { mean, lo: means[Math.floor((alpha / 2) * resamples)], hi: means[Math.min(resamples - 1, Math.floor((1 - alpha / 2) * resamples))] };
}

/** Median and 95th percentile of a list of numbers. */
export function percentiles(xs) {
  const s = [...xs].sort((a, b) => a - b);
  const at = (p) => (s.length ? s[Math.min(s.length - 1, Math.floor(p * s.length))] : 0);
  return { p50: at(0.5), p95: at(0.95) };
}
