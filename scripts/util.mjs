/** Arithmetic mean, 0 for an empty list. */
export const mean = (xs) => xs.reduce((a, b) => a + b, 0) / (xs.length || 1);
