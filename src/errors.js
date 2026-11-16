/** Raised when a document file or the corpus bundle is malformed. */
export class CorpusError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = "CorpusError";
  }
}

/** Raised when a configuration or label file is invalid. */
export class ConfigError extends Error {
  /** @param {string} message */
  constructor(message) {
    super(message);
    this.name = "ConfigError";
  }
}
