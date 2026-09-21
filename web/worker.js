/**
 * Runs the model off the main thread so the page stays responsive while a lane is
 * measuring. Lanes run strictly one after another: measuring both at once on one
 * GPU would make both numbers meaningless.
 */

import { Engine, auditJson } from './engine.js';

const engine = new Engine();
const post = (message) => self.postMessage(message);

/** Permute a criterion's options, so a sceptic can check the answer tracks the
 *  content and not the letter position. */
function shuffleOptions(criterion) {
  const options = criterion.options.slice();
  for (let i = options.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [options[i], options[j]] = [options[j], options[i]];
  }
  return { ...criterion, options };
}

self.onmessage = async ({ data }) => {
  try {
    if (data.type === 'load') {
      const seen = new Map();
      const info = await engine.load(data.repo, data.dtypes, (p) => {
        if (p.status === 'progress' && p.file) {
          seen.set(p.file, { loaded: p.loaded ?? 0, total: p.total ?? 0 });
          let loaded = 0, total = 0;
          for (const v of seen.values()) { loaded += v.loaded; total += v.total; }
          post({ type: 'progress', loaded, total, file: p.file });
        } else if (p.status === 'ready' || p.status === 'done') {
          post({ type: 'progress', file: p.file, done: true });
        }
      });
      post({ type: 'ready', ...info });
      return;
    }

    if (data.type === 'run') {
      const criteria = data.shuffle ? data.criteria.map(shuffleOptions) : data.criteria;

      /* ---- lane 1: typed readout, one state prefill then one branch per criterion */
      const laneStart = performance.now();
      const pre = await engine.prefill(data.state);
      post({ type: 'prefill', ms: pre.ms, tokens: pre.tokens, at: pre.ms });

      const results = [];
      for (let i = 0; i < criteria.length; i++) {
        const t0 = performance.now() - laneStart;
        const result = await engine.score(criteria[i]);
        const t1 = performance.now() - laneStart;
        results.push(result);
        post({ type: 'decision', index: i, criterion: criteria[i], result, t0, t1 });
      }
      const readoutMs = performance.now() - laneStart;
      post({
        type: 'readout-done',
        ms: readoutMs,
        promptTokens: pre.tokens + results.reduce((a, r) => a + r.suffixTokens, 0),
        freshTokens: results.reduce((a, r) => a + r.fullPromptTokens, 0),
        perDecision: results.reduce((a, r) => a + r.ms, 0) / results.length,
        allowedMass: results.reduce((a, r) => a + r.allowedMass, 0) / results.length,
      });

      /* ---- lane 1b: optional, the same readout with no state reuse at all */
      if (data.measureCold) {
        const coldStart = performance.now();
        for (const c of criteria) await engine.scoreFresh(c, data.state);
        post({ type: 'cold-done', ms: performance.now() - coldStart });
      }

      /* ---- lane 2: the same model, asked to write the same answers as JSON */
      const genStart = performance.now();
      post({ type: 'gen-start' });
      const gen = await engine.generate(data.state, criteria, {
        nameKeys: data.nameKeys !== false,
        onToken: (t) => post({ type: 'gen-token', text: t, at: performance.now() - genStart }),
      });
      post({ type: 'gen-done', ...gen, audit: auditJson(gen.text, criteria) });
      return;
    }
  } catch (error) {
    post({ type: 'error', message: error?.message ?? String(error) });
  }
};
