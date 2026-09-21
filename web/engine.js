/**
 * Typed decisions read straight out of the logits.
 *
 * The state is prefilled into a KV cache exactly once. Each criterion is then a
 * short suffix evaluated against that same cache, and the answer is a softmax over
 * the token ids of its own answer letters. Nothing is sampled, so nothing can be
 * malformed.
 *
 * Verified in Node against onnx-community/Qwen3-0.6B-ONNX and gemma-3-270m-it:
 * shared-prefix scoring is bit-identical to scoring each full prompt fresh
 * (max probability drift 0.000e+0).
 */

import {
  AutoTokenizer, AutoModelForCausalLM, Tensor, TextStreamer, InterruptableStoppingCriteria,
} from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.0';

import { LETTERS, headText, tailText } from './prompt.js';

export { LETTERS, headText, tailText };

const JSON_INSTRUCTION =
  'Read the evidence and answer every criterion. Reply with a single JSON object that ' +
  'contains every criterion key listed, exactly once, and maps each one to a single ' +
  'option id belonging to that same criterion. Output only the JSON object.';

/* --------------------------------------------------------------- tensor plumbing */

const enc = (tok, text) => Array.from(tok.encode(text, { add_special_tokens: false }));

const i64 = (rows) =>
  new Tensor('int64', BigInt64Array.from(rows.flat().map(BigInt)), [rows.length, rows[0].length]);

const ones = (n) => new Array(n).fill(1);

/** Decode an IEEE half. Some builds hand back float16 logits; we normalise here. */
function half(u) {
  const s = (u >> 15) & 1, e = (u >> 10) & 31, f = u & 1023;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

const at = (data, i) => (data instanceof Uint16Array ? half(data[i]) : Number(data[i]));

export function softmax(xs) {
  const m = Math.max(...xs);
  const e = xs.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
}

/**
 * Read one position's logits, restricted to this criterion's answer letters.
 * Also reports how much of the model's *entire* vocabulary mass landed on those
 * letters, which is the evidence that the output format is actually learned.
 */
function readSlots(logits, slots) {
  // Always the final position of whatever came back. An export trimmed to emit only
  // the last position returns `[1, 1, V]`, so indexing by the suffix length would
  // run off the end and read adjacent memory rather than fail. Deriving the position
  // from the tensor is correct against both the full and the trimmed graph.
  const V = logits.dims.at(-1);
  const data = logits.data;
  const off = (logits.dims.at(1) - 1) * V;

  const picked = slots.map((s) => at(data, off + s));
  if (!picked.every(Number.isFinite)) {
    throw new Error('An answer letter scored non-finite; the vocabulary does not match this prompt.');
  }

  let max = -Infinity;
  for (let i = 0; i < V; i++) { const v = at(data, off + i); if (v > max) max = v; }
  let total = 0;
  for (let i = 0; i < V; i++) total += Math.exp(at(data, off + i) - max);
  const allowed = picked.reduce((a, v) => a + Math.exp(v - max), 0);

  return { logits: picked, probabilities: softmax(picked), allowedMass: allowed / total };
}

/**
 * Find the token ids for this criterion's answer letters, at this exact prompt
 * boundary. These are not contiguous in any vocabulary we tested, and whether the
 * natural continuation is " A" or "A" depends on the preceding character, so both
 * are probed and we keep whichever appends exactly one token without disturbing the
 * prompt's own tokenization.
 */
export function answerSlots(tok, prompt, count) {
  const base = enc(tok, prompt);
  if (!base.length) throw new Error('Prompt encoded to zero tokens');
  for (const separator of [' ', '']) {
    const slots = [];
    let ok = true;
    for (const letter of LETTERS.slice(0, count)) {
      const e = enc(tok, prompt + separator + letter);
      if (e.length !== base.length + 1 || e.slice(0, base.length).join() !== base.join()) { ok = false; break; }
      slots.push(e[e.length - 1]);
    }
    if (ok && new Set(slots).size === slots.length) return { slots, separator };
  }
  throw new Error('No answer-letter continuation keeps the prompt boundary stable for this tokenizer.');
}

/* -------------------------------------------------------------------- the engine */

export class Engine {
  constructor() { this.tok = null; this.model = null; this.device = null; this.prefix = null; }

  async load(repo, dtypes, onProgress) {
    const attempts = [];
    if (navigator.gpu) attempts.push(['webgpu', dtypes.webgpu]);
    attempts.push(['wasm', dtypes.wasm]);

    this.tok = await AutoTokenizer.from_pretrained(repo, { progress_callback: onProgress });

    let lastError;
    for (const [device, dtype] of attempts) {
      try {
        this.model = await AutoModelForCausalLM.from_pretrained(repo, {
          device, dtype, progress_callback: onProgress,
        });
        this.device = device; this.dtype = dtype;
        return { device, dtype };
      } catch (error) { lastError = error; }
    }
    throw new Error(`Could not start the model on WebGPU or WASM. ${lastError?.message ?? ''}`);
  }

  /** Prefill the state once. Everything after this is cheap. */
  async prefill(state) {
    const head = headText(state);
    // Drop the final token: the character that follows the state can merge into it,
    // which would stop the head being an exact token-level prefix of every prompt.
    const ids = enc(this.tok, head).slice(0, -1);
    if (ids.length < 2) throw new Error('State prefix is too short to reuse.');

    const started = performance.now();
    const out = await this.model({ input_ids: i64([ids]), attention_mask: i64([ones(ids.length)]) });

    const kv = {};
    for (const [k, v] of Object.entries(out)) {
      if (k.startsWith('present.')) kv[k.replace('present.', 'past_key_values.')] = v;
    }
    if (!Object.keys(kv).length) throw new Error('This export returned no reusable KV cache.');

    this.prefix = { head, ids, kv };
    return { ms: performance.now() - started, tokens: ids.length };
  }

  /** Score one criterion against the prefilled state. Two tensors in, one row out. */
  async score(criterion) {
    if (!this.prefix) throw new Error('Call prefill() before score().');
    const full = this.prefix.head + tailText(criterion);
    const ids = enc(this.tok, full);

    const p = this.prefix.ids;
    if (ids.length <= p.length || ids.slice(0, p.length).join() !== p.join()) {
      throw new Error('The state prefix is not a token-level prefix of this prompt.');
    }
    const suffix = ids.slice(p.length);
    const { slots, separator } = answerSlots(this.tok, full, criterion.options.length);

    const started = performance.now();
    const out = await this.model({
      input_ids: i64([suffix]),
      attention_mask: i64([ones(p.length + suffix.length)]),
      past_key_values: this.prefix.kv,          // reused, never mutated, never re-sent
    });
    const read = readSlots(out.logits, slots);
    const ms = performance.now() - started;

    return {
      key: criterion.key,
      optionIds: criterion.options.map((o) => o.id),
      ...read,
      answerTokenIds: slots,
      answerSeparator: separator,
      suffixTokens: suffix.length,
      fullPromptTokens: ids.length,
      ms,
    };
  }

  /** Score without the shared cache, to show what the prefill is actually saving. */
  async scoreFresh(criterion, state) {
    const full = headText(state) + tailText(criterion);
    const ids = enc(this.tok, full);
    const { slots } = answerSlots(this.tok, full, criterion.options.length);
    const started = performance.now();
    const out = await this.model({ input_ids: i64([ids]), attention_mask: i64([ones(ids.length)]) });
    const read = readSlots(out.logits, slots);
    return { key: criterion.key, ...read, fullPromptTokens: ids.length, ms: performance.now() - started };
  }

  /* ------------------------------------------------ the baseline: writing JSON */

  generationPrompt(state, criteria, nameKeys = true) {
    const rubric = (cs) => cs.map((c) =>
      `- ${c.key}: ${c.question}\n  options: ${c.options.map((o) => o.id).join(', ')}`).join('\n');

    // A worked example of the same shape, so the baseline knows the key convention
    // and that it is expected to answer every row. Without it a model invents its own
    // keys and the comparison stops being a fair one.
    const example =
      'Evidence:\nOrder C-2010 arrived cracked. The customer asks for a replacement and ' +
      'says the box was crushed in transit.\n\n' +
      'Criteria:\n' +
      '- queue: Which queue should handle this?\n  options: billing, shipping\n' +
      '- severity: How bad is the damage?\n  options: minor, major\n' +
      '- compensation: Money back or a fix?\n  options: money, fix\n' +
      '- deadline: Is a deadline stated?\n  options: yes, no\n\n' +
      'JSON:\n{"queue": "shipping", "severity": "major", "compensation": "fix", "deadline": "no"}\n';

    // Naming every required key is what a careful prompt engineer does, and without it
    // this model silently drops rows. The toggle exists so both are visible: a baseline
    // that was never told what was wanted is a strawman, and a sceptic will say so.
    const required = nameKeys
      ? `Required keys, all ${criteria.length} of them, in this order: ${criteria.map((c) => c.key).join(', ')}\n\n`
      : '';

    return `${JSON_INSTRUCTION}\n\n${example}\n` +
           `Evidence:\n${state}\n\nCriteria:\n${rubric(criteria)}\n\n${required}JSON:\n`;
  }

  async generate(state, criteria, { maxTokens = 400, onToken, nameKeys = true } = {}) {
    const prompt = this.generationPrompt(state, criteria, nameKeys);
    const inputs = this.tok(prompt, { add_special_tokens: false });
    const promptTokens = inputs.input_ids.dims.at(-1);

    // Stop the moment the object closes. Letting it run to the token cap would
    // inflate the generation lane's time for no reason a caller would accept.
    const stopper = new InterruptableStoppingCriteria();
    let seen = '';
    const streamer = new TextStreamer(this.tok, {
      skip_prompt: true, skip_special_tokens: true,
      callback_function: (t) => { seen += t; onToken?.(t); if (firstJsonObject(seen)) stopper.interrupt(); },
    });

    const started = performance.now();
    const out = await this.model.generate({
      ...inputs, max_new_tokens: maxTokens, do_sample: false, streamer, stopping_criteria: stopper,
    });
    const ms = performance.now() - started;

    const produced = Array.from(out.tolist ? out.tolist()[0] : out[0]).slice(promptTokens);
    const text = this.tok.decode(produced, { skip_special_tokens: true });
    return { text, ms, promptTokens, generatedTokens: produced.length };
  }
}

/** The first *balanced* JSON object in the text, or null if none has closed yet. */
export function firstJsonObject(text) {
  const start = text.indexOf('{');
  if (start === -1) return null;
  let depth = 0, inString = false, escaped = false;
  for (let i = start; i < text.length; i++) {
    const ch = text[i];
    if (inString) {
      if (escaped) escaped = false;
      else if (ch === '\\') escaped = true;
      else if (ch === '"') inString = false;
      continue;
    }
    if (ch === '"') inString = true;
    else if (ch === '{') depth++;
    else if (ch === '}' && --depth === 0) return text.slice(start, i + 1);
  }
  return null;
}

/**
 * Check what the generation lane actually produced. This is the comparison that
 * matters: the readout cannot return an option that does not belong to the
 * criterion, because its softmax only ranges over that criterion's own letters.
 */
export function auditJson(text, criteria) {
  const slice = firstJsonObject(text);
  if (!slice) {
    return { parsed: false, reason: 'The output never closed a JSON object.', rows: [], invalid: 0, missing: criteria.length, answered: 0 };
  }
  let obj;
  try { obj = JSON.parse(slice); }
  catch (error) { return { parsed: false, reason: `Invalid JSON: ${error.message}`, rows: [], invalid: 0, missing: criteria.length, answered: 0 }; }

  let invalid = 0, missing = 0;
  const rows = criteria.map((c) => {
    const value = obj[c.key];
    const legal = c.options.map((o) => o.id);
    let status, detail;
    if (value === undefined) { status = 'missing'; detail = 'never answered'; missing++; }
    else if (legal.includes(value)) { status = 'ok'; detail = ''; }
    else {
      const owner = criteria.find((o) => o.options.some((x) => x.id === value));
      status = 'invalid';
      detail = owner ? `an option of "${owner.key}"` : 'not an option anywhere';
      invalid++;
    }
    return { key: c.key, value: value ?? null, status, detail };
  });

  const extra = Object.keys(obj).filter((k) => !criteria.some((c) => c.key === k));
  return { parsed: true, rows, invalid, missing, answered: criteria.length - missing, extra, raw: obj };
}
