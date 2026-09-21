/**
 * Score openjev's authored144 fixture, which has published numbers for several
 * models, so our results sit on a scale someone else built.
 *
 * Metric is mean family balanced accuracy, as openjev's metric contract specifies:
 * per family, the recall of each gold class averaged, then averaged over the three
 * families. Chance is 0.333.
 *
 *   node authored144.mjs --mode readout
 *   node authored144.mjs --mode readout-cyclic --shots 3
 *   node authored144.mjs --mode generate
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { AutoTokenizer, AutoModelForCausalLM, Tensor } from '@huggingface/transformers';
import { LETTERS, headText, tailText } from '../web/prompt.js';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.join(HERE, '../reference-openjev/benchmarks/data/authored144.jsonl');

const argv = Object.fromEntries(
  process.argv.slice(2).join(' ').split('--').filter(Boolean)
    .map((s) => s.trim().split(/\s+/)).map(([k, ...v]) => [k, v.join(' ') || true]),
);
const MODE = argv.mode ?? 'readout';
const MODEL = argv.model ?? 'onnx-community/Qwen3-0.6B-ONNX';
const DTYPE = argv.dtype ?? 'q4f16';
const SHOTS = Number(argv.shots ?? 0);
const LIMIT = Number(argv.limit ?? 0);

/* Project-authored demonstrations. Deliberately not drawn from the fixture, and the
   correct letters are spread across positions so the format is taught without
   teaching a letter prior. */
const FEWSHOT = [
  { state: 'The parcel left the depot on Tuesday. The recipient reports it has not arrived.',
    question: 'Assess the claim: the parcel has been delivered.',
    options: ['The evidence establishes the claim', 'The evidence does not establish either', 'The evidence establishes the opposite'],
    answer: 2 },
  { state: 'A customer writes that their card was charged twice for one order and asks for one charge to be reversed.',
    question: 'Which queue should handle this request?',
    options: ['Account access and authentication', 'Billing and payment support', 'Sales and product evaluation'],
    answer: 1 },
  { state: 'Policy: refunds above 5000 need manager approval. Request: refund 900 for a damaged item.',
    question: 'Does this request need manager approval under the stated policy?',
    options: ['Manager approval is required', 'Manager approval is not required', 'The policy does not cover this'],
    answer: 1 },
];

const preamble = (shots) => shots === 0 ? '' :
  FEWSHOT.slice(0, shots).map((d) => {
    const criterion = { question: d.question, options: d.options.map((x) => ({ description: x })) };
    return headText(d.state) + tailText(criterion) + ' ' + LETTERS[d.answer] + '\n';
  }).join('\n') + '\n';

const enc = (tok, t) => Array.from(tok.encode(t, { add_special_tokens: false }));
const i64 = (rows) => new Tensor('int64', BigInt64Array.from(rows.flat().map(BigInt)), [rows.length, rows[0].length]);
const softmax = (xs) => { const m = Math.max(...xs); const e = xs.map((x) => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map((x) => x / s); };
const half = (u) => { const s = (u >> 15) & 1, e = (u >> 10) & 31, f = u & 1023;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 31) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024); };
const at = (d, i) => (d instanceof Uint16Array ? half(d[i]) : Number(d[i]));

function answerSlots(tok, prompt, count) {
  const base = enc(tok, prompt);
  for (const sep of [' ', '']) {
    const slots = []; let ok = true;
    for (const L of LETTERS.slice(0, count)) {
      const e = enc(tok, prompt + sep + L);
      if (e.length !== base.length + 1 || e.slice(0, base.length).join() !== base.join()) { ok = false; break; }
      slots.push(e[e.length - 1]);
    }
    if (ok && new Set(slots).size === slots.length) return slots;
  }
  throw new Error('no stable answer-letter continuation');
}

/** All permutations of [0..n-1]. */
function permutations(n) {
  if (n === 1) return [[0]];
  const out = [];
  for (const rest of permutations(n - 1)) {
    for (let i = 0; i <= rest.length; i++) out.push([...rest.slice(0, i), n - 1, ...rest.slice(i)]);
  }
  return out;
}
const cyclic = (n) => Array.from({ length: n }, (_, k) => Array.from({ length: n }, (_, i) => (i + k) % n));

let model, tok;

/** One scored prompt: probabilities over this ordering's answer letters. */
async function scoreOrder(state, question, options) {
  const criterion = { question, options };
  const prompt = preamble(SHOTS) + headText(state) + tailText(criterion);
  const ids = enc(tok, prompt);
  const slots = answerSlots(tok, prompt, options.length);
  const out = await model({ input_ids: i64([ids]), attention_mask: i64([ids.map(() => 1)]) });
  const V = out.logits.dims.at(-1), off = (ids.length - 1) * V;
  return softmax(slots.map((s) => at(out.logits.data, off + s)));
}

/** Free-text baseline: same question, but the model writes the letter. */
async function generateLetter(state, question, options) {
  const criterion = { question, options };
  const prompt = preamble(Math.max(SHOTS, 1)) + headText(state) + tailText(criterion);
  const inputs = tok(prompt, { add_special_tokens: false });
  const out = await model.generate({ ...inputs, max_new_tokens: 6, do_sample: false });
  const produced = Array.from(out.tolist()[0]).slice(inputs.input_ids.dims.at(-1));
  const text = tok.decode(produced, { skip_special_tokens: true });
  const match = text.match(/[A-P]/);
  return match ? LETTERS.indexOf(match[0]) : -1;
}

const NULL_STATE = 'N/A';
const SEP = '|#|';

async function decide(row) {
  const n = row.options.length;
  const base = row.options;

  if (MODE === 'generate') {
    const index = await generateLetter(row.state, row.question, base);
    return { id: index >= 0 && index < n ? base[index].id : null, parsed: index >= 0 && index < n, flips: 0 };
  }

  if (MODE === 'readout') {
    const p = await scoreOrder(row.state, row.question, base);
    return { id: base[p.indexOf(Math.max(...p))].id, parsed: true, flips: 0 };
  }

  const orders = MODE === 'readout-full' ? permutations(n) : cyclic(n);
  const totals = new Map(base.map((o) => [o.id, 0]));
  const picks = [];

  for (const order of orders) {
    const options = order.map((i) => base[i]);
    let p = await scoreOrder(row.state, row.question, options);

    if (MODE === 'readout-calibrated') {
      // Contextual calibration: divide out the prior this ordering assigns with
      // no evidence at all, then renormalise. Cached per (question, ordering).
      const key = row.question + SEP + options.map((o) => o.id).join('/');
      if (!decide.nullCache.has(key)) {
        decide.nullCache.set(key, await scoreOrder(NULL_STATE, row.question, options));
      }
      const prior = decide.nullCache.get(key);
      const adjusted = p.map((v, i) => v / Math.max(prior[i], 1e-9));
      const sum = adjusted.reduce((a, b) => a + b, 0);
      p = adjusted.map((v) => v / sum);
    }

    options.forEach((o, i) => totals.set(o.id, totals.get(o.id) + p[i] / orders.length));
    picks.push(options[p.indexOf(Math.max(...p))].id);
  }

  let best = null, bestP = -1;
  for (const [id, v] of totals) if (v > bestP) { bestP = v; best = id; }
  return { id: best, parsed: true, flips: new Set(picks).size > 1 ? 1 : 0 };
}
decide.nullCache = new Map();

/** Mean family balanced accuracy, per openjev's metric contract. */
function meanFamilyBalancedAccuracy(records) {
  const families = [...new Set(records.map((r) => r.family))].sort();
  const per = {};
  for (const family of families) {
    const rows = records.filter((r) => r.family === family);
    const classes = [...new Set(rows.map((r) => r.gold))].sort();
    const recalls = classes.map((c) => {
      const of = rows.filter((r) => r.gold === c);
      return of.filter((r) => r.pred === c).length / of.length;
    });
    per[family] = recalls.reduce((a, b) => a + b, 0) / recalls.length;
  }
  return { per, mean: Object.values(per).reduce((a, b) => a + b, 0) / families.length };
}

const main = async () => {
  let rows = fs.readFileSync(FIXTURE, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  if (LIMIT) rows = rows.slice(0, LIMIT);

  console.log(`fixture   authored144, ${rows.length} rows, 3 options each`);
  console.log(`model     ${MODEL} (${DTYPE})`);
  console.log(`mode      ${MODE}   shots ${SHOTS}\n`);

  tok = await AutoTokenizer.from_pretrained(MODEL);
  model = await AutoModelForCausalLM.from_pretrained(MODEL, { dtype: DTYPE, device: 'cpu' });

  const started = Date.now();
  const records = [];
  let flips = 0, unparsed = 0;

  for (let i = 0; i < rows.length; i++) {
    const row = rows[i];
    const { id, parsed, flips: f } = await decide(row);
    records.push({
      id: row.id, family: row.family, gold: row.options[row.label].id, pred: id,
      unstable: f === 1,
    });
    flips += f; if (!parsed) unparsed++;
    if ((i + 1) % 12 === 0) process.stdout.write(`\r  ${i + 1}/${rows.length}`);
  }
  console.log(`\r  ${rows.length}/${rows.length}   ${((Date.now() - started) / 1000).toFixed(1)}s\n`);

  const { per, mean } = meanFamilyBalancedAccuracy(records);
  const hits = records.filter((r) => r.pred === r.gold).length;

  for (const [family, value] of Object.entries(per)) console.log(`  ${family.padEnd(26)} ${value.toFixed(3)}`);
  console.log(`\n  mean family balanced accuracy   ${mean.toFixed(3)}   (chance 0.333)`);
  console.log(`  plain accuracy                  ${(hits / records.length).toFixed(3)}`);
  if (MODE.startsWith('readout-')) {
    console.log(`  order-unstable rows             ${flips}/${rows.length}  (argmax changed under reordering)`);
  }
  if (MODE === 'generate') console.log(`  unparseable outputs             ${unparsed}/${rows.length}`);

  const out = path.join(HERE, 'results');
  fs.mkdirSync(out, { recursive: true });
  const name = `authored144-${MODEL.split('/').pop()}-${DTYPE}-${MODE}-shots${SHOTS}.json`;
  fs.writeFileSync(path.join(out, name), JSON.stringify({
    fixture: 'openjev/authored144', rows: rows.length, model: MODEL, dtype: DTYPE,
    mode: MODE, shots: SHOTS, per_family: per, mean_family_balanced_accuracy: mean,
    plain_accuracy: hits / records.length, order_unstable: flips, unparsed,
    seconds: (Date.now() - started) / 1000,
  }, null, 2));
  const predictions = name.replace(/\.json$/, '.predictions.jsonl');
  fs.writeFileSync(path.join(out, predictions),
    records.map((r) => JSON.stringify(r)).join('\n') + '\n');
  console.log(`\n  wrote results/${name}`);
  console.log(`  wrote results/${predictions}`);
};

main();
