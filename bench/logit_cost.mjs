/**
 * How much of a branch forward pass is spent producing logits nobody reads?
 *
 * The readout looks at exactly one position, but the ONNX export returns logits for
 * every position in the suffix: `[1, suffix, 151936]`. Both the unembedding matmul
 * and the copy out of the accelerator scale with that. This measures the marginal
 * cost per suffix token by holding the prefilled state fixed and varying only the
 * suffix length, which isolates the per-token term from the fixed overhead.
 *
 *   node logit_cost.mjs
 */

import { AutoTokenizer, AutoModelForCausalLM, Tensor } from '@huggingface/transformers';

const MODEL = process.env.MODEL ?? 'onnx-community/Qwen3-0.6B-ONNX';
const DTYPE = process.env.DTYPE ?? 'q4f16';
const LENGTHS = [1, 8, 16, 32, 64, 96];
const REPEATS = Number(process.env.REPEATS ?? 5);

const i64 = (rows) => new Tensor('int64', BigInt64Array.from(rows.flat().map(BigInt)), [rows.length, rows[0].length]);
const ones = (n) => new Array(n).fill(1);
const median = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];

const tok = await AutoTokenizer.from_pretrained(MODEL);
const model = await AutoModelForCausalLM.from_pretrained(MODEL, { dtype: DTYPE, device: 'cpu' });
const vocab = model.config.vocab_size;
const hidden = model.config.hidden_size;

// A realistic shared state, prefilled once, exactly as the engine does it.
const head = 'Evidence:\nTicket #4471. The customer was charged twice for order A-9923 and has '
  + 'attached both statements. They have emailed twice with no reply.\n';
const prefix = Array.from(tok.encode(head, { add_special_tokens: false })).slice(0, -1);

const pre = await model({ input_ids: i64([prefix]), attention_mask: i64([ones(prefix.length)]) });
const kv = {};
for (const [k, v] of Object.entries(pre)) if (k.startsWith('present.')) kv[k.replace('present.', 'past_key_values.')] = v;

const filler = Array.from(tok.encode('\nCriterion: which queue should handle this ticket and how urgent is it really\nOptions:\nA. Billing and payments\nB. Account access\nC. Shipping\nAnswer:', { add_special_tokens: false }));
const suffixOf = (n) => Array.from({ length: n }, (_, i) => filler[i % filler.length]);

console.log(`${MODEL} (${DTYPE})  vocab ${vocab.toLocaleString()}  hidden ${hidden}`);
console.log(`prefix ${prefix.length} tokens, prefilled once\n`);
console.log('suffix   median ms   logits tensor   MB/s implied');

const points = [];
for (const n of LENGTHS) {
  const suffix = suffixOf(n);
  const input = { input_ids: i64([suffix]), attention_mask: i64([ones(prefix.length + n)]), past_key_values: kv };
  await model(input);                                   // warm
  const times = [];
  for (let r = 0; r < REPEATS; r++) {
    const t = performance.now();
    const out = await model(input);
    times.push(performance.now() - t);
    if (r === 0 && out.logits.dims.at(1) !== n) console.log(`  (note: logits dims ${out.logits.dims})`);
  }
  const ms = median(times);
  const bytes = n * vocab * 4;
  points.push({ n, ms });
  console.log(`${String(n).padStart(6)}   ${ms.toFixed(1).padStart(9)}   ${(bytes / 1e6).toFixed(2).padStart(9)} MB   ${(bytes / 1e6 / (ms / 1000)).toFixed(0).padStart(6)}`);
}

// Least-squares fit: time = fixed + perToken * n
const n = points.length;
const sx = points.reduce((a, p) => a + p.n, 0);
const sy = points.reduce((a, p) => a + p.ms, 0);
const sxy = points.reduce((a, p) => a + p.n * p.ms, 0);
const sxx = points.reduce((a, p) => a + p.n * p.n, 0);
const perToken = (n * sxy - sx * sy) / (n * sxx - sx * sx);
const fixed = (sy - perToken * sx) / n;

const REAL = 45;   // a typical criterion suffix in this project
const projected = fixed + perToken * REAL;
const trimmed = fixed + perToken * 1;

console.log(`\nfit: ${fixed.toFixed(1)} ms fixed + ${perToken.toFixed(2)} ms per suffix token`);
console.log(`\nA typical ${REAL}-token criterion costs ~${projected.toFixed(0)} ms.`);
console.log(`Only the final position is ever read. If the export returned logits for that`);
console.log(`position alone, the per-token term that survives is attention and MLP only;`);
console.log(`the unembedding and the readback would collapse to one position.`);
console.log(`\nUpper bound on the saving if the whole per-token term were logits-related:`);
console.log(`  ${projected.toFixed(0)} ms -> ${trimmed.toFixed(0)} ms  (${(projected / trimmed).toFixed(1)}x)`);
console.log(`That is an upper bound, not a measurement: the per-token term also contains`);
console.log(`real transformer work. The unembedding's share of it is`);
console.log(`  ${(hidden * vocab / 1e6).toFixed(0)}M MACs/token against ~${(0.44e9).toExponential(1)} for the rest of a 0.6B model,`);
console.log(`  so roughly ${(100 * (hidden * vocab) / (hidden * vocab + 0.44e9)).toFixed(0)}% of per-token FLOPs, plus a ${(vocab * 4 / 1e6).toFixed(2)} MB copy per token.`);
