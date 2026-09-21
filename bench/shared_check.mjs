import { AutoTokenizer, AutoModelForCausalLM, Tensor } from '@huggingface/transformers';

const MODEL = process.env.MODEL ?? 'onnx-community/gemma-3-270m-it-ONNX';
const DTYPE = process.env.DTYPE ?? 'fp32';
const LETTERS = 'ABCDEFGHIJKLMNOP';
const INSTRUCTION = 'Apply the criterion to the evidence and choose exactly one listed option. Answer with only the uppercase letter of that option.';
const STATE = 'Ticket #4471. Customer reports they were charged twice for order A-9923 on 14 March. They have attached both card statements. They say they have already emailed twice with no reply and will dispute the charge with their bank tomorrow if nobody responds.';
const CRITERIA = [
  { q: 'Which queue should handle this ticket?', o: ['Account access and authentication', 'Billing and payments', 'Sales and product evaluation', 'Shipping and logistics'] },
  { q: 'How urgent is this ticket?', o: ['P0 - immediate', 'P1 - same day', 'P2 - this week', 'P3 - backlog'] },
  { q: 'Is the customer dissatisfied?', o: ['Yes', 'No'] },
  { q: 'Does this ticket carry a risk of chargeback or escalation?', o: ['Yes', 'No'] },
  { q: 'Assess the claim: the customer has received a reply.', o: ['Supported by the evidence', 'Not determined by the evidence', 'Contradicted by the evidence'] },
];
const head = s => `${INSTRUCTION}\n\nEvidence:\n${s}\n`;
const tail = c => ['\nCriterion: ' + c.q, 'Options:', ...c.o.map((d, i) => `${LETTERS[i]}. ${d}`), 'Answer:'].join('\n');
const softmax = xs => { const m = Math.max(...xs); const e = xs.map(x => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };
const enc = (t, x) => Array.from(t.encode(x, { add_special_tokens: false }));
const i64 = a => new Tensor('int64', BigInt64Array.from(a.flat().map(BigInt)), [a.length, a[0].length]);

function answerSlots(tok, prompt, count) {
  const base = enc(tok, prompt);
  for (const sep of [' ', '']) {
    const slots = []; let ok = true;
    for (const L of LETTERS.slice(0, count)) {
      const e = enc(tok, prompt + sep + L);
      if (e.length !== base.length + 1 || e.slice(0, base.length).join() !== base.join()) { ok = false; break; }
      slots.push(e[e.length - 1]);
    }
    if (ok && new Set(slots).size === slots.length) return { slots, sep };
  }
  throw new Error('no stable continuation');
}

const tok = await AutoTokenizer.from_pretrained(MODEL);
const model = await AutoModelForCausalLM.from_pretrained(MODEL, { dtype: DTYPE, device: 'cpu' });
console.log(`${MODEL} ${DTYPE} | ${model.config.model_type} | vocab ${model.config.vocab_size}`);

const prompts = CRITERIA.map(c => head(STATE) + tail(c));
const slotInfo = prompts.map((p, i) => answerSlots(tok, p, CRITERIA[i].o.length));
const readLast = (logits, slots, at) => {
  const V = logits.dims.at(-1); const off = at * V;
  return softmax(slots.map(s => Number(logits.data[off + s])));
};

// PATH A -- fresh full prompt per criterion
let t0 = performance.now();
const fresh = [];
for (let i = 0; i < prompts.length; i++) {
  const ids = enc(tok, prompts[i]);
  const out = await model({ input_ids: i64([ids]), attention_mask: i64([ids.map(() => 1)]) });
  fresh.push(readLast(out.logits, slotInfo[i].slots, ids.length - 1));
}
const tFresh = performance.now() - t0;
const freshTokens = prompts.reduce((a, p) => a + enc(tok, p).length, 0);

// PATH B -- prefill the state once, then reuse that KV for every criterion (batch 1 each)
t0 = performance.now();
const prefix = enc(tok, head(STATE)).slice(0, -1);
const suffixes = prompts.map(p => {
  const ids = enc(tok, p);
  if (ids.slice(0, prefix.length).join() !== prefix.join()) throw new Error('prefix not token-level prefix');
  return ids.slice(prefix.length);
});
const pre = await model({ input_ids: i64([prefix]), attention_mask: i64([prefix.map(() => 1)]) });
const prefixKV = {};
for (const [k, v] of Object.entries(pre)) if (k.startsWith('present.')) prefixKV[k.replace('present.', 'past_key_values.')] = v;
const tPrefill = performance.now() - t0;

const shared = []; const perBranch = [];
for (let i = 0; i < suffixes.length; i++) {
  const b = performance.now();
  const s = suffixes[i];
  const out = await model({
    input_ids: i64([s]),
    attention_mask: i64([prefix.map(() => 1).concat(s.map(() => 1))]),
    past_key_values: prefixKV,            // same prefix cache reused, never mutated
  });
  shared.push(readLast(out.logits, slotInfo[i].slots, s.length - 1));
  perBranch.push(performance.now() - b);
}
const tShared = performance.now() - t0;
const sharedTokens = prefix.length + suffixes.reduce((a, s) => a + s.length, 0);

console.log(`\nprefix ${prefix.length} tok  suffixes [${suffixes.map(s => s.length)}]`);
console.log(`fresh   ${tFresh.toFixed(0)} ms   ${freshTokens} prompt tokens`);
console.log(`shared  ${tShared.toFixed(0)} ms   ${sharedTokens} prompt tokens   (prefill ${tPrefill.toFixed(0)} ms, branches ${perBranch.map(x => x.toFixed(0)).join('/')} ms)`);
console.log(`speedup ${(tFresh / tShared).toFixed(2)}x   tokens ${(freshTokens / sharedTokens).toFixed(2)}x fewer   marginal cost/decision ${(perBranch.reduce((a, b) => a + b, 0) / perBranch.length).toFixed(0)} ms`);

let worst = 0, agree = 0;
const am = a => a.indexOf(Math.max(...a));
console.log('\n' + 'criterion'.padEnd(36) + 'answer'.padEnd(30) + 'p'.padEnd(8) + 'drift');
for (let i = 0; i < shared.length; i++) {
  const d = Math.max(...fresh[i].map((p, j) => Math.abs(p - shared[i][j])));
  worst = Math.max(worst, d); if (am(fresh[i]) === am(shared[i])) agree++;
  console.log(CRITERIA[i].q.slice(0, 34).padEnd(36) + CRITERIA[i].o[am(shared[i])].slice(0, 28).padEnd(30) + (Math.max(...shared[i]) * 100).toFixed(0).padStart(3) + '%    ' + d.toExponential(2));
}
console.log(`\nargmax agreement ${agree}/${shared.length}   max drift ${worst.toExponential(3)}`);
console.log(worst < 1e-4 ? '\nVERDICT: shared-prefix KV reuse is numerically exact. Mechanism confirmed.' : '\nVERDICT: drift present.');
