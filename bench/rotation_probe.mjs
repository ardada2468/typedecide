import { AutoTokenizer, AutoModelForCausalLM, Tensor } from '@huggingface/transformers';
const MODEL = process.env.MODEL ?? 'onnx-community/gemma-3-270m-it-ONNX';
const DTYPE = process.env.DTYPE ?? 'fp32';
const L = 'ABCDEFGHIJKLMNOP';
const INSTRUCTION = 'Apply the criterion to the evidence and choose exactly one listed option. Answer with only the uppercase letter of that option.';
const enc = (t, x) => Array.from(t.encode(x, { add_special_tokens: false }));
const i64 = a => new Tensor('int64', BigInt64Array.from(a.flat().map(BigInt)), [a.length, a[0].length]);
const softmax = xs => { const m = Math.max(...xs); const e = xs.map(x => Math.exp(x - m)); const s = e.reduce((a, b) => a + b, 0); return e.map(x => x / s); };

// 8 decisions with unambiguous ground truth, stated as option TEXT (order gets permuted below)
const ITEMS = [
  { s: 'Customer was charged twice for one order and wants one charge reversed.', q: 'Which queue should handle this?', o: ['Billing and payments', 'Shipping and logistics', 'Account access and authentication'], a: 'Billing and payments' },
  { s: 'Customer cannot log in; password reset emails never arrive.', q: 'Which queue should handle this?', o: ['Billing and payments', 'Shipping and logistics', 'Account access and authentication'], a: 'Account access and authentication' },
  { s: 'Parcel left the depot Tuesday; recipient says it never arrived.', q: 'Which queue should handle this?', o: ['Billing and payments', 'Shipping and logistics', 'Account access and authentication'], a: 'Shipping and logistics' },
  { s: 'Policy: refunds above 5000 need manager approval. Request: refund 900 for a damaged item.', q: 'Does this need manager approval?', o: ['Approval is required', 'Approval is not required'], a: 'Approval is not required' },
  { s: 'Policy: refunds above 5000 need manager approval. Request: refund 7400 for a damaged item.', q: 'Does this need manager approval?', o: ['Approval is required', 'Approval is not required'], a: 'Approval is required' },
  { s: 'The user writes: "This is the third time. Absolutely unacceptable. I want a manager."', q: 'Is the customer angry?', o: ['Yes, the customer is angry', 'No, the customer is calm'], a: 'Yes, the customer is angry' },
  { s: 'The user writes: "No rush at all, whenever you get a chance. Thanks for the help!"', q: 'Is the customer angry?', o: ['Yes, the customer is angry', 'No, the customer is calm'], a: 'No, the customer is calm' },
  { s: 'Server logs show disk usage at 41% and no errors in the last 24 hours.', q: 'Is there an active incident?', o: ['Yes, an incident is active', 'No, systems are healthy'], a: 'No, systems are healthy' },
];

function blockText(s, q, opts) {
  return `Evidence:\n${s}\n` + ['\nCriterion: ' + q, 'Options:', ...opts.map((d, i) => `${L[i]}. ${d}`), 'Answer:'].join('\n');
}
function buildPrompt(tok, style, s, q, opts) {
  const body = blockText(s, q, opts);
  if (style === 'completion') return INSTRUCTION + '\n\n' + body;
  return tok.apply_chat_template([{ role: 'user', content: INSTRUCTION + '\n\n' + body }], { tokenize: false, add_generation_prompt: true });
}
function slotsFor(tok, prompt, n) {
  const base = enc(tok, prompt);
  for (const sep of [' ', '']) {
    const out = []; let ok = true;
    for (const c of L.slice(0, n)) { const e = enc(tok, prompt + sep + c); if (e.length !== base.length + 1 || e.slice(0, base.length).join() !== base.join()) { ok = false; break; } out.push(e[e.length - 1]); }
    if (ok && new Set(out).size === out.length) return out;
  }
  throw new Error('no stable continuation');
}
const rotate = (a, k) => a.slice(k).concat(a.slice(0, k));

const tok = await AutoTokenizer.from_pretrained(MODEL);
const model = await AutoModelForCausalLM.from_pretrained(MODEL, { dtype: DTYPE, device: 'cpu' });

for (const style of ['completion', 'chat']) {
  let correct = 0, total = 0, consistent = 0, groups = 0;
  const letterCount = {};
  for (const it of ITEMS) {
    const picks = [];
    for (let k = 0; k < it.o.length; k++) {           // every rotation of the option list
      const opts = rotate(it.o, k);
      const prompt = buildPrompt(tok, style, it.s, it.q, opts);
      const slots = slotsFor(tok, prompt, opts.length);
      const ids = enc(tok, prompt);
      const out = await model({ input_ids: i64([ids]), attention_mask: i64([ids.map(() => 1)]) });
      const V = out.logits.dims.at(-1), off = (ids.length - 1) * V;
      const p = softmax(slots.map(x => Number(out.logits.data[off + x])));
      const idx = p.indexOf(Math.max(...p));
      letterCount[L[idx]] = (letterCount[L[idx]] ?? 0) + 1;
      picks.push(opts[idx]);
      total++; if (opts[idx] === it.a) correct++;
    }
    groups++; if (new Set(picks).size === 1) consistent++;
  }
  const dist = Object.entries(letterCount).sort().map(([k, v]) => `${k}:${(100 * v / total).toFixed(0)}%`).join(' ');
  console.log(`${style.padEnd(11)} accuracy ${(100 * correct / total).toFixed(1).padStart(5)}%  (chance ~${(100 * ITEMS.reduce((a, i) => a + 1 / i.o.length, 0) / ITEMS.length).toFixed(0)}%)  order-consistent ${consistent}/${groups}  picked letters ${dist}`);
}
