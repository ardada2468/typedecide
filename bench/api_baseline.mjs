/**
 * Run authored144 through a hosted frontier model, on the same fixture and the same
 * metric as authored144.mjs, so the rows can sit in one table.
 *
 * Keys come from the environment and are never written to disk or to results:
 *   ANTHROPIC_API_KEY / OPENAI_API_KEY / GEMINI_API_KEY
 *
 *   node api_baseline.mjs --provider anthropic --model <model-id>
 *   node api_baseline.mjs --provider openai --model <model-id> --mode json
 *
 * Responses are cached under .cache/ by (provider, model, mode, prompt), so an
 * interrupted run costs nothing to resume and a re-run costs nothing at all.
 *
 * Prices move, so nothing is hardcoded. Tokens are always reported; pass
 * --in-price and --out-price (dollars per million tokens, from the provider's own
 * page on the day you run it) to get a cost column.
 */

import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { LETTERS, headText, tailText } from '../web/prompt.js';
import { meanFamilyBalancedAccuracy } from './metric.mjs';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURE = path.join(HERE, '../reference-openjev/benchmarks/data/authored144.jsonl');
const CACHE = path.join(HERE, '.cache');

const argv = Object.fromEntries(
  process.argv.slice(2).join(' ').split('--').filter(Boolean)
    .map((s) => s.trim().split(/\s+/)).map(([k, ...v]) => [k, v.join(' ') || true]),
);
const PROVIDER = argv.provider ?? 'anthropic';
const MODEL = argv.model;
const MODE = argv.mode ?? 'letter';          // letter | json
const LIMIT = Number(argv.limit ?? 0);
const CONCURRENCY = Number(argv.concurrency ?? 4);
const MAX_TOKENS = Number(argv['max-tokens'] ?? 16);
const IN_PRICE = argv['in-price'] ? Number(argv['in-price']) : null;
const OUT_PRICE = argv['out-price'] ? Number(argv['out-price']) : null;

if (!MODEL) {
  console.error('Pass --model <id>. Model names change often, so this script does not guess one.');
  process.exit(2);
}

const KEY_VAR = { anthropic: 'ANTHROPIC_API_KEY', openai: 'OPENAI_API_KEY', google: 'GEMINI_API_KEY' }[PROVIDER];
const KEY = process.env[KEY_VAR];
if (!KEY) {
  console.error(`${KEY_VAR} is not set. Export it in your shell; this script never stores it.`);
  process.exit(2);
}

/* ------------------------------------------------------------------ the prompt */

const JSON_TAIL = 'Reply with a single JSON object of the form {"answer": "<option id>"} and nothing else.';

function buildPrompt(row) {
  const criterion = { question: row.question, options: row.options };
  const body = headText(row.state) + tailText(criterion);
  if (MODE === 'json') {
    const ids = row.options.map((o) => o.id).join(', ');
    return body.replace(/Answer:$/, `Option ids: ${ids}\n${JSON_TAIL}`);
  }
  return body;
}

/** Map whatever came back onto one option id, or null if it cannot be read. */
function parseAnswer(text, options) {
  if (!text) return null;
  if (MODE === 'json') {
    const match = text.match(/\{[\s\S]*?\}/);
    if (match) {
      try {
        const value = JSON.parse(match[0]).answer;
        if (options.some((o) => o.id === value)) return value;
      } catch { /* fall through to the letter reading below */ }
    }
  }
  const letter = text.trim().match(/\b([A-P])\b/) ?? text.trim().match(/[A-P]/);
  if (letter) {
    const index = LETTERS.indexOf(letter[1] ?? letter[0]);
    if (index >= 0 && index < options.length) return options[index].id;
  }
  const direct = options.find((o) => text.includes(o.id));
  return direct ? direct.id : null;
}

/* ---------------------------------------------------------------- the providers */

async function callAnthropic(prompt) {
  const response = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-api-key': KEY, 'anthropic-version': '2023-06-01' },
    body: JSON.stringify({ model: MODEL, max_tokens: MAX_TOKENS, messages: [{ role: 'user', content: prompt }] }),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  const data = await response.json();
  return {
    text: (data.content ?? []).filter((b) => b.type === 'text').map((b) => b.text).join(''),
    inTokens: data.usage?.input_tokens ?? 0,
    outTokens: data.usage?.output_tokens ?? 0,
  };
}

async function callOpenAI(prompt) {
  const response = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${KEY}` },
    body: JSON.stringify({ model: MODEL, messages: [{ role: 'user', content: prompt }], max_completion_tokens: MAX_TOKENS }),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  const data = await response.json();
  return {
    text: data.choices?.[0]?.message?.content ?? '',
    inTokens: data.usage?.prompt_tokens ?? 0,
    outTokens: data.usage?.completion_tokens ?? 0,
  };
}

async function callGoogle(prompt) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`;
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-goog-api-key': KEY },
    body: JSON.stringify({
      contents: [{ parts: [{ text: prompt }] }],
      generationConfig: { maxOutputTokens: MAX_TOKENS, temperature: 0 },
    }),
  });
  if (!response.ok) throw new Error(`${response.status} ${await response.text()}`);
  const data = await response.json();
  return {
    text: (data.candidates?.[0]?.content?.parts ?? []).map((p) => p.text ?? '').join(''),
    inTokens: data.usageMetadata?.promptTokenCount ?? 0,
    outTokens: data.usageMetadata?.candidatesTokenCount ?? 0,
  };
}

const CALL = { anthropic: callAnthropic, openai: callOpenAI, google: callGoogle }[PROVIDER];
if (!CALL) { console.error(`Unknown provider ${PROVIDER}. Use anthropic, openai or google.`); process.exit(2); }

/* ---------------------------------------------------------- cache, retry, pool */

const cacheKey = (prompt) =>
  crypto.createHash('sha256').update([PROVIDER, MODEL, MODE, MAX_TOKENS, prompt].join('␟')).digest('hex');

async function callCached(prompt) {
  fs.mkdirSync(CACHE, { recursive: true });
  const file = path.join(CACHE, `${cacheKey(prompt)}.json`);
  if (fs.existsSync(file)) return { ...JSON.parse(fs.readFileSync(file, 'utf8')), cached: true };

  let lastError;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      const started = Date.now();
      const result = await CALL(prompt);
      const payload = { ...result, ms: Date.now() - started };
      fs.writeFileSync(file, JSON.stringify(payload));
      return { ...payload, cached: false };
    } catch (error) {
      lastError = error;
      const wait = 1000 * 2 ** attempt;
      if (!/429|5\d\d|fetch failed|ETIMEDOUT/.test(String(error.message))) break;
      await new Promise((r) => setTimeout(r, wait));
    }
  }
  throw lastError;
}

async function pool(items, limit, worker) {
  const out = new Array(items.length);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, async () => {
    while (next < items.length) {
      const index = next++;
      out[index] = await worker(items[index], index);
    }
  }));
  return out;
}

/* --------------------------------------------------------------------- the run */

const main = async () => {
  let rows = fs.readFileSync(FIXTURE, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
  if (LIMIT) rows = rows.slice(0, LIMIT);

  console.log(`fixture   authored144, ${rows.length} rows, 3 options each`);
  console.log(`provider  ${PROVIDER}  ${MODEL}`);
  console.log(`mode      ${MODE}   max_tokens ${MAX_TOKENS}\n`);

  let done = 0, fresh = 0, inTokens = 0, outTokens = 0, latency = 0, failed = 0;

  const results = await pool(rows, CONCURRENCY, async (row) => {
    let answer = null, error = null, call = null;
    try {
      call = await callCached(buildPrompt(row));
      answer = parseAnswer(call.text, row.options);
    } catch (e) { error = e.message; failed++; }

    if (call) {
      inTokens += call.inTokens; outTokens += call.outTokens;
      if (!call.cached) { latency += call.ms; fresh++; }
    }
    process.stdout.write(`\r  ${++done}/${rows.length}`);
    return { family: row.family, gold: row.options[row.label].id, pred: answer, error };
  });
  console.log();

  if (failed) console.log(`\n  ${failed} request(s) failed; those rows count as wrong.`);

  const { per, mean } = meanFamilyBalancedAccuracy(results);
  const hits = results.filter((r) => r.pred === r.gold).length;
  const unparsed = results.filter((r) => r.pred === null).length;

  console.log();
  for (const [family, value] of Object.entries(per)) console.log(`  ${family.padEnd(26)} ${value.toFixed(3)}`);
  console.log(`\n  mean family balanced accuracy   ${mean.toFixed(3)}   (chance 0.333)`);
  console.log(`  plain accuracy                  ${(hits / results.length).toFixed(3)}`);
  console.log(`  unreadable answers              ${unparsed}/${results.length}`);
  console.log(`  tokens                          ${inTokens.toLocaleString()} in, ${outTokens.toLocaleString()} out`);
  if (fresh) console.log(`  mean latency                    ${Math.round(latency / fresh)} ms over ${fresh} uncached calls`);

  let cost = null;
  if (IN_PRICE !== null && OUT_PRICE !== null) {
    cost = (inTokens / 1e6) * IN_PRICE + (outTokens / 1e6) * OUT_PRICE;
    console.log(`  cost for these ${results.length} decisions   $${cost.toFixed(4)}   ($${(cost / results.length * 1000).toFixed(2)} per 1,000)`);
  } else {
    console.log('  cost                            pass --in-price and --out-price to compute');
  }

  const out = path.join(HERE, 'results');
  fs.mkdirSync(out, { recursive: true });
  const name = `authored144-${PROVIDER}-${MODEL.replace(/[^\w.-]/g, '_')}-${MODE}.json`;
  fs.writeFileSync(path.join(out, name), JSON.stringify({
    fixture: 'openjev/authored144', rows: results.length, provider: PROVIDER, model: MODEL,
    mode: MODE, per_family: per, mean_family_balanced_accuracy: mean,
    plain_accuracy: hits / results.length, unparsed, failed,
    in_tokens: inTokens, out_tokens: outTokens,
    mean_latency_ms: fresh ? Math.round(latency / fresh) : null,
    prices: IN_PRICE === null ? null : { in_per_mtok: IN_PRICE, out_per_mtok: OUT_PRICE },
    cost_usd: cost,
  }, null, 2));
  console.log(`\n  wrote results/${name}`);
};

main();
