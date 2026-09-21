/**
 * Merge everything in results/ into the one table the paper reports.
 *
 *   node summarize.mjs           # markdown
 *   node summarize.mjs --csv     # csv
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RESULTS = path.join(HERE, 'results');
const asCsv = process.argv.includes('--csv');

if (!fs.existsSync(RESULTS)) {
  console.error('No results/ yet. Run authored144.mjs or api_baseline.mjs first.');
  process.exit(1);
}

const MODE_LABEL = {
  'readout': 'typed readout',
  'readout-cyclic': 'readout + cyclic permutation averaging',
  'readout-full': 'readout + full permutation averaging',
  'readout-calibrated': 'readout + contextual calibration',
  'generate': 'generation (writes the answer)',
  'letter': 'generation (writes the letter)',
  'json': 'generation (writes JSON)',
};

const rows = fs.readdirSync(RESULTS).filter((f) => f.endsWith('.json')).map((f) => {
  const r = JSON.parse(fs.readFileSync(path.join(RESULTS, f), 'utf8'));
  const local = !r.provider;
  return {
    system: local ? `${r.model.split('/').pop()} (${r.dtype}, local)` : `${r.model} (${r.provider})`,
    local,
    method: MODE_LABEL[r.mode] ?? r.mode,
    shots: r.shots ?? null,
    balanced: r.mean_family_balanced_accuracy,
    plain: r.plain_accuracy,
    unreadable: r.unparsed ?? 0,
    unstable: r.order_unstable ?? null,
    costPer1k: r.cost_usd != null ? (r.cost_usd / r.rows) * 1000 : (local ? 0 : null),
    latency: r.mean_latency_ms ?? (r.seconds ? Math.round((r.seconds / r.rows) * 1000) : null),
    n: r.rows,
  };
}).sort((a, b) => b.balanced - a.balanced);

const pct = (v) => (v == null ? '—' : v.toFixed(3));
const money = (v) => (v == null ? '—' : v === 0 ? 'free' : `$${v.toFixed(2)}`);
const msFmt = (v) => (v == null ? '—' : `${v} ms`);

const header = ['System', 'Method', 'Shots', 'Balanced acc', 'Plain acc', 'Unreadable', 'Order-unstable', 'Cost / 1k', 'Latency / decision'];
const body = rows.map((r) => [
  r.system, r.method, r.shots ?? '—', pct(r.balanced), pct(r.plain),
  `${r.unreadable}/${r.n}`, r.unstable == null ? '—' : `${r.unstable}/${r.n}`,
  money(r.costPer1k), msFmt(r.latency),
]);

if (asCsv) {
  console.log([header, ...body].map((r) => r.map((c) => `"${c}"`).join(',')).join('\n'));
} else {
  const widths = header.map((_, i) => Math.max(...[header, ...body].map((r) => String(r[i]).length)));
  const line = (cells) => '| ' + cells.map((c, i) => String(c).padEnd(widths[i])).join(' | ') + ' |';
  console.log(line(header));
  console.log('|' + widths.map((w) => '-'.repeat(w + 2)).join('|') + '|');
  body.forEach((r) => console.log(line(r)));
  console.log(`\nFixture: openjev/authored144, ${rows[0]?.n ?? 0} rows, 3 options each. Chance = 0.333.`);
  console.log('Balanced acc is mean family balanced accuracy, per openjev\'s metric contract.');
  console.log('Local latency is wall time per decision on CPU in Node and is not comparable to');
  console.log('the browser figures, which run on WebGPU. API latency is one network round trip.');
}
