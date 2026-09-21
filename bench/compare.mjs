/**
 * Is the gap between two modes real, or is it 144 rows of noise?
 *
 * Compares two prediction files row by row with an exact McNemar test on the rows
 * where they disagree. Quoting one aggregate as better than another without this is
 * how a four-point difference becomes a claim it cannot support.
 *
 *   node compare.mjs results/<a>.predictions.jsonl results/<b>.predictions.jsonl
 */

import fs from 'node:fs';

const read = (p) => fs.readFileSync(p, 'utf8').trim().split('\n').map((l) => JSON.parse(l));
const [pathA, pathB] = process.argv.slice(2);
if (!pathA || !pathB) { console.error('Pass two .predictions.jsonl files.'); process.exit(2); }

const a = read(pathA), b = read(pathB);
const byId = new Map(b.map((r) => [r.id, r]));
if (a.length !== b.length) console.error(`warning: ${a.length} rows against ${b.length}`);

let onlyA = 0, onlyB = 0, both = 0, neither = 0;
for (const rowA of a) {
  const rowB = byId.get(rowA.id);
  if (!rowB) continue;
  const hitA = rowA.pred === rowA.gold, hitB = rowB.pred === rowB.gold;
  if (hitA && hitB) both++;
  else if (hitA) onlyA++;
  else if (hitB) onlyB++;
  else neither++;
}

// Exact binomial two-sided p on the discordant pairs (McNemar).
const n = onlyA + onlyB;
const choose = (n, k) => { let r = 1; for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1); return r; };
const smaller = Math.min(onlyA, onlyB);
let tail = 0;
for (let k = 0; k <= smaller; k++) tail += choose(n, k) * 0.5 ** n;
const p = Math.min(1, 2 * tail);

console.log(`A = ${pathA.split('/').pop()}`);
console.log(`B = ${pathB.split('/').pop()}\n`);
console.log(`  both right        ${both}`);
console.log(`  only A right      ${onlyA}`);
console.log(`  only B right      ${onlyB}`);
console.log(`  both wrong        ${neither}`);
console.log(`\n  discordant pairs  ${n}`);
console.log(`  exact McNemar p   ${p.toFixed(4)}`);
console.log(p < 0.05
  ? '\n  The difference is unlikely to be chance at this sample size.'
  : '\n  This sample cannot distinguish them. Report both, and do not rank them.');
