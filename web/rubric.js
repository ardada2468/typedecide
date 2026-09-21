/**
 * The editable rubric.
 *
 * Criteria and their options are yours to write. The engine never assumes anything
 * about them beyond what it can check: two to twenty options, each with text, and
 * ids that are unique within their criterion. Everything is kept in localStorage so
 * a reload does not throw away what you typed.
 */

import { LETTERS } from './prompt.js';

export const MAX_OPTIONS = LETTERS.length;
export const MIN_OPTIONS = 2;
const STORAGE_KEY = 'jev-custom.rubric.v1';

// Words that carry no meaning in a key. Dropping them is what turns
// "Is this ticket about a security incident?" into "ticket_security_incident"
// rather than a thirty-character slug nobody wants in their JSON.
const FILLER = new Set([
  'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'this', 'that',
  'these', 'those', 'it', 'its', 'of', 'for', 'to', 'in', 'on', 'at', 'by', 'with',
  'and', 'or', 'but', 'if', 'do', 'does', 'did', 'has', 'have', 'had', 'should',
  'would', 'could', 'can', 'will', 'shall', 'may', 'might', 'must', 'what', 'which',
  'who', 'whom', 'how', 'any', 'some', 'there', 'about', 'from', 'as', 'so',
]);

const words = (text) => (text ?? '').toLowerCase().normalize('NFKD')
  .replace(/[^\w\s-]/g, ' ').split(/[\s_-]+/).filter(Boolean);

function unique(base, taken) {
  if (!taken.has(base)) return base;
  for (let n = 2; ; n++) if (!taken.has(`${base}_${n}`)) return `${base}_${n}`;
}

/** A machine-usable id derived from what the user wrote, unique within its criterion. */
export function slugify(text, taken = new Set()) {
  const base = words(text).join('_').slice(0, 32).replace(/_+$/, '') || 'option';
  return unique(base, taken);
}

/** A short key for the JSON a caller will consume: meaningful words only. */
export function deriveKey(question, taken = new Set()) {
  const meaningful = words(question).filter((w) => !FILLER.has(w));
  const picked = (meaningful.length ? meaningful : words(question)).slice(0, 3);
  const base = picked.join('_').slice(0, 24).replace(/_+$/, '') || 'criterion';
  return unique(base, taken);
}

/** Fill in ids and keys, so the caller only ever has to manage the visible text. */
export function normalise(rubric) {
  const keys = new Set();
  return rubric.map((criterion, index) => {
    const key = criterion.key?.trim() || deriveKey(criterion.question || `criterion ${index + 1}`, keys);
    keys.add(key);
    const ids = new Set();
    const options = criterion.options.map((option) => {
      const id = option.id?.trim() || slugify(option.description, ids);
      ids.add(id);
      return { id, description: option.description ?? '' };
    });
    return { key, question: criterion.question ?? '', options };
  });
}

/**
 * Everything wrong with the rubric as written, in the order a person would fix it.
 * Returned rather than thrown: the editor shows them inline while you type.
 */
export function problems(rubric) {
  const found = [];
  if (!rubric.length) found.push({ criterion: null, message: 'Add at least one criterion.' });

  rubric.forEach((criterion, index) => {
    const label = `Criterion ${index + 1}`;
    if (!criterion.question?.trim()) found.push({ criterion: index, message: `${label} needs a question.` });

    const filled = criterion.options.filter((o) => o.description?.trim());
    if (filled.length < MIN_OPTIONS) {
      found.push({ criterion: index, message: `${label} needs at least ${MIN_OPTIONS} options with text.` });
    }
    if (criterion.options.length > MAX_OPTIONS) {
      found.push({ criterion: index, message: `${label} has more than ${MAX_OPTIONS} options, which is as many answer letters as there are.` });
    }
    const seen = new Set();
    for (const option of filled) {
      const text = option.description.trim().toLowerCase();
      if (seen.has(text)) {
        found.push({ criterion: index, message: `${label} lists "${option.description.trim()}" twice.` });
        break;
      }
      seen.add(text);
    }
  });

  const keys = rubric.map((c) => c.key);
  const duplicate = keys.find((k, i) => keys.indexOf(k) !== i);
  if (duplicate) found.push({ criterion: null, message: `Two criteria share the name "${duplicate}".` });

  return found;
}

/** Drop empty option rows, then normalise. What the engine actually receives. */
export function forEngine(rubric) {
  return normalise(rubric.map((criterion) => ({
    ...criterion,
    options: criterion.options.filter((o) => o.description?.trim()),
  })));
}

export const blankCriterion = () => ({
  key: '', question: '',
  options: [{ id: '', description: '' }, { id: '', description: '' }],
});

/* ------------------------------------------------------------------- storage */

export function save(rubric) {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(rubric)); }
  catch { /* private window, or storage full; the rubric still works in-page */ }
}

export function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) && parsed.length ? parsed : null;
  } catch { return null; }
}

export function clear() {
  try { localStorage.removeItem(STORAGE_KEY); } catch { /* nothing to clean up */ }
}
