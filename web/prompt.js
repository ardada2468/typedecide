/**
 * The canonical prompt. Both the browser engine and the trainer build their text
 * from here, because a fine-tune that learns a slightly different prompt than the
 * one being served degrades silently and looks like bad data.
 *
 * train/prompt.py mirrors this file exactly; train/check_parity.py proves it.
 */

export const LETTERS = 'ABCDEFGHIJKLMNOPQRST';

export const INSTRUCTION =
  'Apply the criterion to the evidence and choose exactly one listed option. ' +
  'Answer with only the uppercase letter of that option.';

/** Everything up to and including the state. Identical across every criterion. */
export const headText = (state) => `${INSTRUCTION}\n\nEvidence:\n${state}\n`;

/** One criterion's own question and options. Never shared. */
export const tailText = (criterion) => [
  `\nCriterion: ${criterion.question}`,
  'Options:',
  ...criterion.options.map((o, i) => `${LETTERS[i]}. ${o.description}`),
  'Answer:',
].join('\n');

export const fullPrompt = (state, criterion) => headText(state) + tailText(criterion);
