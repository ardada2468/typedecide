/* Page controller. The model work happens in worker.js; this file only measures,
   lays out the shared time axis, and reports what came back. */

import { LETTERS } from './prompt.js';
import * as Rubric from './rubric.js';

const $ = (id) => document.getElementById(id);

const state = {
  presets: null, model: null, worker: null, running: false,
  rubric: [],            // editable: [{ key, question, options: [{ id, description }] }]
  blocks: [],            // { lane, t0, t1, label, kind }
  scale: 400,            // milliseconds currently spanned by the axis
  genStartedAt: 0, genTimer: null,
};

/* ------------------------------------------------------------------- helpers */

const ms = (v) => (v >= 1000 ? `${(v / 1000).toFixed(v >= 10000 ? 0 : 2)} s` : `${Math.round(v)} ms`);

function niceStep(span) {
  const raw = span / 5;
  const mag = 10 ** Math.floor(Math.log10(raw));
  return [1, 2, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? mag * 10;
}

function setStatus(text, tone) {
  const el = $('status');
  el.textContent = text;
  if (tone) el.dataset.tone = tone; else delete el.dataset.tone;
}

/* ------------------------------------------------------- the shared time axis */

function drawTrace() {
  const span = Math.max(state.scale, 1);
  for (const lane of ['readout', 'generate']) {
    const track = $(`track-${lane}`);
    track.replaceChildren(...state.blocks.filter((b) => b.lane === lane).map((b) => {
      const el = document.createElement('div');
      el.className = `blk${b.kind ? ` ${b.kind}` : ''}`;
      el.style.left = `${(b.t0 / span) * 100}%`;
      el.style.width = `${Math.max(((b.t1 - b.t0) / span) * 100, 0.35)}%`;
      if (b.label) {
        const label = document.createElement('span');
        label.className = 'blk-label';
        label.textContent = b.label;
        el.append(label);
      }
      el.title = `${b.title ?? b.label ?? ''} — ${ms(b.t1 - b.t0)}`;
      return el;
    }));
  }

  const step = niceStep(span);
  const ticks = [];
  for (let t = 0; t <= span; t += step) {
    const el = document.createElement('span');
    el.className = 'tick';
    el.style.left = `${(t / span) * 100}%`;
    el.textContent = t === 0 ? '0' : ms(t);
    ticks.push(el);
  }
  $('axis').replaceChildren(...ticks);
}

function addBlock(block) {
  state.blocks.push(block);
  state.scale = Math.max(state.scale, block.t1 * 1.06);
  drawTrace();
}

/* -------------------------------------------------------------------- results */

function readoutRow(criterion, result) {
  const winner = result.probabilities.indexOf(Math.max(...result.probabilities));
  const row = document.createElement('div');
  row.className = 'rrow';

  const top = document.createElement('div');
  top.className = 'rrow-top';
  top.innerHTML = `<span class="rrow-key">${criterion.key}</span><span class="rrow-ms">${ms(result.ms)} · ${result.suffixTokens} tok</span>`;

  const answer = document.createElement('p');
  answer.className = 'rrow-answer';
  answer.textContent = criterion.options[winner].description;

  const bars = document.createElement('div');
  bars.className = 'bars';
  criterion.options.forEach((option, i) => {
    const p = result.probabilities[i];
    const bar = document.createElement('div');
    bar.className = `bar${i === winner ? ' win' : ''}`;
    bar.innerHTML =
      `<span class="bar-track"><span class="bar-fill" style="width:${(p * 100).toFixed(1)}%"></span>` +
      `<span class="bar-name">${option.id}</span></span>` +
      `<span class="bar-pct">${(p * 100).toFixed(1)}%</span>`;
    bars.append(bar);
  });

  row.append(top, answer, bars);
  return row;
}

function renderAudit(audit, generatedTokens) {
  const box = $('gen-audit');
  const verdict = document.createElement('p');
  verdict.className = 'audit-verdict';

  if (!audit.parsed) {
    verdict.dataset.ok = 'false';
    verdict.textContent = audit.reason;
    box.replaceChildren(verdict);
    return;
  }

  const n = audit.rows.length;
  const ok = audit.invalid === 0 && audit.missing === 0;
  verdict.dataset.ok = String(ok);

  const faults = [];
  if (audit.missing) faults.push(`${audit.missing} of ${n} criteria were never answered`);
  if (audit.invalid) faults.push(`${audit.invalid} value${audit.invalid > 1 ? 's' : ''} belong to a different criterion`);
  verdict.textContent = ok
    ? `Valid JSON, all ${n} keys present, every value legal for its criterion.`
    : `Valid JSON — and ${faults.join(', ')}. A schema check without a required-keys rule would wave this through.`;

  const rows = audit.rows.map((r) => {
    const el = document.createElement('div');
    el.className = 'audit-row';
    el.dataset.status = r.status;
    el.innerHTML = `<span class="k">${r.key}</span><span class="v">${r.value ?? '—'}</span><span class="s">${r.detail || 'ok'}</span>`;
    return el;
  });

  box.replaceChildren(verdict, ...rows);
}

function reading(label, value, note) {
  const d = document.createElement('div');
  d.innerHTML = `<dt>${label}</dt><dd>${value}${note ? `<small>${note}</small>` : ''}</dd>`;
  return d;
}


/* ------------------------------------------------------------- rubric editor */

function optionRow(option, index, criterionIndex, removable) {
  const row = document.createElement('div');
  row.className = 'option-row';
  row.innerHTML =
    `<span class="option-letter">${LETTERS[index]}</span>` +
    `<input class="option-input" type="text" value="" ` +
    `aria-label="Criterion ${criterionIndex + 1}, option ${LETTERS[index]}" ` +
    `placeholder="What this option means" data-field="option" data-c="${criterionIndex}" data-o="${index}">` +
    (removable
      ? `<button type="button" class="icon-btn" data-act="remove-option" data-c="${criterionIndex}" data-o="${index}" ` +
        `title="Remove this option" aria-label="Remove option ${LETTERS[index]}">&times;</button>`
      : '<span class="icon-slot"></span>');
  row.querySelector('.option-input').value = option.description ?? '';
  return row;
}

function criterionCard(criterion, index) {
  const li = document.createElement('li');
  li.className = 'criterion';

  const head = document.createElement('div');
  head.className = 'criterion-head';
  head.innerHTML =
    `<span class="criterion-number">${String(index + 1).padStart(2, '0')}</span>` +
    `<input class="criterion-question" type="text" value="" aria-label="Criterion ${index + 1} question" ` +
    `placeholder="What should the model decide about this state?" data-field="question" data-c="${index}">` +
    `<button type="button" class="icon-btn" data-act="remove-criterion" data-c="${index}" ` +
    `title="Remove this criterion" aria-label="Remove criterion ${index + 1}">&times;</button>`;
  head.querySelector('.criterion-question').value = criterion.question ?? '';

  // The key is what a caller sees in the returned JSON, so it is visible and yours
  // to set. Left blank it is derived from the question.
  const meta = document.createElement('div');
  meta.className = 'criterion-meta';
  meta.innerHTML =
    `<label class="key-label" for="key-${index}">json key</label>` +
    `<input id="key-${index}" class="key-input" type="text" value="" spellcheck="false" ` +
    `data-field="key" data-c="${index}" placeholder="${Rubric.deriveKey(criterion.question ?? '')}">`;
  meta.querySelector('.key-input').value = criterion.key ?? '';

  const options = document.createElement('div');
  options.className = 'options';
  const removable = criterion.options.length > Rubric.MIN_OPTIONS;
  criterion.options.forEach((option, i) => options.append(optionRow(option, i, index, removable)));

  if (criterion.options.length < Rubric.MAX_OPTIONS) {
    const add = document.createElement('button');
    add.type = 'button';
    add.className = 'add-option';
    add.dataset.act = 'add-option';
    add.dataset.c = String(index);
    add.textContent = '+ option';
    options.append(add);
  }

  li.append(head, meta, options);
  return li;
}

/** Repaint the editor. Called on structural changes only, never on every keystroke. */
function renderRubric() {
  $('rubric').replaceChildren(...state.rubric.map(criterionCard));
  refreshRubricStatus();
}

function refreshRubricStatus() {
  const ready = Rubric.forEngine(state.rubric);
  const issues = Rubric.problems(Rubric.normalise(state.rubric));
  const options = ready.reduce((a, c) => a + c.options.length, 0);

  $('rubric-tally').textContent = ready.length
    ? `${ready.length} ${ready.length === 1 ? 'criterion' : 'criteria'}, ${options} options, one shared state`
    : '';

  const box = $('rubric-problems');
  box.hidden = issues.length === 0;
  box.textContent = issues.map((p) => p.message).join(' ');
  $('run').disabled = state.running || !state.ready || issues.length > 0;
  Rubric.save(state.rubric);
}

function wireRubric() {
  const rubric = $('rubric');

  // Typing only updates the model and the tally; repainting would move the caret.
  rubric.addEventListener('input', (event) => {
    const field = event.target.dataset.field;
    if (!field) return;
    const c = Number(event.target.dataset.c);
    if (field === 'question') {
      state.rubric[c].question = event.target.value;
      const key = $('rubric').children[c]?.querySelector('.key-input');
      if (key) key.placeholder = Rubric.deriveKey(event.target.value);
    } else if (field === 'key') {
      state.rubric[c].key = event.target.value;
    } else {
      state.rubric[c].options[Number(event.target.dataset.o)].description = event.target.value;
    }
    refreshRubricStatus();
  });

  rubric.addEventListener('click', (event) => {
    const button = event.target.closest('[data-act]');
    if (!button) return;
    const c = Number(button.dataset.c);
    if (button.dataset.act === 'add-option') state.rubric[c].options.push({ id: '', description: '' });
    if (button.dataset.act === 'remove-option') state.rubric[c].options.splice(Number(button.dataset.o), 1);
    if (button.dataset.act === 'remove-criterion') state.rubric.splice(c, 1);
    renderRubric();

    if (button.dataset.act === 'add-option') {
      const inputs = $('rubric').children[c]?.querySelectorAll('.option-input');
      inputs?.[inputs.length - 1]?.focus();
    }
  });

  $('add-criterion').addEventListener('click', () => {
    state.rubric.push(Rubric.blankCriterion());
    renderRubric();
    const cards = $('rubric').children;
    cards[cards.length - 1]?.querySelector('.criterion-question')?.focus();
  });

  $('reset-rubric').addEventListener('click', () => {
    Rubric.clear();
    state.rubric = structuredClone(state.presets.rubric);
    renderRubric();
  });
}

/* ----------------------------------------------------------------- the worker */

function startWorker() {
  const worker = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => handle(data);
  worker.onerror = (e) => setStatus(`Worker failed: ${e.message}`, 'error');
  return worker;
}

function handle(m) {
  switch (m.type) {
    case 'progress': {
      if (m.total) {
        const pct = (m.loaded / m.total) * 100;
        $('loadbar').hidden = false;
        $('loadbar-fill').style.width = `${pct}%`;
        setStatus(`Downloading weights — ${(m.loaded / 1e6).toFixed(0)} of ${(m.total / 1e6).toFixed(0)} MB. Cached after the first visit.`);
      }
      break;
    }

    case 'ready': {
      $('loadbar').hidden = true;
      $('rig-device').textContent = m.device === 'webgpu' ? 'WebGPU' : 'WebAssembly';
      $('rig-dtype').textContent = m.dtype;
      state.ready = true;
      $('run').textContent = 'Run both lanes';
      refreshRubricStatus();
      setStatus(m.device === 'webgpu'
        ? 'Ready. Running on your GPU, in this tab.'
        : 'Ready. WebGPU was unavailable, so this is running on WebAssembly and will be slower.');
      break;
    }

    case 'prefill': {
      addBlock({ lane: 'readout', t0: 0, t1: m.ms, kind: 'prefill', label: 'prefill', title: `State prefill, ${m.tokens} tokens` });
      setStatus(`State prefilled once: ${m.tokens} tokens in ${ms(m.ms)}. Every criterion below reuses it.`);
      break;
    }

    case 'decision': {
      if (m.index === 0) state.firstDecisionAt = m.t1;
      addBlock({ lane: 'readout', t0: m.t0, t1: m.t1, title: m.criterion.key });
      $('readout-rows').append(readoutRow(m.criterion, m.result));
      $('total-readout').textContent = ms(m.t1);
      break;
    }

    case 'readout-done': {
      state.readout = m;
      $('total-readout').textContent = ms(m.ms);
      setStatus(`Readout finished: ${m.ms.toFixed(0)} ms for ${$('readout-rows').children.length} decisions. Now running the generation lane.`);
      break;
    }

    case 'cold-done': {
      state.coldMs = m.ms;
      break;
    }

    case 'gen-start': {
      state.genStartedAt = performance.now();
      $('gen-text').textContent = '';
      $('gen-text').dataset.done = 'false';
      state.genTimer = setInterval(() => {
        const now = performance.now() - state.genStartedAt;
        const block = state.blocks.find((b) => b.lane === 'generate');
        if (block) { block.t1 = now; state.scale = Math.max(state.scale, now * 1.06); drawTrace(); }
        $('total-generate').textContent = ms(now);
      }, 80);
      addBlock({ lane: 'generate', t0: 0, t1: 1, title: 'writing JSON' });
      break;
    }

    case 'gen-token': {
      $('gen-text').textContent += m.text;
      $('gen-text').scrollTop = $('gen-text').scrollHeight;
      break;
    }

    case 'gen-done': {
      clearInterval(state.genTimer);
      const block = state.blocks.find((b) => b.lane === 'generate');
      if (block) { block.t1 = m.ms; block.label = `${m.generatedTokens} tokens written`; }
      state.scale = Math.max(state.scale, m.ms * 1.06);
      drawTrace();
      $('total-generate').textContent = ms(m.ms);
      $('gen-text').dataset.done = 'true';
      renderAudit(m.audit, m.generatedTokens);
      finish(m);
      break;
    }

    case 'error': {
      clearInterval(state.genTimer);
      setStatus(m.message, 'error');
      state.running = false;
      $('run').textContent = 'Run both lanes';
      refreshRubricStatus();
      break;
    }
  }
}

function finish(gen) {
  const r = state.readout;
  const n = $('readout-rows').children.length;
  const perAnswer = gen.audit.answered ? gen.ms / gen.audit.answered : gen.ms;
  const readings = [
    reading('First decision', ms(state.firstDecisionAt), 'readout; generation has nothing yet'),
    reading('Per decision', ms(r.perDecision), 'readout, after the prefill'),
    reading('Readout total', ms(r.ms), `all ${n} answered`),
    reading('Generation total', ms(gen.ms), `${gen.generatedTokens} tokens, ${gen.audit.answered}/${n} answered`),
    reading('Cost per answer', ms(perAnswer), 'generation, over what it answered'),
    reading('Prompt tokens', r.promptTokens.toLocaleString(), `vs ${r.freshTokens.toLocaleString()} without state reuse`),
    reading('Mass on the letters', `${(r.allowedMass * 100).toFixed(1)}%`, 'of the whole vocabulary'),
    reading('Unusable answers', `${gen.audit.invalid + gen.audit.missing} / ${n}`,
      'generation lane; the readout cannot produce one'),
  ];
  if (state.coldMs) readings.splice(2, 0, reading('Without state reuse', ms(state.coldMs), `${(state.coldMs / r.ms).toFixed(1)}× the readout`));
  $('readings').replaceChildren(...readings);

  const bad = gen.audit.invalid + gen.audit.missing;
  setStatus(bad
    ? `Done. ${ms(r.perDecision)} per decision from the readout, all ${n} answered. The generation lane returned ${bad} answer${bad > 1 ? 's' : ''} a caller could not use.`
    : `Done. ${ms(r.perDecision)} per decision from the readout, all ${n} answered.`);
  state.running = false;
  $('run').textContent = 'Run again';
  refreshRubricStatus();
}

/* --------------------------------------------------------------------- wiring */

function run() {
  if (state.running) return;
  state.running = true;
  state.blocks = []; state.scale = 400; state.coldMs = 0; state.readout = null; state.firstDecisionAt = 0;
  $('run').disabled = true;
  $('run').textContent = 'Measuring…';
  $('trace-panel').hidden = false;
  $('results').hidden = false;
  $('readout-rows').replaceChildren();
  $('gen-text').textContent = '';
  $('gen-audit').replaceChildren();
  $('readings').replaceChildren();
  $('total-readout').textContent = '—';
  $('total-generate').textContent = '—';
  drawTrace();

  state.criteria = Rubric.forEngine(state.rubric);
  state.worker.postMessage({
    type: 'run',
    state: $('state').value,
    criteria: state.criteria,
    shuffle: $('shuffle').checked,
    nameKeys: $('namekeys').checked,
    measureCold: $('cold').checked,
  });
}

async function init() {
  const [presets, models] = await Promise.all([
    fetch('presets.json').then((r) => r.json()),
    fetch('models.json').then((r) => r.json()),
  ]);
  state.presets = presets;
  state.model = models.models[0];
  $('rig-model').textContent = state.model.name;

  state.rubric = Rubric.load() ?? structuredClone(presets.rubric);
  wireRubric();
  renderRubric();

  const buttons = presets.tickets.map((t, i) => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = t.label;
    b.setAttribute('aria-pressed', String(i === 0));
    b.addEventListener('click', () => {
      $('state').value = t.text;
      buttons.forEach((other) => other.setAttribute('aria-pressed', String(other === b)));
    });
    return b;
  });
  $('ticket-picker').replaceChildren(...buttons);
  $('state').value = presets.tickets[0].text;

  $('run').addEventListener('click', run);

  state.worker = startWorker();
  state.worker.postMessage({ type: 'load', repo: state.model.repo, dtypes: state.model.dtype });
}

init().catch((e) => setStatus(`Could not start: ${e.message}`, 'error'));
