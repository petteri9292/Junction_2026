'use strict';
const $ = (s) => document.querySelector(s);
const state = { examples: [], current: null, openCitations: new Set(), citationTrigger: null, generation: 0, live: false, session: null, busy: false };
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const el = (tag, className, text) => { const node = document.createElement(tag); if (className) node.className = className; if (text !== undefined) node.textContent = text; return node; };
const normalize = (value) => value.trim().toLocaleLowerCase().replace(/[?.!]+$/, '').replace(/\s+/g, ' ');
function announce(message) { $('#notice').textContent = message; $('#notice').hidden = !message; }
function setBusy(busy) { state.busy = busy; $('#ask').disabled = busy || (state.live && !['ready', 'answering'].includes(state.session?.status)); $('#question-form').setAttribute('aria-busy', String(busy)); }
function setContextExpanded(expanded) {
  $('#context-editor').hidden = !expanded;
  $('#context-toggle').setAttribute('aria-expanded', String(expanded));
  $('#context-symbol').textContent = expanded ? '−' : '+';
}
function resetOptions() {
  $('#deep-search').setAttribute('aria-pressed', 'false');
  $('#deep-search-preview').hidden = true;
  $('#research-context').value = '';
  setContextExpanded(false);
}
$('#deep-search').addEventListener('click', () => {
  const enabled = $('#deep-search').getAttribute('aria-pressed') !== 'true';
  $('#deep-search').setAttribute('aria-pressed', String(enabled));
  $('#deep-search-preview').hidden = !enabled;
});
$('#context-toggle').addEventListener('click', () => {
  const expanded = $('#context-editor').hidden;
  setContextExpanded(expanded);
  if (expanded) $('#research-context').focus();
});
$('#context-clear').addEventListener('click', () => {
  $('#research-context').value = ''; $('#research-context').focus();
});
$('#context-done').addEventListener('click', () => {
  setContextExpanded(false); $('#context-toggle').focus();
});
function syncCitations() {
  const hasOpen = state.openCitations.size > 0;
  $('#citation-panel').hidden = !hasOpen;
  $('.research-grid').classList.toggle('citations-open', hasOpen);
  document.querySelectorAll('.evidence-card').forEach(card => {
    const open = state.openCitations.has(card.dataset.evidence);
    card.hidden = !open; card.classList.toggle('selected', open);
    if (!open) card.querySelectorAll('details').forEach(detail => { detail.open = false; });
  });
  document.querySelectorAll('.citation').forEach(button => {
    const open = state.openCitations.has(button.dataset.evidence);
    button.classList.toggle('active', open); button.setAttribute('aria-pressed', String(open));
  });
}
function closeCitations(restoreFocus = false) {
  state.openCitations.clear();
  syncCitations();
  if (restoreFocus && state.citationTrigger?.isConnected) state.citationTrigger.focus({ preventScroll: true });
  state.citationTrigger = null;
}
function toggleCitation(id, trigger) {
  const card = document.getElementById(`evidence-${id}`);
  if (!card) return;
  const opening = !state.openCitations.has(id);
  if (opening) state.openCitations.add(id);
  else state.openCitations.delete(id);
  state.citationTrigger = trigger;
  syncCitations();
  if (opening) {
    card.focus({ preventScroll: true });
    card.scrollIntoView({ behavior: reducedMotion.matches ? 'instant' : 'smooth', block: 'nearest' });
  } else if (trigger?.isConnected) {
    trigger.focus({ preventScroll: true });
  }
}
$('#close-citations').addEventListener('click', () => closeCitations(true));
$('#citation-panel').addEventListener('keydown', event => {
  if (event.key === 'Escape') { event.preventDefault(); closeCitations(true); }
});
function render(fixture) {
  const display = groupParagraphCitations(fixture);
  document.body.classList.add('has-result');
  const theme = $('#theme');
  if (theme) theme.textContent = fixture.theme;
  $('#answer-title').textContent = fixture.title;
  const datasetLabel = /^l-/.test(fixture.dataset_release || '') ? 'Dataset L' : /^s-/.test(fixture.dataset_release || '') ? 'Dataset S' : 'Dataset not recorded';
  const resultLabel = fixture.status === 'live' ? 'Live result' : fixture.status === 'placeholder' ? 'Sample answer' : 'Saved run';
  const recordingNote = $('.recording-note');
  if (recordingNote) recordingNote.textContent = `${resultLabel} · ${datasetLabel}`;
  $('#answer').replaceChildren();
  let section = null;
  display.answer.forEach(claim => {
    if (claim.section && claim.section !== section) {
      section = claim.section; $('#answer').append(el('h3', '', section));
    }
    const lastSpace = claim.text.lastIndexOf(' ');
    const p = el('p', '', claim.text.slice(0, lastSpace + 1));
    const ending = el('span', 'claim-ending', claim.text.slice(lastSpace + 1));
    claim.citations.forEach(id => {
      const index = display.evidence.findIndex(e => e.id === id);
      if (index < 0) throw new Error('Unmapped fixture citation');
      const source = display.evidence[index];
      const button = el('button', 'citation', String(source.number));
      button.type = 'button'; button.dataset.evidence = id;
      button.setAttribute('aria-label', `Citation ${source.number}: ${source.title}, ${source.language_label}, paragraph ${source.paragraph_number}`);
      button.setAttribute('aria-controls', `evidence-${id}`); button.setAttribute('aria-pressed', 'false');
      button.addEventListener('click', () => toggleCitation(id, button)); ending.append(button);
    });
    p.append(ending);
    $('#answer').append(p);
  });
  $('#evidence').replaceChildren();
  display.evidence.forEach(source => {
    const card = el('article', 'evidence-card'); card.id = `evidence-${source.id}`; card.dataset.language = source.lang; card.dataset.evidence = source.id; card.tabIndex = -1;
    const top = el('div', 'source-top');
    const number = el('button', 'source-number', String(source.number)); number.type = 'button';
    number.setAttribute('aria-label', `Close citation ${source.number}: ${source.title}, paragraph ${source.paragraph_number}`);
    number.addEventListener('click', () => {
      const trigger = [...document.querySelectorAll('.citation')].find(button => button.dataset.evidence === source.id);
      toggleCitation(source.id, trigger);
    });
    const sourceLabel = document.body.dataset.primaryInterface ? 'Wikipedia' : `${source.language_label} Wikipedia`;
    top.append(number, el('span', 'source-lang', sourceLabel));
    const link = el('a', 'source-link', '↗'); link.href = source.revision_url; link.target = '_blank'; link.rel = 'noopener noreferrer'; link.title = 'Open stored Wikipedia revision (requires internet)'; link.setAttribute('aria-label', `Open ${source.title}, revision ${source.rev_id} (requires internet)`); top.append(link);
    const title = el('h3', 'source-title', source.title); title.id = `title-${source.id}`; card.setAttribute('aria-labelledby', title.id);
    card.append(top, title, el('p', 'source-context', `Paragraph ${source.paragraph_number} · ${source.quotes.length} ${source.quotes.length === 1 ? 'excerpt' : 'excerpts'}`));
    const quotations = el('ol', 'source-quotes');
    source.quotes.forEach(reference => {
      const item = el('li');
      const quote = el('blockquote', '', reference.passage); quote.lang = source.lang;
      item.append(quote); quotations.append(item);
    });
    card.append(quotations);
    if (source.reading_aid) {
      const details = el('details'); details.append(el('summary', '', '↳ English reading aid'));
      details.append(el('p', '', source.reading_aid + ' — Manual summary, not a verbatim translation.')); card.append(details);
    }
    $('#evidence').append(card);
  });
  closeCitations();
  $('#results').hidden = false;
  $('#results').classList.remove('appear'); void $('#results').offsetWidth; $('#results').classList.add('appear');
}
async function play(fixture) {
  if (state.live) return searchLive();
  if (normalize($('#research-context').value) !== normalize(fixture.search_focus || '')) {
    announce('Use the saved example’s context: ' + (fixture.search_focus || 'leave context empty') + '.');
    return;
  }
  const generation = ++state.generation;
  state.current = fixture; closeCitations();
  $('#question').value = fixture.question; announce(''); setBusy(true);
  $('#results').hidden = true; $('#welcome').hidden = true; $('#progress').hidden = false;
  const steps = ['Opening saved example…', 'Arranging the answer and citations…', 'Opening supporting passages…'];
  for (const step of steps) {
    if (generation !== state.generation) return;
    $('#progress-text').textContent = step;
    await new Promise(resolve => setTimeout(resolve, reducedMotion.matches ? 0 : 500));
  }
  if (generation !== state.generation) return;
  $('#progress').hidden = true; setBusy(false); render(fixture);
  const timing = $('#result-timing');
  if (timing) timing.textContent = 'Saved H100 run · ' + fixture.timings.total_seconds.toFixed(1) + ' s recorded · not a live search';
}
function reset(clearOptions = true) {
  if (clearOptions) resetOptions();
  document.body.classList.remove('has-result');
  const timing = $('#result-timing'); if (timing) timing.textContent = '';
  ++state.generation; state.current = null; state.citationTrigger = null;
  $('#question').value = ''; $('#results').hidden = true; $('#progress').hidden = true; $('#welcome').hidden = false;
  closeCitations(); announce(''); setBusy(false);
  window.scrollTo({ top: 0, behavior: reducedMotion.matches ? 'instant' : 'smooth' }); $('#question').focus({ preventScroll: true });
}
$('#question-form').addEventListener('submit', event => {
  event.preventDefault();
  if (state.live) { searchLive(); return; }
  const fixture = state.examples.find(f => normalize(f.question) === normalize($('#question').value) && normalize(f.search_focus || '') === normalize($('#research-context').value));
  if (fixture) play(fixture);
  else {
    const question = $('#question').value;
    reset(false);
    $('#question').value = question;
    announce('Saved runs: enter “talvisota” with empty context, or add context “videopeli”. Playback performs no inference.');
  }
});
$('#reset').addEventListener('click', () => reset());
$('#replay').addEventListener('click', () => { if (state.current) play(state.current); });
async function getJSON(url, options) {
  const response = await fetch(url, { cache: 'no-store', ...options });
  const value = await response.json();
  if (!response.ok) throw new Error(value.error || 'Request failed');
  return value;
}
async function refreshSession() {
  const session = await getJSON('/api/status');
  state.session = session;
  document.dispatchEvent(new CustomEvent('wikilens:session', {detail:session}));
  setBusy(state.busy);
  const status = $('#connection-status');
  if (status) {
    const ready = ['ready', 'answering'].includes(session.status);
    status.textContent = ready ? 'Connected · Model stays warm' : session.stage || 'Connecting to Modal…';
    status.dataset.state = ready ? 'ready' : session.status;
    if (session.deadline_unix) status.title = 'Warm until ' + new Date(session.deadline_unix * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'});
  }
  return session;
}
async function searchLive() {
  const question = $('#question').value.trim();
  if (!question) { announce('Enter a question first.'); return; }
  const generation = ++state.generation;
  state.current = { question }; closeCitations(); announce(''); setBusy(true);
  $('#results').hidden = true; $('#welcome').hidden = true; $('#progress').hidden = false;
  const started = performance.now();
  const timer = setInterval(() => {
    const caption = $('.progress-caption');
    if (caption) caption.textContent = ((performance.now() - started) / 1000).toFixed(1) + ' s';
  }, 100);
  try {
    $('#progress-text').textContent = 'Sending question to the warm model…';
    const request_id = crypto.randomUUID().replaceAll('-', '');
    await getJSON('/api/questions', {method: 'POST', headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({request_id, question, context: $('#research-context').value.trim(), settings: state.settings})});
    state.lastRequest = request_id;
    document.dispatchEvent(new CustomEvent('wikilens:request', {detail:{request_id}}));
    while (generation === state.generation) {
      if (['closed', 'failed', 'unavailable'].includes(state.session?.status)) throw new Error(state.session.stage || 'Research session is unavailable');
      const result = await getJSON('/api/questions/' + request_id);
      if (result.status === 'failed') throw new Error(result.error || 'Research failed');
      if (result.status === 'missing') throw new Error('Question was not found in the active session');
      if (result.status === 'complete') {
        document.dispatchEvent(new CustomEvent('wikilens:request-finished'));
        render(result.result);
        const timing = $('#result-timing');
        if (timing) timing.textContent = result.seconds.toFixed(1) + ' s · LLM reviewed';
        return;
      }
      $('#progress-text').textContent = result.stage || 'Researching Wikipedia…';
      await new Promise(resolve => setTimeout(resolve, 600));
    }
  } catch (error) {
    document.dispatchEvent(new CustomEvent('wikilens:request-finished'));
    if (generation === state.generation) { announce(error.message); $('#welcome').hidden = false; }
  } finally {
    clearInterval(timer);
    if (generation === state.generation) { $('#progress').hidden = true; setBusy(false); }
  }
}
async function init() {
  try {
    const config = await getJSON('/api/status');
    if (!config.live) { const status = $('#connection-status'); if (status) status.textContent = 'Offline preview'; }
    if (config.live) {
      state.live = true;
      const deep = $('#deep-search'); deep.hidden = true;
      $('#deep-search-preview').hidden = true;
      $('#context-disclosure').textContent = 'Your question and context are sent to the research session.';
      $('#replay').textContent = 'Run again ↻';
      await refreshSession();
      setInterval(() => refreshSession().catch(() => {
        const status = $('#connection-status'); if (status) status.textContent = 'Connection unavailable';
      }), 10000);
      document.dispatchEvent(new CustomEvent('wikilens:ready', {detail: {examples: []}}));
      return;
    }
  } catch (error) {
    // Old static preview servers have no API. Keep their explicit offline mode.
    const status = $('#connection-status'); if (status) status.textContent = 'Offline preview';
  }

  try {
    const response = await fetch('fixtures/examples.json'); if (!response.ok) throw new Error('Fixture request failed');
    const data = await response.json(); state.examples = data.examples;
    if (data.schema_version !== 2 || state.examples.length !== 2 || state.examples.some(f => f.status !== 'recorded')) throw new Error('Unexpected saved-run format');
    if (data.source_languages?.join(',') !== 'fi' || state.examples.some(f => f.evidence.some(source => source.lang !== 'fi'))) throw new Error('This interface supports Finnish Wikipedia sources only');
    $('#deep-search').hidden = true;
    const status = $('#connection-status'); if (status) status.textContent = 'Saved runs · no inference';
    announce('Saved H100 runs: search “talvisota”, then add context “videopeli” to compare. Model text is unedited and may contain errors.');
    document.dispatchEvent(new CustomEvent('wikilens:ready', { detail: { examples: state.examples.map(({ id, question, search_focus }) => ({ id, question, search_focus })) } }));

  } catch (error) {
    $('#ask').disabled = true;
    announce('The saved examples could not be loaded. Start the local server with python -m wikilens.recording from the project folder, then reload.');
  }
}
init();
