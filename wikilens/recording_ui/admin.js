'use strict';
if (document.querySelector('#admin-dialog')) {
  const adminDefaults = {score_cutoff:-4, relative_gap:null, retrieval_top_k:60, max_nodes:25, article_tokens:6000};
  state.settings = {...adminDefaults};
  const dialog = $('#admin-dialog');
  let snapshot = null;
  let requestID = null;
  let loadVersion = 0;
  const number = value => value === null || value === undefined ? '—' : Number(value).toFixed(3);
  function fillSettings(settings) {
    $('#admin-cutoff').value = settings.score_cutoff ?? -4;
    $('#admin-no-cutoff').checked = settings.score_cutoff === null;
    $('#admin-cutoff').disabled = settings.score_cutoff === null;
    $('#admin-gap').value = settings.relative_gap ?? '';
    $('#admin-top-k').value = settings.retrieval_top_k;
    $('#admin-max-nodes').value = settings.max_nodes;
    $('#admin-tokens').value = settings.article_tokens;
  }
  $('#admin-no-cutoff').addEventListener('change', () => {$('#admin-cutoff').disabled = $('#admin-no-cutoff').checked;});
  $('#admin-settings').addEventListener('submit', event => {
    event.preventDefault();
    if (!$('#admin-settings').reportValidity()) return;
    state.settings = {score_cutoff:$('#admin-no-cutoff').checked ? null : Number($('#admin-cutoff').value),
      relative_gap:$('#admin-gap').value === '' ? null : Number($('#admin-gap').value),
      retrieval_top_k:Number($('#admin-top-k').value), max_nodes:Number($('#admin-max-nodes').value), article_tokens:Number($('#admin-tokens').value)};
    $('#admin-settings-status').textContent = 'Applied to the next search. Existing results keep their recorded settings.';
  });
  $('#admin-defaults').addEventListener('click', () => {
    state.settings = {...adminDefaults}; fillSettings(state.settings);
    $('#admin-settings-status').textContent = 'Defaults restored for the next search.';
  });
  async function openDocument(article, host, capturedID) {
    host.replaceChildren(el('p', '', 'Loading retrieved text and passage scores…'));
    try {
      const doc = await getJSON('/api/questions/'+capturedID+'/documents/'+article.id);
      if (doc.error) throw new Error(doc.error);
      const source = el('a', '', 'Open original Wikipedia revision ↗'); source.href = doc.revision_url; source.target='_blank'; source.rel='noopener noreferrer';
      host.replaceChildren(source);
      host.append(el('h4', '', 'Reranked passages'));
      if (!doc.passages.length) host.append(el('p', '', 'No passage scores available yet.'));
      for (const passage of doc.passages) {
        const box = el('details', 'admin-passage'); box.dataset.selected = String(passage.selected);
        box.append(el('summary', '', `Score ${number(passage.rerank_score)} · ${passage.selected ? 'Used for generation '+passage.source_id : passage.passed_cutoff ? 'Passed cutoff, not used' : 'Rejected by cutoff'} · characters ${passage.start_char}–${passage.end_char}`));
        box.append(el('pre', '', passage.text)); host.append(box);
      }
      const full = el('details'); full.append(el('summary', '', 'Full retrieved article')); full.append(el('pre', '', doc.text)); host.append(full);
    } catch (error) {host.replaceChildren(el('p', '', error.message));}
  }
  function drawDocuments() {
    $('#admin-documents').replaceChildren();
    if (!snapshot) return;
    const selected = $('#admin-selected-only').checked;
    const capturedID = requestID;
    for (const article of snapshot.documents || []) {
      if (selected && !article.selected_count) continue;
      const row = el('details', 'admin-document');
      row.append(el('summary', '', `${article.retrieval_rank}. ${article.title} · ${article.selected_count} used`));
      row.append(el('p', 'admin-meta', `Vector distance ${number(article.distance)} (lower is closer). Best reranker score ${number(article.best_score)}. ${article.passing_count}/${article.passage_count} passages pass cutoff.`));
      const body=el('div'); row.append(body); let loaded=false;
      row.addEventListener('toggle', () => {if(row.open && !loaded){loaded=true;openDocument(article,body,capturedID);}});
      $('#admin-documents').append(row);
    }
  }
  async function loadDiagnostics() {
    const id = state.lastRequest || state.session?.last_request_id;
    if (!id) return;
    const version = ++loadVersion; const previousID = requestID; requestID = id;
    try {
      const data = await getJSON('/api/questions/'+id+'/diagnostics');
      if (version !== loadVersion) return;
      if (previousID === id && JSON.stringify(snapshot) === JSON.stringify(data)) return;
      snapshot = data;
      const settings = data.settings || {};
      const filter = data.score_filter;
      $('#admin-run-summary').textContent = `${data.question || 'Current question'}${data.context ? ' · '+data.context : ''} · Stage: ${data.stage}. Retrieved ${(data.documents || []).length} articles. Recorded cutoff: ${settings.score_cutoff === null ? 'disabled' : settings.score_cutoff ?? 'pending'}. Top-k: ${settings.retrieval_top_k ?? 'pending'}.${filter ? ' Passages kept: '+filter.counts.kept+'/'+filter.counts.input+'. Effective cutoff: '+number(filter.effective_min_score)+'.' : ''}${data.error ? ' '+data.error : ''}`;
      $('#admin-hyde').replaceChildren();
      if (data.hyde) {
        const aid=el('details');aid.append(el('summary', '', 'HyDE retrieval aid · detected language: '+(data.hyde.detected_language || 'unknown')));
        aid.append(el('p', '', 'Hypothetical text guides retrieval and is not source evidence.'));aid.append(el('pre', '', data.hyde.text || ''));$('#admin-hyde').append(aid);
      }
      drawDocuments();
    } catch (error) {if(version===loadVersion) $('#admin-run-summary').textContent=error.message;}
  }
  $('#admin-open').addEventListener('click', () => {dialog.showModal();loadDiagnostics();});
  $('#admin-close').addEventListener('click', () => dialog.close());
  $('#admin-refresh').addEventListener('click', loadDiagnostics);
  $('#admin-selected-only').addEventListener('change', drawDocuments);
  document.addEventListener('wikilens:request', () => {snapshot=null;if(dialog.open)loadDiagnostics();});
  document.addEventListener('wikilens:request-finished', () => {if(dialog.open)loadDiagnostics();});
  document.addEventListener('wikilens:session', () => {if(dialog.open)loadDiagnostics();});
  document.addEventListener('wikilens:ready', () => {
    if (state.live) return;
    document.querySelectorAll('#admin-settings input, #admin-settings button').forEach(control => {control.disabled = true;});
    $('#admin-settings-status').textContent = 'Settings require a live connection. Saved runs keep their original settings.';
  });
  fillSettings(state.settings);
}
