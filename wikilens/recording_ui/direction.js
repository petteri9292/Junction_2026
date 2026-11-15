'use strict';
// Design previews replay the same fixture through the regular search interaction.
const directionParams = new URLSearchParams(location.search);
if (directionParams.get('preview') === '1') document.body.classList.add('design-thumbnail');
document.addEventListener('wikilens:ready', event => {
  const example = event.detail.examples.find(item => item.id === directionParams.get('example'));
  if (!example) return;
  document.getElementById('question').value = example.question;
  document.getElementById('research-context').value = example.search_focus || '';
  document.getElementById('question-form').requestSubmit();
}, { once: true });
const directionQuestion = document.getElementById('question');
if (directionQuestion.tagName === 'TEXTAREA') {
  directionQuestion.addEventListener('keydown', event => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!document.getElementById('ask').disabled) document.getElementById('question-form').requestSubmit();
    }
  });
}
