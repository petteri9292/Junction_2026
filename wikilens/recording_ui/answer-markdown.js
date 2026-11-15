'use strict';

const answerMarkdownFactory = typeof module !== 'undefined' && module.exports
  ? require('./vendor/markdown-it.min.js') : window.markdownit;
const answerMarkdownParser = answerMarkdownFactory({html:false, linkify:false, breaks:true});
// Generated images should not initiate external requests. Keep their alt text.
answerMarkdownParser.renderer.rules.image = (tokens, index) =>
  answerMarkdownParser.utils.escapeHtml(tokens[index].content);

function renderAnswerMarkdown(text) {
  // Finnish dates at the start of a paragraph are prose, not numbered lists.
  const prepared = text.replace(/^(\s{0,3})([1-9]|[12]\d|3[01])\.(?=\s+(?:tammikuuta|helmikuuta|maaliskuuta|huhtikuuta|toukokuuta|kesäkuuta|heinäkuuta|elokuuta|syyskuuta|lokakuuta|marraskuuta|joulukuuta)\b)/gim, '$1$2\\.');
  return answerMarkdownParser.render(prepared);
}

function appendAnswerContent(host, text) {
  const template = document.createElement('template');
  // HTML comes only from our pinned renderer with raw HTML disabled.
  template.innerHTML = renderAnswerMarkdown(text);
  host.append(template.content);
}

function answerCitationTarget(host) {
  let target = host.lastElementChild;
  while (target && (target.matches('ul, ol, blockquote') ||
    (target.matches('li') && target.lastElementChild?.matches('p, ul, ol, blockquote')))) {
    target = target.lastElementChild;
  }
  return !target || target.matches('pre, table, hr') ? host : target;
}

if (typeof module !== 'undefined' && module.exports) module.exports = {renderAnswerMarkdown};
