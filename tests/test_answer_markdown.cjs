const {test} = require('node:test');
const assert = require('node:assert/strict');
const {renderAnswerMarkdown} = require('../wikilens/recording_ui/answer-markdown.js');

test('the reported Finnish list renders as list items with bold labels', () => {
  const raw='- **Talvisota (1939–1940)**: Ensimmäinen kohta.\n- **Jatkosota (1941–1944)**: Toinen kohta.\n- **Lapin sota (1944–1945)**: Kolmas kohta.';
  const html=renderAnswerMarkdown(raw);
  assert.match(html, /<ul>/);
  assert.equal((html.match(/<li>/g)||[]).length,3);
  assert.match(html, /<strong>Talvisota \(1939–1940\)<\/strong>/);
  assert.ok(!html.includes('**'));
});

test('inline emphasis, headings, numbered lists and paragraphs remain structured', () => {
  const html=renderAnswerMarkdown('### Heading\n\nA **bold** and *emphasised* sentence.\n\n1. First\n2. Second');
  assert.match(html, /<h3>Heading<\/h3>/);
  assert.match(html, /<strong>bold<\/strong>/);
  assert.match(html, /<em>emphasised<\/em>/);
  assert.match(html, /<ol>/);
});

test('untrusted HTML stays text and unsafe links do not become clickable', () => {
  const html=renderAnswerMarkdown('<script>alert(1)</script>\n\n<img src=x onerror=alert(1)>\n\n[bad](javascript:alert%281%29)');
  assert.ok(!html.includes('<script>'));
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('href="javascript:'));
  assert.match(html, /&lt;script&gt;/);
});

test('Markdown images do not load external resources', () => {
  const html=renderAnswerMarkdown('![Visible description](https://example.com/tracker.png)');
  assert.ok(!html.includes('<img'));
  assert.ok(!html.includes('src='));
  assert.match(html, /Visible description/);
});

test('a paragraph beginning with a Finnish date does not become a numbered list', () => {
  const html=renderAnswerMarkdown('30. marraskuuta 1939 alkoi **talvisota**.');
  assert.match(html, /<p>30\. marraskuuta 1939/);
  assert.ok(!html.includes('<ol'));
  assert.match(html, /<strong>talvisota<\/strong>/);
});
