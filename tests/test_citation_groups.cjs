const {test} = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {groupParagraphCitations} = require('../wikilens/recording_ui/citations.js');

const source = (id, passage, revision = 1) => ({id, passage, lang:'fi',
  title:'Article', revision_url:`https://fi.wikipedia.org/w/index.php?oldid=${revision}`, source_id:id});
const paragraph = (...citations) => ({text:'Generated paragraph.', citations});

test('one badge contains all excerpts from a source within its paragraph', () => {
  const raw = {answer:[paragraph('Q1','Q2','Q3')], evidence:[source('Q1','First quote.'),source('Q2','Second quote.'),source('Q3','Third quote.')]};
  const original = JSON.stringify(raw);
  const view = groupParagraphCitations(raw);
  assert.equal(view.answer[0].citations.length, 1);
  assert.deepEqual(view.evidence[0].quotes.map(q => q.passage), ['First quote.','Second quote.','Third quote.']);
  assert.equal(JSON.stringify(raw), original, 'Raw model evidence must remain unchanged');
});

test('the same source gets distinct numbers in different paragraphs', () => {
  const view = groupParagraphCitations({answer:[paragraph('Q1'),paragraph('Q2')], evidence:[source('Q1','Earlier paragraph quote.'),source('Q2','Later paragraph quote.')]});
  assert.deepEqual(view.evidence.map(s => s.number), [1,2]);
  assert.notEqual(view.answer[0].citations[0], view.answer[1].citations[0]);
  assert.deepEqual(view.evidence[1].quotes.map(q => q.passage), ['Later paragraph quote.']);
});

test('distinct sources or revisions keep distinct badges and first-use numbering', () => {
  const view = groupParagraphCitations({answer:[paragraph('Q2','Q1'),paragraph('Q1')], evidence:[source('Q1','One.',1),source('Q2','Two.',2)]});
  assert.equal(view.answer[0].citations.length, 2);
  assert.deepEqual(view.evidence.map(s => s.number), [1,2,3]);
});

test('duplicate quotations appear once while preserving original reference identities', () => {
  const view = groupParagraphCitations({answer:[paragraph('Q1','Q2','Q1')], evidence:[source('Q1','Same exact quote.'),source('Q2','Same exact quote.')]});
  assert.equal(view.evidence[0].quotes.length, 1);
  assert.deepEqual(view.evidence[0].quotes[0].reference_ids, ['Q1','Q2']);
});

test('missing and duplicate reference identifiers fail explicitly', () => {
  assert.throws(() => groupParagraphCitations({answer:[paragraph('missing')],evidence:[]}), /Unmapped/);
  assert.throws(() => groupParagraphCitations({answer:[],evidence:[source('Q1','A'),source('Q1','B')]}), /Duplicate/);
});

test('recorded demo keeps every exact quotation and groups repeated sources', () => {
  const fixtures=JSON.parse(fs.readFileSync(path.join(__dirname,'../wikilens/recording_ui/fixtures/examples.json'))).examples;
  for (const fixture of fixtures) {
    const view=groupParagraphCitations(fixture);
    fixture.answer.forEach((raw,index) => {
      const groups=view.evidence.filter(g => g.paragraph_number === index + 1);
      const original=raw.citations.map(id => fixture.evidence.find(e => e.id === id));
      assert.equal(view.answer[index].citations.length, new Set(original.map(e => e.revision_url)).size);
      assert.deepEqual(new Set(groups.flatMap(g => g.quotes.map(q => q.passage))), new Set(original.map(e => e.passage)));
    });
  }
});
