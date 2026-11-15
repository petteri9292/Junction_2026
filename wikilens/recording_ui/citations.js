'use strict';

// Presentation only: retain the original evidence and group its exact excerpts.
function groupParagraphCitations(fixture) {
  const references = new Map();
  for (const source of fixture.evidence) {
    if (references.has(source.id)) throw new Error('Duplicate evidence identifier');
    references.set(source.id, source);
  }
  const numbers = new Map();
  const evidence = [];
  const answer = fixture.answer.map((paragraph, paragraphIndex) => {
    const groups = new Map();
    for (const id of paragraph.citations) {
      const reference = references.get(id);
      if (!reference) throw new Error('Unmapped paragraph citation');
      // Passage IDs can differ while referring to the same Wikipedia revision.
      const key = JSON.stringify([reference.lang, reference.revision_url]);
      if (!numbers.has(key)) numbers.set(key, numbers.size + 1);
      if (!groups.has(key)) {
        const number = numbers.get(key);
        const group = {...reference, id:`P${paragraphIndex + 1}-R${number}`,
          number, label:`${number}.${paragraphIndex + 1}p`, paragraph_number:paragraphIndex + 1, quotes:[]};
        groups.set(key, group);
        evidence.push(group);
      }
      const group = groups.get(key);
      let quote = group.quotes.find(item => item.passage === reference.passage);
      if (!quote) {
        quote = {passage:reference.passage, reference_ids:[], source_ids:[]};
        group.quotes.push(quote);
      }
      if (!quote.reference_ids.includes(id)) quote.reference_ids.push(id);
      if (reference.source_id && !quote.source_ids.includes(reference.source_id)) quote.source_ids.push(reference.source_id);
    }
    return {...paragraph, citations:[...groups.values()].map(group => group.id)};
  });
  return {answer, evidence};
}

if (typeof module !== 'undefined' && module.exports) module.exports = {groupParagraphCitations};
