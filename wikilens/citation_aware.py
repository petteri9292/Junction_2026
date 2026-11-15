"""Parse model-suggested citation IDs without treating them as verified support."""

import re

from .finnish_prompts import INSUFFICIENT_EVIDENCE_FI

MARKER = r"\[(?:[SP]\d+)(?:\s*,\s*[SP]\d+)*\]"


def parse_citation_aware(text):
    """Retain markers before/after punctuation; exclude the model bibliography."""
    heading = re.search(r"(?im)^\s*(?:#{1,6}\s*|\*\*)?Lähteet(?:\*\*)?:?\s*$", text)
    body = text[: heading.start()].strip() if heading else text.strip()
    bibliography = text[heading.end() :].strip() if heading else ""
    output = []
    section = "Yleiskatsaus"
    paragraph_id = 0
    for block in re.split(r"\n\s*\n", body):
        lines = block.strip().splitlines()
        if lines and (
            lines[0].startswith("#") or re.fullmatch(r"\*\*[^*]+\*\*", lines[0].strip())
        ):
            section = lines.pop(0).strip("#* ")
        paragraph = " ".join(line.strip() for line in lines).strip()
        if not paragraph:
            continue
        paragraph_id += 1
        boundaries = []
        last = 0
        # Include optional post-punctuation markers in the preceding sentence.
        for match in re.finditer(
            r'[.!?]["”]?(?:\s*' + MARKER + r')*\s+(?=[A-ZÅÄÖ“"])', paragraph
        ):
            if re.search(
                r"\b(?:[A-ZÅÄÖ]|Mr|Mrs|Dr|Gen|St|No|mm|esim|nyk)\.$",
                paragraph[: match.start() + 1],
            ):
                continue
            boundaries.append(paragraph[last : match.end()].strip())
            last = match.end()
        boundaries.append(paragraph[last:].strip())
        for raw in filter(None, boundaries):
            ids = list(
                dict.fromkeys(re.findall(r"[SP]\d+", " ".join(re.findall(MARKER, raw))))
            )
            clean = re.sub(r"\s*" + MARKER, "", raw).strip()
            output.append(
                {
                    "claim_id": f"C{len(output) + 1}",
                    "section": section,
                    "text": clean,
                    "raw_text": raw,
                    "paragraph_id": paragraph_id,
                    "requires_citation": clean != INSUFFICIENT_EVIDENCE_FI,
                    "citations": [
                        {
                            "source_id": source_id,
                            "quote": None,
                            "quotation_origin": "none_model_source_id_only",
                        }
                        for source_id in ids
                    ],
                }
            )
    return output, bibliography
