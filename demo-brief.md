---
title: WikiLens — From retrieval to an article
audience: Junction 2026 judges and first-time users
language: en
goal: Show the MariaDB retrieval-to-generation pipeline, using context disambiguation as an example
target_duration_seconds: 115
maximum_duration_seconds: 120
width: 1920
height: 1080
fps: 30
brand: WikiLens
captions: English, matching the narration
delivery_format: MP4, H.264 video with AAC audio
---

# Purpose

Show how the same question produces a different, relevant answer when the user
adds context. Demonstrate **talvisota** without context and then with
**videopeli** as context. Explain how a question becomes a search passage,
retrieved evidence, a generated article and inspectable citations.

This is a working application demonstration, not a claim that generated answers
are error-free.

# Audience and language

Assume the viewer understands ordinary web search but has no background in
retrieval-augmented generation. Narrate and caption in English. Keep the two
Finnish demo inputs visible and explain their meanings: talvisota means
“Winter War”; videopeli means “video game”. The corresponding live answers
can remain Finnish.

# Inputs

- Project folder: `/home/petterihallipelto/Documents/Codex/Junction_2026`
- Application: `http://127.0.0.1:8041/`
- Selected interface: `/home/petterihallipelto/Documents/Codex/Junction_2026/wikilens/recording_ui/paper.html`
- Interface styling: `/home/petterihallipelto/Documents/Codex/Junction_2026/wikilens/recording_ui/directions.css`
- Startup and session instructions: `/home/petterihallipelto/Documents/Codex/Junction_2026/docs/recording-demo.md`
- Submission context: `/home/petterihallipelto/Documents/Codex/Junction_2026/docs/submission.md`
- Experiment records: `/home/petterihallipelto/Documents/Codex/Junction_2026/docs/experiments.md`
- Live-demo measurements: `/home/petterihallipelto/Documents/Codex/Junction_2026/reports/metrics.json`
- Language-routing measurements: `/home/petterihallipelto/Documents/Codex/Junction_2026/reports/metrics.json`
- Paper interface reference screenshot: `/home/petterihallipelto/Documents/Codex/Junction_2026/reports/paper-language-english.png`
- Citation-panel reference screenshot: `/home/petterihallipelto/Documents/Codex/Junction_2026/reports/paper-admin-tools.png`

No recorded video footage is supplied. Capture fresh application footage.
The screenshots are visual references, not a substitute for the two live
searches. Check that the application reports a ready connection before recording;
the Modal session is time-bounded. Follow the session instructions above if it
has expired.

# Story

Begin with the reader problem: turn a large Wikipedia collection into a readable article with inspectable evidence. Introduce the retrieval–generation pipeline, then run the same
question twice and make the single changed context field easy to see. Show
the historical subject first and the game second. Open a citation so viewers
can distinguish generated prose from the original evidence.

Then explain the pipeline using a simple diagram. Return to the application
for the final evidence inspection and closing statement.

Use these diagram stages, in this order:

1. Question + optional context.
2. Qwen HyDE: hypothetical passage + detected question language.
3. BGE-M3: embed only the hypothetical passage.
4. MariaDB native vector search over Finnish Dataset L.
5. Heading-aware passage extraction + cross-encoder reranking.
6. Article generation in the detected answer language.
7. LLM review and revision against the source passages.
8. Citation attachment + Python quotation checks.

Animate a highlight moving through the stages while they are explained.
Keep labels short and readable; the narration supplies the detail.

# Demo actions

1. Open Paper at `http://127.0.0.1:8041/` at a readable desktop size.
2. Click **New question** to clear the previous question and context.
3. Enter **talvisota** in the question field. Leave context empty.
4. Click **Search**. Expected result: an article about the historical Winter War.
5. Open an inline citation. Expected result: the citation panel displays a
   literal source quotation and its Wikipedia revision link.
6. Close the citation panel. Keep **talvisota** unchanged in the question field.
7. Click **Add context**, enter **videopeli**, and click **Done** if needed.
8. Click **Search** again. Expected result: an article about the video game,
   with evidence from the corresponding game article.
9. Show the pipeline diagram and highlight the stages as narrated.
10. Return to the game answer and open a citation. Hold the generated text and
    supporting quotation together for the closing shot.

Rehearse both searches before the final recording. If a search fails or selects
the wrong topic, fix the issue or re-record; do not substitute a hand-written
answer and present it as a live result. Preserve generated wording in the footage.

# Timed script

## 0:00–0:12 — Introduce the question

**Show:** Empty Paper interface. Type **talvisota**, leaving context empty.

**Say:** “A search term doesn’t always tell us what someone means. WikiLens
combines a question with optional context to retrieve Wikipedia evidence and
generate a focused article.”

## 0:12–0:28 — Show the historical meaning

**Show:** Submit the question. Show the historical answer and open a citation.

**Say:** “With no added context, ‘talvisota’ leads to the Winter War between
Finland and the Soviet Union. The answer includes citations that open the
original source quotations.”

## 0:28–0:47 — Change only the context

**Show:** Keep the question unchanged. Add **videopeli**, submit, and show the
game answer. Make the changed context visible before the result appears.

**Say:** “Now I keep exactly the same question and add one word: ‘videopeli’—video
game. The answer now concerns the game. Context changes which evidence the
system selects, rather than simply changing the writing style.”

## 0:47–1:05 — Explain the search representation

**Show:** Highlight question and context, HyDE, then embedding in the diagram.

**Say:** “Here’s how it works. Qwen produces a hypothetical search passage and
detects the question’s language in one structured response. BGE-M3 embeds only
the passage. This hypothetical text guides retrieval; it is never evidence
for the article.”

## 1:05–1:23 — Follow retrieval into generation

**Show:** Highlight MariaDB, passage extraction, reranking and generation.

**Say:** “MariaDB’s native vector search retrieves candidates from 568,416 Finnish
Wikipedia articles. We split them into heading-aware passages and rerank them
for relevance. Qwen then writes an article in the detected language using the
selected source passages.”

## 1:23–1:43 — Explain review and citations

**Show:** Highlight review, citation attachment and Python checks. Return to the
game answer and open a citation.

**Say:** “A second LLM pass reviews and revises the draft before citations are
attached. The citation stage selects source quotations, and Python checks their
identifiers and exact text offsets. These checks establish quotation provenance;
they do not guarantee factual correctness.”

## 1:43–1:55 — Close on the product

**Show:** Hold the answer and its open source quotation together.

**Say:** “The model and database stay loaded on Modal between questions. WikiLens
brings contextual retrieval, generation and inspectable evidence together in
one reading interface.”

# Narration

Use a clear, conversational tone. Aim for roughly 130–140 words per minute,
with brief pauses when the context changes and the source quotation opens.
Explain what each step does rather than reading technical labels mechanically.

Pronunciation guidance:

- **WikiLens:** “WIK-ee lens”.
- **talvisota:** approximately “TAL-vee-soh-tah”; stress the first syllable.
- **videopeli:** approximately “VEE-deh-oh-peh-lee”; stress the first syllable.
- **Qwen:** “kwen”.
- **HyDE:** “hide”.
- **BGE-M3:** “B G E, M three”.
- **MariaDB:** “mah-REE-ah, D B”.
- **LLM:** say the letters, “L L M”.

In speech, the article count may be rounded to “over five hundred and sixty-eight
thousand”. Keep **568,416 Finnish articles** as the on-screen figure.

# Visual style and delivery

Match Paper’s warm off-white background, serif article typography, rust-brown
accent and WikiLens wordmark. Use the same palette for the pipeline diagram.
Keep the application as the main visual; avoid stock footage and decorative
technical imagery.

Record at **1920 × 1080, 30 fps**. Keep the cursor deliberate and use restrained
zooms to emphasize the unchanged question, the added context and the citation.
Keep text legible at normal playback size. Avoid lengthy scrolling.

Use accurate English captions with high contrast and no more than two lines
at a time. Keep them clear of the question field, context and source quotations.
Use simple cuts or short dissolves between the app and pipeline diagram.

Cut waiting time where needed to meet the target. Mark those transitions
**“Waiting time shortened”**; do not present edited playback as measured latency.
Keep spoken audio prominent over any optional, quiet background music.

Deliver an MP4 using H.264 video and AAC audio. Aim for **115 seconds** and verify
the exported duration is **no longer than 120 seconds**. Check caption timing,
audio clarity and source-text readability in the exported file.

# Required points and limitations

Include:

- The same **talvisota** question without context and with **videopeli** context.
- The corresponding historical and game results.
- The path from query through retrieval to generation.
- HyDE and language detection sharing one Pydantic-structured model response.
- The distinction between a hypothetical retrieval aid and real source evidence.
- LLM review occurring **before** citation attachment.
- Clickable literal quotations and original Wikipedia revision links.
- Resident Modal models and database, avoiding repeated startup between questions.

Leave out:

- Installation, account setup, cloud dashboards, credentials and notebook execution.
- Deep search, which is not enabled in this live demo.
- Older placeholder playback and unrelated examples.
- A GPU benchmark detour or claims that edited video timing measures inference.
- Claims of a final LLM verification after citation attachment; there is no such step.
- Claims of error-free Finnish, guaranteed factual correctness or fully supported
  claims merely because quotations and offsets match.

Claims that need verification before recording:

- Both current searches select the intended subject and return usable results.
- The opened citations display the matching quotation and revision link.
- Any visible runtime belongs to the actual run being shown.
- The live connection is ready and the final video stays within 120 seconds.

Finnish and English language routing have been smoke-tested. Do not turn that
into a claim of validated performance across all languages. Dataset L contains
Finnish sources; multilingual answers do not imply multilingual source coverage.
