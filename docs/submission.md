# WikiLens — submission copy

## Punchline

From Wikipedia vectors to readable articles: a MariaDB retrieval–generation pipeline with inspectable source evidence.

## Description

WikiLens is a working retrieval-augmented generation system over the challenge's full Dataset L: 568,416 Finnish Wikipedia articles.

A question and optional context become a Pydantic-structured HyDE response containing a hypothetical search passage and detected answer language. BGE-M3 embeds only the search passage. MariaDB native vector search retrieves articles; heading-aware passage extraction and a multilingual cross-encoder then select the evidence for generation.

Qwen3-4B-Instruct-2507-FP8 drafts an article from the selected passages. A second LLM call reviews the draft, followed by post-hoc citation selection and Python checks that each displayed quotation occurs in the stored source. Readers can inspect the quotations and open the original Wikipedia revision. The admin panel exposes retrieval controls and the actual retrieved documents, scores and selected passages.

The live demo keeps MariaDB and the models resident on one Modal H100 using vLLM. Observed warm end-to-end requests took 16.8 seconds in English and 20.6 seconds in Finnish, excluding startup. These are individual measurements, not latency guarantees. A separate frozen-input comparison records the combined change from L40S/Transformers to H100/vLLM.

Context is one practical demonstration: `talvisota` without context retrieves the historical Winter War, while context `videopeli` directs the pipeline toward the video game. The retrieval corpus remains Finnish even when the generated answer is English.

The project includes reproducible source, locked environments, corpus verification, measured timings, tests, a presentation and recorded fallback runs. Literal quotation checks establish provenance, not semantic correctness. Generated prose can still contain factual and language errors, and there is no final LLM audit after citation attachment.

## Challenge alignment

MariaDB RAG track. MariaDB's native indexed vector search is a central pipeline component. The implementation demonstrates retrieval, generation, review and an inspectable viewer experience. The project does not claim system-versioned time travel, model training or independent factual verification.

## Presentation files

- `reports/wikilens-pipeline-pitch.pptx`
- `reports/wikilens-pipeline-pitch.pdf`
- `pipeline-pitch-script.txt`
- `demo-brief.md` — target video length 115 seconds, maximum 120

## Entry status

This file contains submission copy and local deliverables. It does not establish that the event entry has been submitted, its links are publicly accessible, a video has been uploaded, or the entry is set to FINAL. Those actions must be verified on the event platform. The current demonstration endpoint is local; the labelled recorded fallback is included in the source repository.
