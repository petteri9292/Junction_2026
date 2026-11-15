# Presentation verification — 15 November 2026

## H200 session update

The replacement six-hour H200 session is connected at port 8041 and was verified
with both Talvisota contexts, citations and retrieved documents. It uses the
same model and prompts. The current deadline is **13:18:19 Helsinki** on
15 November, with new requests closing at 13:03:19. It supersedes the expired
overnight H100 session described below. Focused checks passed 32 tests. See
`reports/h200-live-verification.json`.

## Working source

- **244 tests passed** in the working checkout, including mocked inference, settings, quotation provenance, admin diagnostics and the localhost HTTP API.
- An export containing only files eligible for Git passed **242 tests, with 2 expected skips**. It reused the installed Python dependencies, but had no local corpus, raw captures, credentials or archive. The skips require the optional original dataset and an ignored real capture. The notebook control-flow test passed with the project environment entry point present.
- Both root and vLLM dependency locks passed `uv lock --check --offline`.
- All public Python source compiled. Ruff formatting and critical syntax/name checks passed. The broad inherited style-rule set is not claimed as a fully passing lint gate.
- All three shipped browser JavaScript files passed Node syntax checks.
- README/documentation links resolve to retained local files.

## Browser and inference

The live admin panel was exercised against the same resident H100/vLLM and MariaDB services. A normal query showed 60 retrieved articles, 965 scored passages and 7 passages selected for generation. The original article and a selected passage were opened. A cutoff of 20 rejected all candidates while preserving the documents for inspection. Default settings were restored.

After repository cleanup, the existing Modal session still reported ready with the full 568,416-row corpus. No new GPU allocation was made for cleanup. The hard deadline remains **03:29:56 Europe/Helsinki, 15 November**; requests close 15 minutes earlier. The session is intentionally bounded and cannot serve as an assumed morning demo endpoint.

The offline Paper interface was started locally and checked in a browser. Both `talvisota` without context and with context `videopeli` opened the correct recorded result. Citation buttons opened exact quotations and original revision links. The header and timings identify saved runs explicitly. The initial context field is empty. Settings in offline mode are disabled; changing recorded results requires live inference.

## Presentation deliverables

The PowerPoint remains editable. Its five-page PDF export was rendered and visually checked page by page for legibility, clipping and layout. The three main slides emphasize the retrieval–generation pipeline, with disambiguation in the backup material. The script and video brief point to retained evidence.

## Sharing and preservation

The Git-eligible tree was scanned for common Hugging Face, GitHub and API credential patterns, private-key headers, email addresses and selected private infrastructure patterns; no matches were found. Existing Git history was also checked for common credential patterns across 18 historical blobs, with no matches. This is a pattern check, not a security audit.

Every pre-cleanup public candidate was copied into the ignored `_local_archive/` directory before editing. Its manifest records 359 original files with SHA-256 checksums; all reopened checksums matched. Retired files remain there, and original versions of rewritten documents/source are retained. Existing ignored `data/`, `artifacts/`, `.venv/` and environment files remain in place so active services and evidence are preserved. Git ignore checks confirm the archive, active job record, environment, credentials and build artifacts are excluded.

The resulting source still includes the supported pipeline variants and the notebooks required by reproduction scripts. Exploratory layouts, session recovery tools, historical reports and verbose operational logs are outside the public tree. No GitHub push or event submission is implied by these checks.

## Paragraph source grouping

The browser now shows one badge per Wikipedia revision within a paragraph, using
stable source numbers across the article. In a fresh live H200 result the Talvisota
paragraphs each showed one source badge; paragraph 3 opened both of its exact
quotations. The recorded first paragraph opened all three original quotations
under one source. Six JavaScript regression checks passed, covering paragraph
isolation, first-use numbering, different revisions, duplicate quotation removal,
raw-data preservation and the recorded examples. Two fixture integrity tests also
passed. This interface-only change leaves the model, prompts, raw evidence and
running H200 services unchanged.

## Markdown and distinct paragraph citation numbers

This supersedes the earlier source-number reuse: every paragraph/source group
now has its own sequential badge number. Repeated excerpts from the same source
within one paragraph still share a single badge. The displayed article uses
local Markdown rendering for emphasis, lists and other block structure. Raw HTML
remains escaped, unsafe links are rejected, and generated images cannot load
external resources. Leading Finnish dates are kept as prose.

Five Markdown tests, six citation-grouping tests and two fixture tests pass.
The exact reported answer can be reopened through a read-only result link;
this fetches the stored result without another generation call. Completed live
answers retain that result ID in the URL so a reload can display them again.
Raw source data, generated wording, prompts and the H200 session remain unchanged.

## Final source/paragraph labels and defaults

The latest requested labels keep source numbers stable and append the original
paragraph index: `1.1p`, `1.2p`, and `1.12p`. Multiple excerpts from one source
within a paragraph remain grouped. Final live defaults are cutoff −3, 60 retrieved
articles and 6000 article tokens. The review budget and prompts remain unchanged.

Twenty-six focused Python checks and six citation-grouping checks passed. The
6000-token limit was exercised in a real query on the same H200; the later change
back from 30 to 60 articles is covered by request-normalization checks. Existing
answer and document links survive controller handoffs via read-only lookup of
prior sessions. The original vLLM process remains PID 45 with the same start time,
and the original session deadline remains in force.

## Section labels and cutoff update

Removed the decorative 01/02 prefixes from the Question and Answer labels.
The user changed the final default cutoff to −4; 60 retrieved articles and 6000
article output tokens remain selected. The API normalizes these settings before
queueing them, so the warm H200 worker receives the new cutoff explicitly.
This requires only reloading the local web server; no GPU or model restart.
