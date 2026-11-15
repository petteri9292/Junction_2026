# Measurements and experiment notes

These are measured development observations, not a held-out answer-quality benchmark. Machine-readable values are in `reports/metrics.json`. Raw prompts, outputs, source passages, code hashes and operational logs remain preserved locally in ignored data and the pre-cleanup archive.

## H200 six-hour live session

On 15 November the user authorized one H200 session for six hours, with a
USD32 reservation. Physical hardware was verified as NVIDIA H200, 143771 MiB.
The same pinned Qwen checkpoint, prompt files, corpus and vLLM lock were reused;
no model training or prompt changes were made. Startup took 410.271 seconds.

Fresh requests through the demo HTTP API completed in **23.645 seconds** for
`talvisota` without context and **14.576 seconds** with context `videopeli`.
Both retrieved 60 articles, displayed source quotations (15 and 11 respectively),
and passed a full retrieved-document inspection on the same resident container.
These are smoke checks with fresh retrieval, not a controlled H100/H200 speed
comparison. See `reports/h200-live-verification.json`.

The demo at port 8041 was switched to this session. It remains warm until
13:18:19 Helsinki, with new questions accepted until 13:03:19. Six hours includes
startup. Combined conservative reservations are USD93.4725 against the
user-approved USD100 cap; these are not reconciled provider charges.

## Corpus

The completed Dataset L import contains 568,416 Finnish Wikipedia articles and 1024-dimensional BGE-M3 vectors. All 57 embedding shards were independently reopened and checked for alignment, hashes, norms and finite values. 2,441 inputs exceeded the 8192-token embedding limit and were marked truncated. Embedding GPU batches took 24.44 minutes across the completed checkpointed run; this excludes CPU preparation, orchestration and verification. This is pretrained inference, not training.

## H100 and vLLM: frozen inputs

The same pinned Qwen3-4B-Instruct-2507-FP8 checkpoint, exact prompts and saved passages were replayed. Both hardware and inference engine changed together.

| Case and stage | L40S / Transformers | H100 / vLLM |
| --- | ---: | ---: |
| talvisota: draft | 70.053 s | 3.544 s |
| talvisota: review | 70.365 s | 3.133 s |
| talvisota + videopeli: draft | 42.534 s | 2.138 s |
| talvisota + videopeli: review | 34.380 s | 1.390 s |

The full fixed-evidence draft → review → citation path took 8.773 s and 5.142 s on H100. Those values exclude upstream HyDE/retrieval/reranking and model startup. Startup for this comparison was 335.863 s. Prompt equality was checked; generated outputs and lengths generally differ, so these are not isolated GPU speedups. One observation per request, no uncertainty estimate.

## Fresh live retrieval

One warm H100 session reused its resident models and complete MariaDB corpus across questions. Initial live setup took 437.873 s, excluded below.

| Question | Context | Worker request time |
| --- | --- | ---: |
| talvisota | empty | 25.408 s |
| talvisota | videopeli | 11.215 s |

The corresponding pipeline-only totals are 25.219 s and 11.104 s. The public offline examples preserve these runs, their exact quotations and pipeline times. They predate automatic answer-language detection. Playback does not measure inference speed.

## Structured HyDE and answer language

The same warm H100 and vLLM process were reused after the controller update. The Pydantic response orders `hypothetical_passage` before nullable `language`; only the Finnish passage is embedded. Language detection adds no model call. Unknown language falls back to Finnish.

| Question | Detected language | Worker request time | Located quotations |
| --- | --- | ---: | ---: |
| What was the Winter War? | en | 16.818 s | 13 |
| Mikä oli talvisota? | fi | 20.568 s | 26 |

For non-Finnish questions, passage reranking includes a bounded Finnish HyDE hint alongside the original question. This fixed one observed cutoff failure in English, but has not been evaluated as a general retrieval improvement. HyDE is never supporting evidence.

## Admin inspector

A live default-settings request retrieved 60 articles, scored 965 passages, retained 13 at the −2 cutoff and selected 7 for the generation context. A deliberately extreme cutoff of 20 with top-k 10 rejected all 138 candidate passages; the request failed clearly and its retrieved documents remained inspectable. The browser successfully opened a full retrieved article and a selected source-aligned passage. Defaults were restored afterward. No additional GPU allocation was created for these checks.

## Quality limitations

The model can produce incorrect Finnish and unsupported claims despite its review pass. Observed examples include a historical draft retained unchanged after review and an inference that a planned platform release failed, which the source passage did not establish. No generated prose was silently corrected for the public fallback. A literal quotation can be valid while failing to support the associated claim.

The LLM reviews the draft before citation attachment. There is no final review of the fully cited article. Current measurements support a working pipeline and substantial observed inference speed improvement; they do not establish factual correctness, better answer quality or a latency service level.

## Repository preparation

The public source tree retains working pipeline variants, experiment runners, notebooks, tests, selected measurements and pitch materials. Development recovery scripts, exploratory UI layouts, superseded notebooks, verbose logs and historical reports were moved into `_local_archive/`, which Git ignores. Original versions of rewritten files are preserved there with a SHA-256 manifest. Existing ignored data, credentials, environments and compute ledgers remain in their working locations. See `docs/verification.md` for the final presentation checks.

## Paragraph citation display

The interface now groups exact quotations by Wikipedia revision within each
paragraph. Live browser verification showed one source badge per paragraph and
all linked excerpts in its source card; the recorded first paragraph retained
all three excerpts. Six JavaScript grouping checks and two fixture integrity
tests passed. Raw citation output and quotations are preserved, with no model
or prompt changes and no H200 restart.

## Rendering correction

The viewer now renders Markdown rather than exposing emphasis/list syntax.
Paragraph/source groups receive distinct sequential citation numbers, including
when the same Wikipedia revision is used in different paragraphs. The exact
reported answer is reused for browser verification through a read-only result
link; no generation is required for that check. Five Markdown and six grouping
checks pass, alongside two fixture integrity checks. This changes presentation
only and does not alter model prompts or stored generated prose.

## Source/paragraph labels and revised demo defaults

Citation labels now retain the source number and add the original paragraph
index: `1.1p`, `1.2p`, and so on. Each source still appears once per paragraph
and opens all associated exact excerpts. The requested live defaults are cutoff
−3, 30 retrieved articles and 6000 article output tokens; relative gap remains
disabled and the passage cap remains 25. Review settings and prompts are unchanged.

The request controller was replaced inside the existing H200 allocation, keeping
vLLM PID 45 and MariaDB resident and preserving the original 13:18:19 deadline.
Saved results and document inspection fall back through prior controller sessions
so existing result URLs remain readable. A real request with no settings overrides
completed in 13.164 seconds, retrieved 30 articles and confirmed all three requested
settings. See `reports/paragraph-defaults-verification.json`. No new GPU job was
created. The existing token-aware packing keeps source evidence and review input
within the unchanged 12,288-token model context.

The user subsequently restored the retrieved-article default to 60. The final
defaults are therefore cutoff −3, top-k 60 and 6000 article tokens. Normalized
requests carry these settings explicitly to the already warm controller; no
second controller or model restart was needed for the change back to 60. The
30-article live check above remains recorded as a historical observation.

## Section labels and cutoff update

Removed the decorative 01/02 prefixes from the Question and Answer labels.
The user changed the final default cutoff to −4; 60 retrieved articles and 6000
article output tokens remain selected. The API normalizes these settings before
queueing them, so the warm H200 worker receives the new cutoff explicitly.
This requires only reloading the local web server; no GPU or model restart.

## Optional one-query deep search

Added an off-by-default evidence-sufficiency review before drafting. A useful
gap can trigger one extra direct vector retrieval, followed by the standard
ranking, context packing, article, review and citation stages. No iterative
loop or change to the normal generation/review prompts is introduced.

Live checks on the existing H200: the start-date question completed in 16.408 s
with no extra query; the trade question in 20.695 s with no extra query; a three-part
question in 30.362 s with exactly one extra query and 58 additional article revisions.
Normal search with Deep search off completed in 14.889 s with no extra reviewer
or retrieval. These are individual observations with different outputs, not
a controlled speed or answer-quality comparison. The multi-part reviewer
omitted an explicit assessment of one component, so its judgment remains
unverified. Raw reviewer output is retained unchanged.

All 259 local tests passed. A temporary HTTP server with the feature flag disabled
reported the capability unavailable and rejected deep requests before queueing.
The original H200 allocation, vLLM process and absolute deadline remain in use.
See `reports/deep-search-verification.json` and `docs/deep-search.md`.

The multi-part final answer rejected the Finlandia component rather than
answering it. This remains a recorded failure example: the implementation proves
the bounded control flow, not reliable completeness or factual correctness.

## Public Modal presentation deployment — 15 November

The user approved a USD150 combined HF/Modal cap. Reserved USD32 for a separate,
one-attempt H200 session with a 21600-second timeout, and USD1 for its small public
CPU frontend. Combined conservative reservations are USD126.4725; these are not
reconciled provider charges. The original H200 remained ready and unchanged.

Published https://petteri-hallipelto--wikilens.us-east.modal.run/ with strict
public routes and a separate localhost-only operator page on port 8043. No public
admin controls, settings overrides, raw retrieval diagnostics or allocation
metadata are exposed. Live checks confirmed denied admin routes and cross-origin
submissions; the private operator endpoint returned all 60 retrieved documents.

The new H200 took 488.996 seconds to initialize, including model/kernel warm-up.
Browser-run `talvisota` without context completed in 33.896 seconds; with
`videopeli` context it completed in 22.4 seconds. Both used the same resident
allocation and unchanged model/prompts. They selected the historical and game
articles respectively. Citation expansion showed the grouped paragraph excerpts
and stored Wikipedia revision links. These are individual observations, not a
controlled performance comparison. Finnish language and factual errors remain;
output was retained unchanged. All 274 tests passed, including public HTTP
privacy and frontend-restart admission checks.

The warm session ends at 15:22 Helsinki time; new work is accepted until 15:07.
See `reports/public-demo-verification.json` and `docs/public-demo.md`.
