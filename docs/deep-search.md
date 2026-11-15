# Optional deep search

Deep search is off by default. Its toggle sits next to Add context in the live
Paper interface. The standard search path remains available with the toggle off.

With deep search on, the initial HyDE, language detection, vector search,
reranking, focus filtering and context packing run normally. Before drafting,
a separate reviewer checks whether the selected source evidence sufficiently
covers the original question in its context. This is an evidence review, not
an independent factual audit or a review of an already written draft.

The reviewer returns a Pydantic-constrained object:

- `good_enough`: whether the selected evidence covers the request;
- `reason`: a brief explanation;
- `follow_up_query`: one Finnish query, or null.

If a useful gap query is proposed, the engine executes **one additional vector
retrieval**. It deduplicates article revisions, then runs the existing chunking,
reranking, cutoff, focus and token-packing rules over the combined evidence.
Article generation, draft review and citation attachment then run once, using
the original question and answer language. There is no recursive search loop.

The query and review are retrieval aids, never evidence for article claims.
The answer uses only original Wikipedia passages. If review/extra retrieval
fails, or combined evidence cannot be used, the initial valid evidence remains
available to the normal pipeline. If neither search supplies usable evidence,
the request fails clearly rather than manufacturing an answer.

## Observability

The live progress display identifies evidence review, the single follow-up
query and combined evidence ranking. The result timing states whether an extra
query was attempted. Admin diagnostics show the decision, reason, follow-up
query and selected sources. Completed run records retain raw review messages, structured output, retrieved
articles and timings with the ordinary private run data.

## Disable or revert

For one question, leave **Deep search off**. That performs no deep reviewer call
and no supplementary retrieval. Existing article and review prompts are unchanged.

To disable the feature for the entire local demo, restart only the web server
with the feature flag below. The existing H200, vLLM and MariaDB can stay warm:

```bash
WIKILENS_DEEP_SEARCH_ENABLED=0 scripts/record-wikilens.sh --port 8041
```

The toggle is hidden and deep-search submissions are rejected before queueing.
Normal search continues. Restart without that environment override to re-enable.
The feature is isolated on branch `codex/deepsearch-once`. The previous working
branch `codex/qwen4b-comparison` remains at baseline commit `2aca800`. Prefer the
feature flag for an immediate demo rollback. For a source rollback, revert the
deep-search commit or return to the baseline branch and restart only the local
web server; do not reset or delete preserved project history.

The existing six-hour H200 deadline and compute reservation are unchanged.
Deep search takes longer when it performs review and supplementary retrieval.
It is bounded experimentation, not a demonstrated improvement in answer quality.
