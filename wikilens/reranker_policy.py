"""Explicit score filtering and whole-passage generation context packing.

No learned/calibrated threshold is implied by this module. Scores are raw BGE
cross-encoder logits. The caller supplies any experimentally chosen cutoffs.
Imports and packing do not load a model or use CUDA.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from numbers import Real


def _finite_number(value, name):
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def filter_ranked(ranked, min_score=None, relative_gap=None):
    """Return ``(kept, diagnostics)`` without sorting or mutating input nodes.

    Both limits are inclusive. Relative filtering retains scores at least
    ``max(observed scores) - relative_gap``. If both policies are supplied, a
    node must pass both. ``None`` disables a policy; there is no default cutoff,
    forced top-one fallback, sigmoid conversion, or probabilistic interpretation.
    """
    if min_score is not None:
        min_score = _finite_number(min_score, "min_score")
    if relative_gap is not None:
        relative_gap = _finite_number(relative_gap, "relative_gap")
        if relative_gap < 0:
            raise ValueError("relative_gap must be nonnegative")
    ranked = list(ranked)
    scores = []
    for index, node in enumerate(ranked):
        if not isinstance(node, Mapping) or "rerank_score" not in node:
            raise ValueError(f"Node {index} must contain rerank_score")
        scores.append(
            _finite_number(node["rerank_score"], f"Node {index} rerank_score")
        )
    best = max(scores) if scores else None
    relative_min = (
        best - relative_gap if best is not None and relative_gap is not None else None
    )
    limits = [x for x in (min_score, relative_min) if x is not None]
    effective_min = max(limits) if limits else None
    kept, rejected = [], []
    counts = {
        "input": len(ranked),
        "kept": 0,
        "rejected": 0,
        "rejected_absolute": 0,
        "rejected_relative": 0,
        "rejected_both": 0,
    }
    for index, (node, score) in enumerate(zip(ranked, scores)):
        absolute_fail = min_score is not None and score < min_score
        relative_fail = relative_min is not None and score < relative_min
        if absolute_fail or relative_fail:
            reasons = []
            if absolute_fail:
                counts["rejected_absolute"] += 1
                reasons.append("below_min_score")
            if relative_fail:
                counts["rejected_relative"] += 1
                reasons.append("outside_relative_gap")
            if absolute_fail and relative_fail:
                counts["rejected_both"] += 1
            rejected.append(
                {
                    "id": node.get("id"),
                    "input_index": index,
                    "score": score,
                    "reasons": reasons,
                }
            )
        else:
            kept.append(copy.deepcopy(dict(node)))
    counts.update(kept=len(kept), rejected=len(rejected))
    return kept, {
        "score_kind": "raw_relevance_logit",
        "min_score": min_score,
        "relative_gap": relative_gap,
        "best_score": best,
        "effective_min_score": effective_min,
        "counts": counts,
        "kept_ids": [n.get("id") for n in kept],
        "rejected": rejected,
        "note": "Ordering is unchanged. Rejection counters overlap when both policies fail.",
    }


def _prompt_input_tokens(nodes, question, focus, settings, tokenizer, *, review=False):
    # Lazy import keeps this policy module usable without model/UI dependencies.
    from .citation_pipelines import build_posthoc_prompt
    from .finnish_prompts import build_citation_aware_prompt

    data = {
        "question": question,
        "search_focus": focus,
        "passages": [
            {
                "source_id": n["id"],
                "title": n["title"],
                "lang": n["lang"],
                "text": n["text"],
                **(
                    {"revision_url": n["revision_url"]}
                    if n.get("revision_url")
                    and (
                        settings.get("generation_method") == "citation_aware"
                        or settings.get("posthoc_prompt") == "finnish_no_length_target"
                    )
                    else {}
                ),
            }
            for n in nodes
        ],
    }
    builder = (
        build_citation_aware_prompt
        if settings.get("generation_method") == "citation_aware"
        else build_posthoc_prompt
    )
    system, data = builder(data, settings)
    if review:
        from .text_review import review_messages

        messages = review_messages(data, "")
    else:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(data, ensure_ascii=False)},
        ]
    # Match TorchRuntime/ResidentModels exactly: render the chat template, then
    # tokenize once without inserting another layer of special tokens.
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    encoded = tokenizer(rendered, truncation=False, add_special_tokens=False)
    ids = encoded["input_ids"]
    if hasattr(ids, "shape"):
        if len(ids.shape) == 1:
            return int(ids.shape[0])
        if len(ids.shape) == 2 and int(ids.shape[0]) == 1:
            return int(ids.shape[1])
        raise ValueError("Tokenizer returned a batch rather than one prompt")
    if ids and isinstance(ids[0], (list, tuple)):
        if len(ids) != 1:
            raise ValueError("Tokenizer returned a batch rather than one prompt")
        ids = ids[0]
    return len(ids)


def _review_context_tokens(nodes, question, focus, settings, tokenizer):
    if not settings.get("review_before_citations"):
        return 0
    if settings.get("generation_method", "post_hoc") != "post_hoc":
        raise ValueError("review_before_citations currently supports post_hoc only")
    output = _positive_integer(
        settings.get("review_tokens", settings["article_tokens"]), "review_tokens"
    )
    # Reserve the full draft and review output budgets plus tokenizer-boundary
    # slack. The runtime checks the actual combined messages before inference.
    return (
        _prompt_input_tokens(nodes, question, focus, settings, tokenizer, review=True)
        + settings["article_tokens"]
        + output
        + 64
    )


def pack_generation_nodes(nodes, question, focus, settings, tokenizer):
    """Greedily retain whole source nodes whose real draft prompt fits context.

    Returns ``(nodes, diagnostics)``. Reads ``settings['context_tokens']`` and
    ``settings['article_tokens']`` and ``settings.get('max_nodes', 25)``. This
    supports both 8192 and 12288 context settings without changing them. The
    caller remains responsible for configuring a model that supports its chosen
    context length. Additional post-hoc attribution prompts have their own budgets.

    Nodes are considered in input order. A too-long node is skipped; later
    smaller nodes may still fit. Source text, IDs, offsets and metadata are never
    truncated, normalized or renumbered. Returned nodes are independent copies.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a nonempty string")
    if not isinstance(settings, Mapping):
        raise ValueError("settings must be a mapping")
    settings = copy.deepcopy(dict(settings))
    context = _positive_integer(settings.get("context_tokens"), "context_tokens")
    output_tokens = _positive_integer(settings.get("article_tokens"), "article_tokens")
    maximum = _positive_integer(settings.get("max_nodes", 25), "max_nodes")
    if output_tokens >= context:
        raise ValueError(
            "article_tokens must leave room for the input prompt inside context_tokens"
        )
    original = list(nodes)
    seen = set()
    for index, node in enumerate(original):
        if not isinstance(node, Mapping):
            raise ValueError(f"Node {index} must be a mapping")
        for field in ("id", "title", "lang", "text"):
            if not isinstance(node.get(field), str) or not node[field].strip():
                raise ValueError(f"Node {index} must contain a nonempty {field}")
        if node["id"] in seen:
            raise ValueError(f"Duplicate source ID: {node['id']}")
        seen.add(node["id"])
    base_tokens = _prompt_input_tokens([], question, focus, settings, tokenizer)
    if base_tokens + output_tokens > context:
        raise ValueError(
            f"Question/system prompt ({base_tokens}) plus requested output ({output_tokens}) "
            f"exceeds context_tokens ({context}) even with no source passages"
        )
    review_total = _review_context_tokens([], question, focus, settings, tokenizer)
    if review_total > context:
        raise ValueError(
            "Draft/review token budgets exceed context_tokens even with no source passages"
        )
    selected, rejected = [], []
    input_tokens = base_tokens
    for index, node in enumerate(original):
        if len(selected) >= maximum:
            rejected.append(
                {"id": node["id"], "input_index": index, "reason": "max_nodes"}
            )
            continue
        proposal_tokens = _prompt_input_tokens(
            [*selected, node], question, focus, settings, tokenizer
        )
        if proposal_tokens + output_tokens > context:
            rejected.append(
                {
                    "id": node["id"],
                    "input_index": index,
                    "reason": "context_budget",
                    "input_tokens_if_added": proposal_tokens,
                    "total_tokens_if_added": proposal_tokens + output_tokens,
                }
            )
            continue
        proposal_review = _review_context_tokens(
            [*selected, node], question, focus, settings, tokenizer
        )
        if proposal_review > context:
            rejected.append(
                {
                    "id": node["id"],
                    "input_index": index,
                    "reason": "review_context_budget",
                    "review_tokens_if_added": proposal_review,
                }
            )
            continue
        selected.append(copy.deepcopy(dict(node)))
        input_tokens = proposal_tokens
        review_total = proposal_review
    diagnostics = {
        "context_tokens": context,
        "requested_output_tokens": output_tokens,
        "max_nodes": maximum,
        "input_tokens": input_tokens,
        "input_token_estimate": input_tokens,
        "total_tokens_with_output_budget": input_tokens + output_tokens,
        "remaining_context_tokens": context - input_tokens - output_tokens,
        "review_context_tokens_with_reserved_draft_and_output": review_total,
        "empty_source_prompt_tokens": base_tokens,
        "counts": {
            "input": len(original),
            "kept": len(selected),
            "rejected": len(rejected),
            "rejected_context_budget": sum(
                x["reason"] == "context_budget" for x in rejected
            ),
            "rejected_review_context_budget": sum(
                x["reason"] == "review_context_budget" for x in rejected
            ),
            "rejected_max_nodes": sum(x["reason"] == "max_nodes" for x in rejected),
        },
        "kept_ids": [n["id"] for n in selected],
        "rejected": rejected,
        "token_count_method": "actual generation prompt ("
        + settings.get("generation_method", "post_hoc")
        + "), rendered chat template, add_special_tokens=False",
        "scope": "Review-enabled runs also reserve the full draft and revised output budgets. Draft generation prompt; citation-aware includes inline markers and bibliography in its output budget. Any post-hoc attribution is budgeted separately.",
    }
    return selected, diagnostics
