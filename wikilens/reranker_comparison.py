"""Compare existing BGE embeddings, a requested static embedder and a cross-encoder.

Pretrained inference only. All methods rank identical prepared source chunks.
"""

import gc
import time
from .personalization import normalize_focus

STATIC_MODEL = "Jarbas/m2v-256-bge-reranker-large"
STATIC_REVISION = "5a3bcf484470f068bb2ca31d3ce40daeb0d4d606"
CROSS_MODEL = "BAAI/bge-reranker-v2-m3"
CROSS_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
MODELS = {
    "bge_m3": ("BAAI/bge-m3", "5617a9f61b028005a4858fdac845db406aefb181"),
    "model2vec": (STATIC_MODEL, STATIC_REVISION),
    "cross_encoder": (CROSS_MODEL, CROSS_REVISION),
}


def prepare_chunks(articles, **chunk_settings):
    from transformers import AutoTokenizer
    from .torch_runtime import BGE_MODEL, BGE_REVISION
    from .heading_postprocessing import heading_chunks

    tokenizer = AutoTokenizer.from_pretrained(
        BGE_MODEL, revision=BGE_REVISION, use_fast=True
    )
    return heading_chunks(articles, tokenizer, **chunk_settings)


def select_nodes(ranked, *, max_nodes=10, context_chars=7000, max_per_article=3):
    nodes, counts, used = [], {}, 0
    for chunk in ranked:
        key = (chunk["lang"], chunk["page_id"], chunk["rev_id"])
        if (
            counts.get(key, 0) >= max_per_article
            or used + len(chunk["text"]) > context_chars
        ):
            continue
        if any(
            (s["lang"], s["page_id"], s["rev_id"]) == key
            and max(
                0,
                min(s["end_char"], chunk["end_char"])
                - max(s["start_char"], chunk["start_char"]),
            )
            / min(len(s["text"]), len(chunk["text"]))
            > 0.5
            for s in nodes
        ):
            continue
        nodes.append({**chunk, "node_id": chunk["id"], "id": f"S{len(nodes) + 1}"})
        used += len(chunk["text"])
        counts[key] = counts.get(key, 0) + 1
        if len(nodes) >= max_nodes:
            break
    return nodes, used


def rank_chunks(
    runtime,
    question,
    chunks,
    *,
    method="cross_encoder",
    search_focus="",
    max_nodes=10,
    context_chars=7000,
    max_per_article=3,
    max_tokens=512,
    cross_batch_size=8,
    progress=None,
):
    started = time.perf_counter()
    focus = normalize_focus(search_focus)
    query = question + ("\nSearch focus: " + focus if focus else "")
    if method not in MODELS:
        raise ValueError("method must be bge_m3, model2vec or cross_encoder")
    model_id, revision = MODELS[method]
    texts = [chunk["embedding_text"] for chunk in chunks]
    diagnostics = {"model": model_id, "revision": revision}
    if progress:
        progress(f"{method}: ranking {len(chunks)} identical chunks")
    if method == "bge_m3":
        q = runtime.embed([query], batch_size=1, max_tokens=max_tokens)[0]
        vectors = runtime.embed(texts, batch_size=2, max_tokens=max_tokens)
        scores = [sum(a * b for a, b in zip(q, v)) for v in vectors]
        diagnostics.update(
            device=runtime.device, score_kind="cosine_similarity", batch_size=2
        )
        runtime.close()
    elif method == "model2vec":
        runtime.close()
        from huggingface_hub import snapshot_download
        from model2vec import StaticModel

        path = snapshot_download(model_id, revision=revision, local_files_only=True)
        model = StaticModel.from_pretrained(path, normalize=True, force_download=False)
        scores = []
        try:
            vectors = model.encode(
                [query] + texts,
                max_length=None,
                batch_size=1024,
                use_multiprocessing=False,
            )
            scores = (vectors[1:] @ vectors[0]).astype(float).tolist()
            diagnostics.update(
                device="cpu",
                score_kind="static_embedding_cosine",
                dimensions=int(vectors.shape[1]),
                batch_size=1024,
            )
        finally:
            model = vectors = None
            gc.collect()
    else:
        runtime.close()
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        if not torch.cuda.is_available():
            raise RuntimeError("Use the existing GPU environment")
        model = tokenizer = inputs = outputs = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, revision=revision, local_files_only=True
            )
            model = (
                AutoModelForSequenceClassification.from_pretrained(
                    model_id,
                    revision=revision,
                    local_files_only=True,
                    dtype=torch.float16,
                    use_safetensors=True,
                )
                .to(runtime.device)
                .eval()
            )
            device = str(next(model.parameters()).device)
            if not device.startswith("cuda"):
                raise RuntimeError("Cross-encoder must be on GPU")
            torch.cuda.reset_peak_memory_stats(runtime.device)
            scores = []
            for start in range(0, len(texts), cross_batch_size):
                pairs = [
                    [query, passage]
                    for passage in texts[start : start + cross_batch_size]
                ]
                inputs = tokenizer(
                    pairs, padding=True, truncation=False, return_tensors="pt"
                )
                if inputs["input_ids"].shape[1] > max_tokens:
                    raise ValueError(
                        "Query + passage exceeds pair token budget; reduce chunk_tokens explicitly"
                    )
                inputs = inputs.to(runtime.device)
                with torch.inference_mode():
                    outputs = model(**inputs, return_dict=True)
                    scores.extend(outputs.logits.view(-1).float().cpu().tolist())
                inputs = outputs = None
                if progress and (start // cross_batch_size) % 20 == 0:
                    progress(
                        f"Cross-encoder scored {min(start + cross_batch_size, len(texts))}/{len(texts)}"
                    )
            diagnostics.update(
                device=device,
                score_kind="raw_relevance_logit",
                batch_size=cross_batch_size,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(runtime.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(runtime.device),
            )
        finally:
            model = tokenizer = inputs = outputs = None
            gc.collect()
            torch.cuda.empty_cache()
    ranked = sorted(
        ({**c, "rerank_score": float(score)} for c, score in zip(chunks, scores)),
        key=lambda c: (-c["rerank_score"], c["retrieval_rank"], c["start_char"]),
    )
    nodes, used = select_nodes(
        ranked,
        max_nodes=max_nodes,
        context_chars=context_chars,
        max_per_article=max_per_article,
    )
    return {
        "method": method,
        "status": "completed" if nodes else "insufficient_evidence",
        "nodes": nodes,
        "ranked": ranked,
        "candidate_count": len(chunks),
        "context_chars": used,
        "seconds": time.perf_counter() - started,
        "diagnostics": diagnostics,
        "config": {
            "query": query,
            "max_tokens": max_tokens,
            "max_nodes": max_nodes,
            "context_chars": context_chars,
            "max_per_article": max_per_article,
        },
        "limitation": "Rank scores are not calibrated probabilities or verified factual support.",
    }
