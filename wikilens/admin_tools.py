"""Validated per-request controls and inspectable retrieval diagnostics."""

from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field


class PaperSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    score_cutoff: float | None = Field(default=-3.0, ge=-20, le=20)
    relative_gap: float | None = Field(default=None, ge=0, le=20)
    retrieval_top_k: int = Field(default=60, ge=1, le=100)
    max_nodes: int = Field(default=25, ge=1, le=40)
    article_tokens: int = Field(default=6000, ge=256, le=6000)


def document_key(item):
    return item["lang"], item["page_id"], item["rev_id"]


def passage_key(item):
    return (*document_key(item), item["start_char"], item["end_char"])


class RetrievalInspector:
    def __init__(self, question, context, settings):
        self.snapshot = {
            "question": question,
            "context": context,
            "settings": settings,
            "stage": "pending",
            "documents": [],
            "hyde": None,
        }
        self.documents = {}
        self.identities = {}

    def capture(self, stage, data):
        self.snapshot["stage"] = stage
        if stage == "hyde":
            self.snapshot["hyde"] = {
                "text": data.get("text"),
                "detected_language": data.get("detected_language"),
                "role": "retrieval_only_never_evidence",
            }
        elif stage == "retrieval":
            for index, article in enumerate(data["articles"], 1):
                link = urlsplit(article["revision_url"])
                if link.scheme != "https" or link.hostname != "fi.wikipedia.org":
                    raise ValueError("Expected Finnish Wikipedia revision URL")
                did = "D" + str(index)
                self.identities[document_key(article)] = did
                self.documents[did] = {
                    key: article.get(key)
                    for key in (
                        "title",
                        "lang",
                        "page_id",
                        "rev_id",
                        "revision_url",
                        "retrieval_rank",
                        "distance",
                        "text",
                    )
                }
                self.documents[did].update(id=did, passages=[])
        elif stage == "reranking":
            policy = data["score_filter"]
            kept = set(policy["kept_ids"])
            self.snapshot["score_filter"] = {
                key: policy.get(key)
                for key in (
                    "score_kind",
                    "min_score",
                    "relative_gap",
                    "best_score",
                    "effective_min_score",
                    "counts",
                )
            }
            self.snapshot["ranking_question"] = data.get("ranking_question")
            for node in data["ranked"]:
                did = self.identities[document_key(node)]
                self.documents[did]["passages"].append(
                    {
                        key: node.get(key)
                        for key in (
                            "id",
                            "text",
                            "heading_path",
                            "start_char",
                            "end_char",
                            "rerank_score",
                        )
                    }
                )
                self.documents[did]["passages"][-1].update(
                    passed_cutoff=node["id"] in kept, selected=False, source_id=None
                )
        elif stage == "selection":
            chosen = {passage_key(node): node["id"] for node in data["nodes"]}
            for document in self.documents.values():
                for passage in document["passages"]:
                    sid = chosen.get(
                        (
                            *document_key(document),
                            passage["start_char"],
                            passage["end_char"],
                        )
                    )
                    passage.update(selected=sid is not None, source_id=sid)
            self.snapshot["selected_passages"] = len(data["nodes"])
        self.snapshot["documents"] = []
        for document in self.documents.values():
            scores = [p["rerank_score"] for p in document["passages"]]
            self.snapshot["documents"].append(
                {
                    **{
                        key: document[key]
                        for key in (
                            "id",
                            "title",
                            "lang",
                            "page_id",
                            "rev_id",
                            "revision_url",
                            "retrieval_rank",
                            "distance",
                        )
                    },
                    "best_score": max(scores) if scores else None,
                    "passage_count": len(scores),
                    "passing_count": sum(
                        p["passed_cutoff"] for p in document["passages"]
                    ),
                    "selected_count": sum(p["selected"] for p in document["passages"]),
                }
            )
        return self.snapshot
