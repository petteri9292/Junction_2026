"""Private local Paper-to-Modal bridge; never starts paid compute on a request."""

import json
import re
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

QUEUE = "wikilens-paper-questions"
RESULTS = "wikilens-paper-results"


def result_for_paper(result):
    """Keep model paragraphs and exact selected quotations, without HTML injection."""
    nodes = {node["id"]: node for node in result["nodes"]}
    evidence, answer = [], []
    for paragraph in result["sentences"]:
        ids = []
        for citation in paragraph["citations"]:
            node = nodes[citation["source_id"]]
            url = urlsplit(node["revision_url"])
            if url.scheme != "https" or url.hostname != "fi.wikipedia.org":
                raise ValueError("Expected a Finnish Wikipedia revision link")
            if not citation.get("quote_located"):
                continue
            quote = citation["literal_quote"]
            if (
                node["text"][citation["quote_start_char"] : citation["quote_end_char"]]
                != quote
            ):
                raise ValueError("Quotation offsets differ from the saved source")
            eid = "Q" + str(len(evidence) + 1)
            ids.append(eid)
            evidence.append(
                {
                    "id": eid,
                    "title": node["title"],
                    "lang": "fi",
                    "language_label": "Finnish",
                    "revision_url": node["revision_url"],
                    "rev_id": node.get("rev_id"),
                    "passage": quote,
                    "source_id": node["id"],
                }
            )
        answer.append(
            {
                "text": paragraph["text"],
                "section": paragraph.get("section"),
                "citations": ids,
            }
        )
    return {
        "status": "live",
        "title": result["question"],
        "question": result["question"],
        "theme": "",
        "dataset_release": result["input_capture"]["corpus"]["dataset"],
        "answer": answer,
        "evidence": evidence,
        "timings": result["interactive_timings"],
        "review_status": result.get("review_status"),
        "support_verified": False,
        "answer_language": result.get("answer_language", "Finnish"),
        "language_detection": result.get("language_detection"),
    }


class ModalPaperBridge:
    def __init__(self, job_path):
        import modal

        self.job = json.loads(Path(job_path).read_text())
        self.session_id = self.job["run_id"]
        self.queue = modal.Queue.from_name(QUEUE, create_if_missing=True)
        self.results = modal.Dict.from_name(RESULTS, create_if_missing=True)
        self.lock = threading.Lock()
        self.active = None

    def status(self):
        value = self.results.get(
            self.session_id,
            {"status": "starting", "stage": "Allocating the demo worker"},
        )
        if time.time() > self.job["deadline_unix"]:
            value = {**value, "status": "closed", "stage": "The warm session has ended"}
        elif value.get("heartbeat_unix", time.time()) < time.time() - 90:
            value = {
                **value,
                "status": "unavailable",
                "stage": "Worker heartbeat is stale; inspect session logs",
            }
        return value

    def submit(self, data):
        from .admin_tools import PaperSettings

        settings = PaperSettings.model_validate(data.get("settings", {})).model_dump()
        question, context = data.get("question"), data.get("context", "")
        request_id = data.get("request_id", "")
        if not isinstance(question, str) or not 1 <= len(question.strip()) <= 500:
            raise ValueError("Enter a question of at most 500 characters")
        if not isinstance(context, str) or len(context) > 2000:
            raise ValueError("Context must be at most 2000 characters")
        if not isinstance(request_id, str) or not re.fullmatch(
            "[a-f0-9]{32}", request_id
        ):
            raise ValueError("Invalid request ID")
        if data.get("deep_search"):
            raise ValueError("Deep search is not enabled in this demo")
        with self.lock:
            key = self.session_id + ":" + request_id
            existing = self.results.get(key)
            if existing is not None:
                if (
                    existing.get("question") != question.strip()
                    or existing.get("context", "") != context.strip()
                    or existing.get("settings", PaperSettings().model_dump())
                    != settings
                ):
                    raise ValueError("Request ID already used for different inputs")
                return {"request_id": request_id}
            session = self.status()
            if session["status"] not in {"ready", "answering"}:
                raise ValueError(session.get("stage", "The model is not ready"))
            if time.time() > session["accepts_requests_until_unix"]:
                raise ValueError("Session is ending; no new question was submitted")
            if self.active:
                current = self.results.get(self.session_id + ":" + self.active, {})
                if current.get("status") not in {"complete", "failed"}:
                    raise ValueError(
                        "A question is still running; wait for it to finish"
                    )
            request = {
                "request_id": request_id,
                "question": question.strip(),
                "context": context.strip(),
                "settings": settings,
            }
            self.results[key] = {
                **request,
                "status": "queued",
                "stage": "Waiting for the warm model",
            }
            self.active = request_id
            self.queue.put(request, partition=self.session_id)
            return {"request_id": request_id}

    def result(self, request_id):
        if not re.fullmatch("[a-f0-9]{32}", request_id):
            raise ValueError("Invalid request ID")
        return self.results.get(
            self.session_id + ":" + request_id,
            {"status": "missing", "stage": "Question not found"},
        )

    def diagnostics(self, request_id, document_id=None):
        if not re.fullmatch("[a-f0-9]{32}", request_id):
            raise ValueError("Invalid request ID")
        if document_id is not None and not re.fullmatch(
            r"D[1-9][0-9]{0,2}", document_id
        ):
            raise ValueError("Invalid document ID")
        suffix = ":document:" + document_id if document_id else ":diagnostics"
        return self.results.get(
            self.session_id + ":" + request_id + suffix,
            {
                "stage": "pending",
                "documents": [],
                "error": "Diagnostics are not available yet",
            },
        )
