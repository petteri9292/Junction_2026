"""Live boundary checks: evidence integrity, duplicate jobs and local HTTP API."""

import copy
import json
import threading
import time
from functools import partial
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from wikilens.paper_live import ModalPaperBridge, result_for_paper
from wikilens.recording import RecordingHandler


def capture():
    text = "This is a literal source sentence."
    return {
        "question": "talvisota",
        "nodes": [
            {
                "id": "S1",
                "title": "Talvisota",
                "text": text,
                "revision_url": "https://fi.wikipedia.org/w/index.php?oldid=1",
                "rev_id": 1,
            }
        ],
        "sentences": [
            {
                "text": "Generated paragraph.",
                "section": "Overview",
                "citations": [
                    {
                        "source_id": "S1",
                        "quote_located": True,
                        "literal_quote": text,
                        "quote_start_char": 0,
                        "quote_end_char": len(text),
                    }
                ],
            }
        ],
        "input_capture": {"corpus": {"dataset": "l-2026-10"}},
        "interactive_timings": {"total_seconds": 2},
    }


def test_preserves_generated_paragraph_and_exact_source_quote():
    result = result_for_paper(capture())
    assert result["answer"][0]["text"] == "Generated paragraph."
    assert result["answer"][0]["citations"] == ["Q1"]
    assert result["evidence"][0]["passage"] == capture()["nodes"][0]["text"]
    assert result["status"] == "live" and result["support_verified"] is False
    damaged = capture()
    damaged["sentences"][0]["citations"][0]["quote_end_char"] -= 1
    with pytest.raises(ValueError, match="offsets"):
        result_for_paper(damaged)
    damaged = capture()
    damaged["nodes"][0]["revision_url"] = "javascript:alert(1)"
    with pytest.raises(ValueError, match="revision link"):
        result_for_paper(damaged)


def bridge():
    result = ModalPaperBridge.__new__(ModalPaperBridge)
    result.session_id = "session"
    result.job = {"deadline_unix": time.time() + 3600}
    result.results = {
        "session": {
            "status": "ready",
            "accepts_requests_until_unix": time.time() + 1800,
        }
    }
    result.lock = threading.Lock()
    result.active = None
    result.sent = []
    result.queue = SimpleNamespace(
        put=lambda request, **kwargs: result.sent.append(copy.deepcopy(request))
    )
    return result


def test_submit_is_idempotent_and_does_not_restart_or_duplicate_compute():
    service = bridge()
    data = {"request_id": "a" * 32, "question": "talvisota", "context": "videopeli"}
    assert service.submit(data) == service.submit(data)
    assert len(service.sent) == 1
    assert service.sent[0]["context"] == "videopeli"
    assert service.sent[0]["settings"]["score_cutoff"] == -2
    with pytest.raises(ValueError, match="different inputs"):
        service.submit({**data, "settings": {"score_cutoff": 0}})
    with pytest.raises(ValueError, match="different inputs"):
        service.submit({**data, "question": "Another question"})
    with pytest.raises(ValueError, match="still running"):
        service.submit({**data, "request_id": "b" * 32})
    service.results["session:" + data["request_id"]]["status"] = "complete"
    service.submit({**data, "request_id": "b" * 32})
    assert len(service.sent) == 2


def test_custom_admin_settings_reach_the_queued_request():
    service = bridge()
    service.submit(
        {
            "request_id": "c" * 32,
            "question": "talvisota",
            "settings": {
                "score_cutoff": -4.5,
                "retrieval_top_k": 12,
                "max_nodes": 8,
                "article_tokens": 1000,
            },
        }
    )
    settings = service.sent[0]["settings"]
    assert settings["score_cutoff"] == -4.5 and settings["retrieval_top_k"] == 12
    assert settings["max_nodes"] == 8 and settings["article_tokens"] == 1000
    assert settings["relative_gap"] is None


def test_closed_or_stale_worker_cannot_accept_paid_requests():
    service = bridge()
    service.results["session"]["heartbeat_unix"] = time.time() - 100
    with pytest.raises(ValueError, match="heartbeat"):
        service.submit({"request_id": "a" * 32, "question": "talvisota"})
    service.results["session"]["heartbeat_unix"] = time.time()
    service.results["session"]["accepts_requests_until_unix"] = time.time() - 1
    with pytest.raises(ValueError, match="ending"):
        service.submit({"request_id": "a" * 32, "question": "talvisota"})
    service.job["deadline_unix"] = time.time() - 1
    assert service.status()["status"] == "closed"


def test_local_http_live_api_and_cross_origin_rejection(tmp_path):
    service = bridge()
    (tmp_path / "paper.html").write_text("PAPER")
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(RecordingHandler, directory=str(tmp_path), bridge=service),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = "http://127.0.0.1:" + str(server.server_port)
    try:
        with urlopen(base + "/") as response:
            assert response.read() == b"PAPER"
        with urlopen(base + "/api/status") as response:
            assert json.load(response)["live"] is True
        data = json.dumps(
            {"request_id": "a" * 32, "question": "talvisota", "context": "videopeli"}
        ).encode()
        request = Request(
            base + "/api/questions",
            data=data,
            headers={"Content-Type": "application/json", "Origin": base},
        )
        with urlopen(request) as response:
            assert response.status == 202
        with urlopen(base + "/api/questions/" + "a" * 32) as response:
            assert json.load(response)["status"] == "queued"
        service.results["session:" + "a" * 32 + ":diagnostics"] = {
            "stage": "retrieval",
            "documents": [{"id": "D1"}],
        }
        service.results["session:" + "a" * 32 + ":document:D1"] = {
            "title": "Original document",
            "text": "Literal source",
        }
        with urlopen(base + "/api/questions/" + "a" * 32 + "/diagnostics") as response:
            assert json.load(response)["documents"][0]["id"] == "D1"
        with urlopen(base + "/api/questions/" + "a" * 32 + "/documents/D1") as response:
            assert json.load(response)["text"] == "Literal source"
        request = Request(
            base + "/api/questions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Origin": "https://other.example",
            },
        )
        with pytest.raises(HTTPError) as error:
            urlopen(request)
        assert error.value.code == 403
        assert len(service.sent) == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
