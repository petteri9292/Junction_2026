import json
import threading
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from wikilens.public_demo import PublicBridge, PublicHandler, public_html


@pytest.fixture
def site():
    values = {}
    backend = SimpleNamespace(session_id="test-session")
    backend.status = lambda: {
        "status": "ready",
        "stage": "/private/path",
        "container_id": "secret",
        "last_request_id": "private",
        "deadline_unix": 123,
        "deep_search_available": True,
    }
    backend.result = lambda rid: values.get(rid, {"status": "missing"})

    def submit(data):
        values[data["request_id"]] = {
            **data,
            "status": "queued",
            "settings": {"private": True},
            "container_id": "secret",
        }
        return {"request_id": data["request_id"]}

    backend.submit = submit
    bridge = PublicBridge(backend, {})
    server = ThreadingHTTPServer(
        ("127.0.0.1", 0),
        partial(PublicHandler, bridge=bridge, origin="https://demo.example"),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(path, method="GET", data=None, headers=None):
        conn = HTTPConnection(*server.server_address)
        conn.request(
            method,
            path,
            body=json.dumps(data) if data is not None else None,
            headers={"Content-Type": "application/json", **(headers or {})},
        )
        response = conn.getresponse()
        body = response.read()
        status = response.status
        conn.close()
        return status, body

    yield request, bridge, values
    server.shutdown()
    server.server_close()


def test_html_removes_admin_entirely():
    html = public_html()
    assert b"admin-" not in html and b"admin.js" not in html
    assert b'id="deep-search"' in html


@pytest.mark.parametrize(
    "path",
    [
        "/admin.js",
        "/admin",
        "/.env",
        "/fixtures/examples.json",
        "/api/questions/" + "a" * 32 + "/diagnostics",
        "/api/questions/" + "a" * 32 + "/documents/D1",
        "/%2e%2e/pyproject.toml",
        "/../pyproject.toml",
    ],
)
def test_private_routes_denied(site, path):
    request, _, _ = site
    assert request(path)[0] == 404
    assert request(path, "HEAD")[0] == 404


def test_status_sanitized(site):
    request, _, _ = site
    status, body = request("/api/status")
    assert status == 200
    assert (
        b"secret" not in body
        and b"/private/path" not in body
        and b"last_request" not in body
    )


def test_settings_and_cross_origin_rejected(site):
    request, _, values = site
    data = {
        "request_id": "a" * 32,
        "question": "Talvisota",
        "settings": {"max_article_tokens": 99999},
    }
    assert request("/api/questions", "POST", data)[0] == 400
    data.pop("settings")
    assert (
        request("/api/questions", "POST", data, {"Origin": "https://evil.example"})[0]
        == 403
    )
    assert not values


def test_private_result_is_not_exposed_or_claimable(site):
    request, _, values = site
    rid = "a" * 32
    values[rid] = {"status": "complete", "question": "private question"}
    assert request("/api/questions/" + rid)[0] == 404
    assert (
        request(
            "/api/questions",
            "POST",
            {"request_id": rid, "question": "private question"},
        )[0]
        == 400
    )


def test_public_roundtrip_filters_private_fields(site):
    request, _, values = site
    rid = "b" * 32
    assert (
        request(
            "/api/questions",
            "POST",
            {"request_id": rid, "question": "Talvisota"},
            {"Origin": "https://demo.example"},
        )[0]
        == 202
    )
    values[rid].update(
        status="complete",
        seconds=10,
        result={
            "answer": [],
            "evidence": [],
            "worker_pid": 123,
            "deep_search": {
                "enabled": True,
                "status": "sufficient",
                "extra_queries": 0,
                "decision": "private prompt",
            },
        },
    )
    status, body = request("/api/questions/" + rid)
    assert status == 200
    assert (
        b"private" not in body and b"secret" not in body and b"worker_pid" not in body
    )
    assert json.loads(body)["result"]["deep_search"]["enabled"] is True


def test_active_public_request_survives_frontend_restart(site):
    _, bridge, _ = site
    bridge.submit({"request_id": "a" * 32, "question": "First"})
    restarted = PublicBridge(bridge.bridge, bridge.admissions)
    with pytest.raises(ValueError, match="question is running"):
        restarted.submit({"request_id": "b" * 32, "question": "Second"})


def test_backend_errors_are_not_public(site):
    request, bridge, _ = site

    def broken():
        raise RuntimeError("database-password-and-private-path")

    bridge.bridge.status = broken
    status, body = request("/api/status")
    assert status == 502
    assert b"database-password" not in body
