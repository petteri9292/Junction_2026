"""Restricted public Paper frontend. Administrative access remains loopback-only."""

import json
import os
import re
import threading
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

STATIC = {
    "/directions.css": "text/css",
    "/direction.js": "text/javascript",
    "/app.js": "text/javascript",
    "/citations.js": "text/javascript",
    "/answer-markdown.js": "text/javascript",
    "/vendor/markdown-it.min.js": "text/javascript",
}
ROOT = Path(__file__).with_name("recording_ui")
RID = r"[a-f0-9]{32}"
# Only fixed pipeline labels may cross the public boundary. Never forward
# arbitrary worker messages, errors, prompts or diagnostic fields.
PIPELINE_STAGES = frozenset(
    {
        "Writing HyDE retrieval aid",
        "Searching Wikipedia articles in MariaDB",
        "Chunking headings and ranking passages",
        "Selecting evidence for your requested context",
        "Deep search: reviewing whether the evidence is sufficient",
        "Deep search: running one follow-up query",
        "Deep search: ranking the combined evidence",
        "Writing the article with Qwen",
        "Reviewing language and source support",
        "Finding references for paragraphs",
        "Selecting exact quotations",
    }
)


def public_stage(value):
    status = value.get("status")
    if status == "running":
        stage = value.get("stage")
        return (
            stage
            if isinstance(stage, str) and stage in PIPELINE_STAGES
            else "Processing your question"
        )
    return {
        "queued": "Waiting for the warm model",
        "complete": "Ready",
        "failed": "Research could not be completed",
        "missing": "Question not found",
    }.get(status, "Processing your question")


def public_html():
    text = (ROOT / "paper.html").read_text()
    text, buttons = re.subn(
        r'<button id="admin-open".*?</button>', "", text, flags=re.DOTALL
    )
    text, dialogs = re.subn(
        r'<dialog id="admin-dialog".*?</dialog>', "", text, flags=re.DOTALL
    )
    text, scripts = re.subn(r'<script src="admin\.js[^\"]*" defer></script>', "", text)
    if (buttons, dialogs, scripts) != (1, 1, 1):
        raise RuntimeError("Public template requires a privacy review")
    return text.encode()


class PublicBridge:
    def __init__(self, bridge, admissions):
        self.bridge = bridge
        self.admissions = admissions
        self.lock = threading.Lock()

    def key(self, rid):
        return self.bridge.session_id + ":" + rid

    def status(self):
        value = self.bridge.status()
        allowed = (
            "status",
            "deadline_unix",
            "accepts_requests_until_unix",
            "deep_search_available",
        )
        result = {key: value[key] for key in allowed if key in value}
        result["live"] = True
        result["stage"] = {
            "ready": "Ready for your question",
            "answering": "Research in progress",
            "starting": "Preparing the research model",
            "closed": "The demo session has ended",
        }.get(value.get("status"), "Research service unavailable")
        return result

    def submit(self, data):
        if not isinstance(data, dict) or set(data) - {
            "request_id",
            "question",
            "context",
            "deep_search",
        }:
            raise ValueError(
                "Only question, context, deep_search and request_id are accepted"
            )
        rid = data.get("request_id", "")
        if not isinstance(rid, str) or not re.fullmatch(RID, rid):
            raise ValueError("Invalid request ID")
        with self.lock:
            if (
                self.admissions.get(self.key(rid)) is None
                and self.bridge.result(rid).get("status") != "missing"
            ):
                raise ValueError("Request ID is unavailable")
            active = self.admissions.get(self.key("active"))
            if (
                active
                and active != rid
                and self.bridge.result(active).get("status")
                not in {"complete", "failed", "missing"}
            ):
                raise ValueError("A question is running. Please wait for it to finish.")
            try:
                accepted = self.bridge.submit(data)
            except ValueError:
                raise ValueError(
                    "Question not accepted. Check the inputs and wait until the model is ready."
                ) from None
            self.admissions[self.key(rid)] = True
            self.admissions[self.key("active")] = rid
            return accepted

    def result(self, rid):
        if self.admissions.get(self.key(rid)) is None:
            return {"status": "missing", "stage": "Question not found"}
        value = self.bridge.result(rid)
        allowed = ("status", "question", "context", "deep_search", "seconds")
        result = {key: value[key] for key in allowed if key in value}
        result["stage"] = public_stage(value)
        if value.get("status") == "failed":
            result["error"] = (
                "Research could not be completed. Please try a new question."
            )
        if value.get("result"):
            result["result"] = {
                key: value["result"][key]
                for key in (
                    "status",
                    "title",
                    "question",
                    "theme",
                    "dataset_release",
                    "answer",
                    "evidence",
                    "timings",
                    "review_status",
                    "support_verified",
                    "answer_language",
                )
                if key in value["result"]
            }
            deep = value["result"].get("deep_search")
            if deep:
                result["result"]["deep_search"] = {
                    key: deep.get(key) for key in ("enabled", "status", "extra_queries")
                }
        return result


class PublicHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, bridge, origin="", **kwargs):
        self.bridge, self.origin = bridge, origin
        super().__init__(*args, **kwargs)

    def log_message(self, *args):
        pass  # Do not log visitor questions or result URLs.

    def respond(self, body, content_type="application/json", status=200):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type + "; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
        )
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def do_GET(self):
        path = urlsplit(self.path).path
        try:
            if path in {"/", "/paper.html"}:
                self.respond(public_html(), "text/html")
            elif path in STATIC:
                self.respond((ROOT / path.lstrip("/")).read_bytes(), STATIC[path])
            elif path == "/api/status":
                self.respond(self.bridge.status())
            elif re.fullmatch("/api/questions/" + RID, path):
                value = self.bridge.result(path.rsplit("/", 1)[1])
                self.respond(
                    value, status=404 if value.get("status") == "missing" else 200
                )
            else:
                self.respond({"error": "Not found"}, status=404)
        except Exception:  # noqa: BLE001 -- never expose remote SDK errors publicly
            self.respond({"error": "Research service unavailable"}, status=502)

    do_HEAD = do_GET

    def do_POST(self):
        if urlsplit(self.path).path != "/api/questions":
            self.respond({"error": "Not found"}, status=404)
            return
        origin = self.headers.get("Origin")
        if origin and origin != self.origin:
            self.respond({"error": "Cross-origin submission refused"}, status=403)
            return
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            self.respond({"error": "Cross-origin submission refused"}, status=403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (
                not 0 < length <= 16384
                or self.headers.get_content_type() != "application/json"
            ):
                raise ValueError("Expected a small JSON question request")
            data = json.loads(self.rfile.read(length))
            self.respond(self.bridge.submit(data), status=202)
        except (ValueError, TypeError) as error:
            self.respond({"error": str(error)}, status=400)
        except Exception:  # noqa: BLE001 -- never expose remote SDK errors publicly
            self.respond(
                {
                    "error": "Question could not be queued. Check status before retrying."
                },
                status=502,
            )


def main():
    import modal

    from .paper_live import ModalPaperBridge

    job_path = Path("/tmp/wikilens-public-job.json")
    job_path.write_text(os.environ["WIKILENS_PUBLIC_JOB"])
    bridge = PublicBridge(
        ModalPaperBridge(job_path),
        modal.Dict.from_name("wikilens-public-admissions", create_if_missing=True),
    )
    handler = partial(
        PublicHandler, bridge=bridge, origin=os.environ["WIKILENS_PUBLIC_ORIGIN"]
    )
    ThreadingHTTPServer(("0.0.0.0", 8000), handler).serve_forever()


if __name__ == "__main__":
    main()
