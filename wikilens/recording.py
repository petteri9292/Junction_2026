"""Local Paper interface with saved-run playback; no third-party imports."""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


class RecordingHandler(SimpleHTTPRequestHandler):
    """Serve the selected Paper interface at the local root."""

    def __init__(self, *args, bridge=None, **kwargs):
        self.bridge = bridge
        super().__init__(*args, **kwargs)

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        target = urlsplit(self.path)
        if target.path.startswith("/api/"):
            try:
                if target.path == "/api/status":
                    self.send_json(
                        {
                            "live": self.bridge is not None,
                            **(
                                self.bridge.status()
                                if self.bridge
                                else {"status": "offline"}
                            ),
                        }
                    )
                elif self.bridge and target.path.startswith("/api/questions/"):
                    parts = target.path.removeprefix("/api/questions/").split("/")
                    if len(parts) == 2 and parts[1] == "diagnostics":
                        self.send_json(self.bridge.diagnostics(parts[0]))
                    elif len(parts) == 3 and parts[1] == "documents":
                        self.send_json(self.bridge.diagnostics(parts[0], parts[2]))
                    elif len(parts) == 1:
                        self.send_json(self.bridge.result(parts[0]))
                    else:
                        self.send_json({"error": "Endpoint not found"}, 404)
                else:
                    self.send_json({"error": "Endpoint not found"}, 404)
            except ValueError as error:
                self.send_json({"error": str(error)}, 400)
            except Exception:  # noqa: BLE001 -- contain remote SDK failures at the HTTP boundary
                if target.path == "/api/status":
                    self.send_json(
                        {
                            "live": self.bridge is not None,
                            "status": "unavailable",
                            "stage": "Cannot reach the Modal session. Inspect the local server log.",
                        }
                    )
                else:
                    self.send_json(
                        {
                            "error": "Cannot reach the Modal session. Inspect the local server log."
                        },
                        502,
                    )
            return
        if target.path == "/":
            self.path = urlunsplit(("", "", "/paper.html", target.query, ""))
        super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path != "/api/questions" or self.bridge is None:
            self.send_json({"error": "Live search is not connected"}, 404)
            return
        origin = self.headers.get("Origin")
        if self.headers.get("Host") not in {
            f"127.0.0.1:{self.server.server_port}",
            f"localhost:{self.server.server_port}",
        }:
            self.send_json({"error": "Only localhost submissions are accepted"}, 403)
            return
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.send_json({"error": "Cross-origin submission refused"}, 403)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if (
                not 0 < length <= 16384
                or self.headers.get_content_type() != "application/json"
            ):
                raise ValueError("Expected a small JSON question request")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise TypeError("Expected a question object")
            self.send_json(self.bridge.submit(data), 202)
        except (TypeError, ValueError) as error:
            self.send_json({"error": str(error)}, 400)
        except Exception:  # noqa: BLE001 -- contain remote SDK failures at the HTTP boundary
            self.send_json(
                {
                    "error": "Question could not be queued. Inspect session status before retrying."
                },
                502,
            )

    def do_HEAD(self):
        target = urlsplit(self.path)
        if target.path == "/":
            self.path = urlunsplit(("", "", "/paper.html", target.query, ""))
        super().do_HEAD()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8040)
    parser.add_argument(
        "--live",
        action="store_true",
        help="Connect Paper to the already-started warm Modal session",
    )
    parser.add_argument(
        "--job",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/modal-paper-live-job.json",
    )
    args = parser.parse_args()
    root = Path(__file__).with_name("recording_ui")
    bridge = None
    if args.live:
        from .paper_live import ModalPaperBridge

        bridge = ModalPaperBridge(args.job)
    handler = partial(RecordingHandler, directory=str(root), bridge=bridge)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"WikiLens · Paper: http://127.0.0.1:{args.port}/", flush=True)
    print(
        "Live Modal session stays warm independently of this page."
        if bridge
        else "Offline saved examples",
        flush=True,
    )
    print("Local interface • Ctrl+C stops only this local server", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
