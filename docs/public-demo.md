# Public Modal demo

Public application: https://petteri-hallipelto--wikilens.us-east.modal.run/

The 15 November session uses a separately allocated H200 for six hours. It keeps
MariaDB, BGE-M3, the reranker and Qwen3-4B-Instruct-2507-FP8 resident. The original
private H200 session is separate. The model, generation/review prompts and default
retrieval settings are unchanged: cutoff −4, 60 articles, 25 generation passages,
6000 maximum article tokens. Deep search remains optional and off by default.

The public CPU frontend sends requests to the existing queue and polls results.
It cannot allocate or restart GPUs. The H200 worker closes at its absolute
session deadline; it stops accepting new work 15 minutes earlier. Session startup
is part of the six-hour allocation. The public status API exposes the deadline.
A first visit can cold-start the small CPU frontend, without restarting the H200.

## Private operator access

Run locally, using the ignored job record:

```bash
.venv/bin/python -m wikilens.recording --live --port 8043 --job data/modal-public-demo-job.json
```

Open http://127.0.0.1:8043/. This server binds only to loopback. Admin controls,
retrieved documents and diagnostics are available here. The original local demo
on port 8041 continues to use its original session.

The public HTML excludes the admin button, dialog and script. The HTTP server
serves an explicit static/API allowlist; admin scripts, diagnostics, document
inspection, fixtures, filesystem paths and arbitrary settings are denied. Public
responses strip internal allocation metadata and raw errors. Only request IDs
admitted through the public endpoint can be read publicly. A durable admission
record also limits public work to one running question across frontend restarts.
Public results are accessible to anyone holding their unguessable result URL;
they are shareable demo results, not confidential storage.

## Reproduction and shutdown

Use Python 3.12 and `uv sync --locked`. Review the combined and provider ledgers
before reserving another paid session. The ignored job record must not already
exist. Starting is deliberately a separate one-attempt action:

```bash
.venv/bin/python scripts/paper-live-control.py reserve --profile h200-public-6h --job data/modal-public-demo-job.json
.venv/bin/modal deploy scripts/modal-public-h200.py
.venv/bin/python scripts/paper-live-control.py start --job data/modal-public-demo-job.json
WIKILENS_PUBLIC_ORIGIN=https://petteri-hallipelto--wikilens.us-east.modal.run .venv/bin/modal deploy scripts/modal-public-demo.py
```

Use the exact HTTPS origin returned by Modal for the workspace; update
`WIKILENS_PUBLIC_ORIGIN` and redeploy only the CPU frontend if the URL changes.
A separate frontend deployment does not restart the H200. The web server's
300-second request timeout is separate from the GPU's six-hour timeout. Modal
web functions do not support a retry setting; GPU allocation has retries=0.

To end early:

```bash
.venv/bin/python scripts/paper-live-control.py stop --job data/modal-public-demo-job.json
.venv/bin/modal app stop wikilens-public-demo
```

The public CPU app scales to zero when idle, but remains deployed after the GPU
session ends. Stop that deployment after the presentation to remove the endpoint.
Budget reservations are conservative estimates, not provider-enforced caps or
reconciled charges. No job records, ledgers, credentials, database snapshots or
model artifacts belong in Git.

## Verification

Tests exercise the actual HTTP route allowlist, sanitized responses, same-origin
submission, settings rejection, private result isolation, public result roundtrip,
frontend-restart admission control and generic error handling. Live checks against
the deployed URL verify the same admin/settings restrictions. Literal quotation
validation proves provenance, not that the model's prose is factually correct.
