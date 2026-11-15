# H200 six-hour presentation session — 15 November update

The user authorized one H200 session with the same pinned model, prompts,
corpus and vLLM settings. The `h200-6h` profile reserves USD32 and uses a
21,600-second execution timeout, one attempt and one maximum container.
The absolute deadline is **13:18:19 Helsinki on 15 November**; new questions
close at 13:03:19 to allow in-flight work to finish. Startup is included.
This replaces the expired overnight H100 session below.

To reproduce this profile with a new unique job file:

```bash
.venv/bin/python scripts/paper-live-control.py reserve --profile h200-6h --job data/NEW-h200-session.json
.venv/bin/modal deploy scripts/modal-paper-h200.py
.venv/bin/python scripts/paper-live-control.py start --job data/NEW-h200-session.json
```

Inspect existing sessions and budgets before these paid actions.

# Presentation runbook

## Reliable offline fallback

```bash
python3 -m wikilens.recording --port 8040
```

Open http://127.0.0.1:8040/. This uses only the Python standard library. Enter `talvisota`, leave context empty and search. Open a citation. Add context `videopeli` and search again. The page identifies both as saved H100 runs and displays the original measured processing time, not the playback duration. The unedited text may contain mistakes. Admin retrieval inspection requires the live session; citation excerpts work offline, while Wikipedia revision links require internet.

## Existing warm live session

```bash
uv sync --locked
.venv/bin/python scripts/paper-live-control.py status
scripts/record-wikilens.sh --port 8041
```

Open http://127.0.0.1:8041/. Wait for **Connected · Model stays warm**. The active job record is private and lives in `data/modal-paper-live-job.json`. Reloading the page does not restart inference services.

Before presenting, check the session deadline and remaining time. A two-hour session stops accepting questions 15 minutes before its hard deadline. The overnight session used during verification expires at **03:29:56 Europe/Helsinki on 15 November**, so it must not be assumed available for the morning pitch. Start a new bounded session beforehand using the existing prepared snapshot, or present the labelled offline fallback.

## Demonstration sequence

1. Search `talvisota` without context. Explain the real pipeline stages while it runs.
2. Open **Admin → Retrieved documents**. Show article count, raw reranker scores, cutoff decisions and a passage selected for generation. Expand the original article text.
3. Close Admin and click a citation to show an exact source quotation and Wikipedia revision link.
4. Add context `videopeli`, repeat the search and explain how source selection changes.
5. Optionally ask `What was the Winter War?` to demonstrate English output from Finnish sources. HyDE and answer-language detection share one model response.

Admin settings affect the next search only. Defaults: cutoff −3, relative gap blank, top-k 60, passages 25, output tokens 6000. Restore defaults after a tuning demonstration. Increasing the cutoff too far can correctly yield no answer; the rejected retrieval diagnostics remain available.

## Starting another paid session

This is an operator workflow for the existing configured account and durable Volume. A fresh clone needs its own Modal authentication, budget ledgers and prepared Dataset L snapshot; see the data preparation guide. No credentials or private snapshot are bundled in Git.

Inspect existing jobs, provider state and the three ignored budget ledgers before reserving. Do not count promotional credit as a second compute allowance. Use a unique job path; never retry a submission whose provider state is ambiguous.

```bash
.venv/bin/python scripts/paper-live-control.py reserve --job data/presentation-session.json
.venv/bin/modal deploy scripts/modal-paper-live.py
.venv/bin/python scripts/paper-live-control.py start --job data/presentation-session.json
.venv/bin/python scripts/paper-live-control.py status --job data/presentation-session.json
scripts/record-wikilens.sh --port 8041 --job data/presentation-session.json
```

Reserve/start are explicit paid actions. The controller reserves a conservative USD10 for one two-hour H100 session with no application retries. The configured source snapshot comes from the completed full-L import. Deploying a changed app while an earlier session is running can interrupt it; inspect and finish the old session first. A fresh startup can take several minutes.

```bash
.venv/bin/python scripts/paper-live-control.py collect --job data/presentation-session.json
.venv/bin/python scripts/paper-live-control.py stop --job data/presentation-session.json
```

Collect preserves raw outputs, prompts, source identities, settings, diagnostics and timings in ignored `data/paper-live/`. Stop closes compute after the current question. Ctrl+C on the local web server stops only that server. The bounded deadline still applies if a laptop disconnects.

## Pitch materials

Use `reports/wikilens-pipeline-pitch.pptx` or its PDF and `pipeline-pitch-script.txt`. The main story is the MariaDB retrieval–generation pipeline; disambiguation is a short demonstration of it. Use `demo-brief.md` for the video. Avoid claiming factual verification, a guaranteed latency, live English Wikipedia retrieval, model training, or system-versioned time travel.
