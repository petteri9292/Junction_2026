# Junction 2026 project instructions

## Event and scope

- This is Petteri's solo ML and data analysis project for Junction 2026.
- Competition coding starts Friday 13 November 2026 at 19:30 Europe/Helsinki.
  Before that, prepare tools, account connections, and non-code documents only.
- Read the selected challenge brief and judging criteria before proposing a model.
- Keep changes focused on the chosen problem and a demonstrable measured result.

## TensorBoard is required

- Every training run must have TensorBoard observation and saved event files,
  including CPU baselines, fine-tuning, tuning trials, and model-fitting smoke checks.
- Before implementing or changing training code for a run, ask Petteri which
  variables and visualisations to log. Wait for his answer before implementation.
  Do not silently reuse an earlier selection; ask him to confirm reuse.
- Agree tag names, units, step or epoch axis, logging frequency, and validation
  frequency. Suggest relevant options without treating them as selected.
- Use unique run directories, flush events during training, preserve completed
  logs, and verify both live observation and reopening downloaded logs.
- For cloud training, verify authenticated dashboard access before a long run.
  Keep event files in permitted durable storage. Stop compute when finished.

## Environment and evaluation

- Use Python 3.12 and uv. The generic pre-event environment is outside this repo:
  /home/petterihallipelto/Documents/Codex/hackathon-tools/.venv/bin/python
- After coding starts, record project dependencies in pyproject.toml and uv.lock.
  Keep notebooks runnable from a restarted kernel and document reproduction.
- Choose validation splits before tuning. Use chronological or grouped splits
  where appropriate. Fit preprocessing only on training data.
- Record data revisions, code commits, seeds, configurations, metrics, runtime,
  error examples, and limitations. Distinguish measurements from hypotheses.
- Run relevant tests and review changes before merging working milestones.

## Git and cloud compute

- GitHub is the source of truth for code and documentation. Preserve existing work.
- Use short feature branches for implementation; keep main usable for the demo.
- Use Hugging Face Jobs for heavier runs and permitted Hub storage for artifacts.
- Petteri approved up to USD 10 total for preparation and the hackathon on
  13 November 2026. Track cumulative costs, including readiness jobs. Specify
  a timeout and one attempt on every paid job. Ask before exceeding that cap.
- Keep tokens, private contacts, restricted data, environments, and large model
  artifacts outside Git. Never print token contents or copy them into documents.

## Submission

- Maintain docs/submission.md and docs/experiments.md as work progresses.
- Submit the preliminary draft by Friday 23:45 Helsinki time as an internal target.
- Target final submission Sunday 15 November at 09:30; the hard deadline is 10:00.
  Explicitly set the entry to FINAL. Draft entries are not reviewed.
- Include all required fields, a PDF pitch deck, accessible demo/source links,
  and an accessible pitch video no longer than 120 seconds. Aim for 115 seconds.
- Attend Sunday live peer review from 10:30 to 14:00 for main-prize eligibility.
