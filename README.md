# Junction_2026
Hackathon project, Junction 2026

Solo ML and data analysis project. The challenge, project name, method, and results
will be added after reviewing the challenge brief. The repository includes the
local notebook environment, project guidance, and submission outlines.

## Preparation

- [Project instructions](AGENTS.md)
- [Challenge selection](docs/challenge-selection.md)
- [Experiment log](docs/experiments.md)
- [Submission worksheet](docs/submission.md)

Generic preparation checks are kept outside this repository. Project dependencies
are recorded in `pyproject.toml` and `uv.lock`.
Every training run requires TensorBoard. Before implementing each run, ask Petteri
which variables to log and wait for his answer.

## Implementation and reproduction

Run `uv sync --locked` from the repository to install Python 3.12 dependencies.
Open this folder in VS Code with `code .`, create or open a `.ipynb` notebook,
and select the Python environment at `.venv/bin/python` in the notebook's kernel
picker. If an older interpreter is already selected, explicitly change the kernel.

Use notebooks for data exploration and small checks. Move reusable preparation,
evaluation, and training code into Python modules as the challenge develops.
Keep notebooks executable from a restarted kernel. Raw data goes in the ignored
`data/` directory; generated models and results go in `artifacts/` or `runs/`.

For TensorBoard, run `uv run tensorboard --logdir runs --host 127.0.0.1 --port 6006`
and open http://127.0.0.1:6006/. Agree logging variables before implementing any
training run. Heavier training will run on Hugging Face Jobs within the approved
USD 10 total budget.

## Submission targets

Preliminary draft: Friday 23:45. Final submission: Sunday 15 November 09:30,
ahead of the hard 10:00 cutoff. The submission must be marked FINAL.
