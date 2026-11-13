# Development environment

## Verified preparation on 13 November 2026

- The generic Python 3.12 environment is in the sibling `hackathon-tools` folder.
- VS Code's Python and Jupyter extensions are installed. The prepared workspace
  file is `../hackathon-tools/Junction_2026.code-workspace`.
- The Jupyter kernel is named **Junction 2026 Python 3.12**.
- A fresh-kernel notebook passed dataframe, SQL, Parquet, and plotting checks.
- A separate generic 20-epoch model check logged all variables agreed with Petteri
  before implementation. It is outside the competition repository.
- TensorBoard served the saved event files locally and through an authenticated
  Hugging Face CPU Job. The cloud viewer job was stopped after the check.
- A non-training Hugging Face CPU Job saved a result to private Hub storage;
  the result was downloaded and verified locally.
- GitHub CLI is authenticated, the repository can be pushed, and this folder is
  registered as a local Codex project.
- Screen and microphone recording were confirmed by Petteri.

## Local commands

The project now has its own environment. From the repository, run
`uv sync --locked`, then open the folder in VS Code with `code .`.
Create a notebook through **Create: New Jupyter Notebook** in the Command Palette.
In **Select Kernel**, choose **Python Environments** and the repository's
`.venv/bin/python`. Change an existing notebook's kernel explicitly if it still
points to the generic preparation environment. Run cells with Shift+Enter.

Check `import sys; print(sys.executable)` in a notebook: it should point to this
repository's `.venv/bin/python`. Explore data in notebooks, then move reusable
logic into Python modules. No browser-based Jupyter server is needed for this
local VS Code workflow.

Launch JupyterLab from a terminal with `junction-lab`. It keeps authentication
enabled and displays the local login URL in the terminal.

Use `junction-tensorboard` to view the saved generic check logs at
http://127.0.0.1:6006/. For project logs after coding begins, use
`junction-tensorboard --logdir "$PWD/runs"` from the repository.

`junction-lab` uses the generic preparation interpreter. For project dependencies,
use the repository's `.venv/bin/python`. Add packages with `uv add PACKAGE` (or
`uv add --dev PACKAGE` for development tools), and commit both dependency files.

## Cloud observation

Before implementing each training run, ask Petteri which TensorBoard variables
and visualisations to log and wait for his answer. Do not silently reuse the
generic check's selection.

Run TensorBoard on port 6006 alongside training and expose that port with the
token-protected Hugging Face Jobs option. Use `junction-hf-tensorboard JOB_ID`
locally to open an authenticated loopback proxy at http://127.0.0.1:6007/.
The token stays in the local credential cache, not in the browser URL or Git.

Save event files periodically to permitted durable storage and again at the
end. Stop the Job after training, then view downloaded logs locally.

## Budget and event boundary

The approved limit is USD 10 total for preparation and the hackathon, including
all readiness checks. Use explicit timeouts, one attempt, and a cumulative cost
record. The readiness CPU checks together cost less than USD 0.01 by the listed
hardware and port rates; final billing may round differently.

No competition source code is prepared in advance. Coding begins at 19:30
Europe/Helsinki on Friday 13 November 2026.
