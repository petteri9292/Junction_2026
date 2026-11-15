"""Execute the paragraph/Finnish examples on the existing project GPU kernel."""

import copy, importlib.util, json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/paragraph-citations" / stamp
    directory.mkdir(parents=True)
    n = json.loads((ROOT / "notebooks/08_paragraph_citations_fi_en.ipynb").read_text())
    n["cells"] = [c for c in n["cells"] if c.get("id") != "view"]
    cells = []
    for c in n["cells"]:
        cells.append(c)
        if c.get("id") in ("english", "finnish"):
            cells.append(
                {
                    "cell_type": "code",
                    "id": "save-" + c["id"],
                    "metadata": {},
                    "execution_count": None,
                    "outputs": [],
                    "source": [
                        "Path("
                        + repr(str(directory / "examples.json"))
                        + ").write_text(json.dumps(examples,ensure_ascii=False,indent=2))\n"
                    ],
                }
            )
    n["cells"] = cells
    spec = importlib.util.spec_from_file_location(
        "runner", ROOT / "scripts/compare-postprocessing.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    timings, wall = m.execute(n, directory / "executed.ipynb")
    examples = json.loads((directory / "examples.json").read_text())
    report = {
        "created_at": stamp,
        "examples": examples,
        "cell_seconds": timings,
        "wall_seconds": wall,
        "support_review": "not_reviewed",
    }
    path = ROOT / f"reports/paragraph-citations-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (ROOT / "reports/paragraph-citations-latest.json").write_text(path.read_text())
    import sys

    sys.path.insert(0, str(ROOT))
    from wikilens.paragraph_citations import publish

    publish(report, path.with_suffix(".html"))
    for name, item in examples.items():
        r = item["result"]
        print(name, r["status"], r["seconds"], r["mechanical_metrics"], flush=True)
    print("REPORT:", path, flush=True)


if __name__ == "__main__":
    main()
