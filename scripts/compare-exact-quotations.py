"""Rerun post-hoc attribution with deterministic source quotation slicing."""

import copy, hashlib, importlib.util, json, re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = ROOT / "data/citation-comparisons" / stamp
    directory.mkdir(parents=True)
    n = json.loads((ROOT / "notebooks/07_exact_quotation_comparison.ipynb").read_text())
    output = directory / "source-spans-run.json"
    n["cells"].append(
        {
            "cell_type": "code",
            "id": "export",
            "metadata": {},
            "outputs": [],
            "execution_count": None,
            "source": [
                "Path("
                + repr(str(output))
                + ').write_text(json.dumps(results["post_hoc"],ensure_ascii=False,indent=2))\n'
            ],
        }
    )
    runner = module("runner", ROOT / "scripts/compare-postprocessing.py")
    runner.execute(n, directory / "source-spans-executed.ipynb")
    new = json.loads(output.read_text())
    old = json.loads(
        (ROOT / "reports/citation-comparison-20261114T120109Z.json").read_text()
    )
    report = {
        k: copy.deepcopy(old[k])
        for k in ["question", "nodes", "passage_registry_sha256", "review_protocol"]
    }
    report.update(
        created_at=stamp,
        baseline_report="reports/citation-comparison-20261114T120109Z.json",
        variants={
            "previous_post_hoc": old["variants"]["post_hoc"],
            "source_spans": new,
        },
        order=["previous_post_hoc", "source_spans"],
        code_sha256=hashlib.sha256(
            (ROOT / "wikilens/citation_pipelines.py").read_bytes()
        ).hexdigest(),
    )
    for v in report["variants"].values():
        cs = [c for s in v["sentences"] for c in s["citations"]]
        v["mechanical_metrics"]["quotes_supplied"] = sum(
            bool(c.get("quote")) for c in cs
        )
        v["mechanical_metrics"]["strict_exact_quotes"] = sum(
            c.get("quote")
            in next(n["text"] for n in report["nodes"] if n["id"] == c["source_id"])
            for c in cs
        )
        v["mechanical_metrics"]["sentences_with_all_quotes_located"] = sum(
            bool(s["citations"]) and all(c.get("quote_located") for c in s["citations"])
            for s in v["sentences"]
        )
    path = ROOT / f"reports/exact-quotation-comparison-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    module("publisher", ROOT / "scripts/compare-citation-pipelines.py").publish(
        report, path.with_suffix(".html")
    )
    page = path.with_suffix(".html")
    text = page.read_text().replace(
        "WikiLens · citation-aware vs post-hoc", "WikiLens · exact quotation comparison"
    )
    text = re.sub(
        r"(<h1>.*?</h1>)<p>.*?</p>",
        r"\1<p>Previous free-text quotation attribution versus source-span selection. Same frozen seven passages, editor prompt, model revision, generation settings and BGE reference lookup. The new run repeats the complete pipeline; the baseline is the preserved earlier measurement. Qwen now selects span IDs and Python copies the source slices. Quote fidelity is guaranteed for accepted spans; semantic support still needs review.</p>",
        text,
        count=1,
    )
    page.write_text(text)
    (ROOT / "reports/exact-quotation-comparison-latest.json").write_text(
        path.read_text()
    )
    print("REPORT:", path, flush=True)
    print("NEW:", new["seconds"], new["mechanical_metrics"], new["stages"], flush=True)


if __name__ == "__main__":
    main()
