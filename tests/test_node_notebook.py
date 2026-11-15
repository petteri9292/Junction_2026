"""Execute notebook control flow with synthetic inputs and mocked inference only."""

import contextlib
import io
import json
from pathlib import Path


def test_new_notebook_runs_in_order_with_mocked_models_and_preserves_citations(
    tmp_path, monkeypatch
):
    import IPython.display
    import wikilens.node_pipeline as nodes
    import wikilens.hyde_search as search
    import wikilens.torch_runtime as torch_runtime

    calls = []
    retrieval_options = []
    text = (
        "The Winter War took place between Finland and the Soviet Union in 1939 and 1940. "
        * 30
    )
    article = dict(
        lang="fi",
        page_id=1,
        rev_id=2,
        wikidata_id="Q134949",
        title="Talvisota",
        text=text,
        origin="test_only",
        retrieval_rank=1,
    )

    class Runtime:
        def __init__(self, **kwargs):
            self.closed = False

        def embed(self, texts, **kwargs):
            return [[1.0] + [0.0] * 1023 for _ in texts]

        def generate(self, messages, **kwargs):
            calls.append(kwargs)
            data = json.loads(messages[1]["content"])
            if "passages" not in data and "text" not in data:
                return (
                    "A hypothetical historical encyclopedia passage for retrieval only."
                )
            if "query" in data:
                return json.dumps(
                    {
                        "score": 0.8,
                        "excerpts": [data["text"][: min(400, len(data["text"]))]],
                    }
                )
            source = data["passages"][0]
            quote = " ".join(source["text"].split()[:12])
            return json.dumps(
                {
                    "insufficient_evidence": False,
                    "claims": [
                        {
                            "text": "Mocked claim for control-flow testing.",
                            "evidence": [
                                {"source_id": source["source_id"], "quote": quote}
                            ],
                        }
                    ],
                }
            )

        def close(self):
            self.closed = True

        def stats(self):
            return {"mocked": True, "closed": self.closed}

    monkeypatch.setattr(torch_runtime, "TorchRuntime", Runtime)
    monkeypatch.setattr(nodes, "load_retrieved", lambda *a, **kw: [article])

    def retrieve_mock(*a, **kw):
        retrieval_options.append(kw)
        return {
            "articles": [article],
            "seconds": 0.0,
            "status": "completed",
            "search_focus": "history",
        }

    monkeypatch.setattr(search, "retrieve_hyde", retrieve_mock)
    monkeypatch.setattr(IPython.display, "display", lambda *a, **kw: None)
    notebook = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "notebooks/02_postprocessing_generation.ipynb"
        ).read_text()
    )
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            source = "".join(cell["source"])
            if cell["id"] == "setup":
                source = source.replace(
                    "runtime_module = importlib.reload(runtime_module)", ""
                )
            if cell["id"] == "hyde":
                source = source.replace(
                    "search_module = importlib.reload(search_module)", ""
                )
            source = source.replace(
                "READY_FOR_GENERATION = False", "READY_FOR_GENERATION = True"
            )
            if cell["id"] == "settings":
                source += '\nSETTINGS.update(hyde_max_new_tokens=150,search_top_k=7,search_language="fi",rerank_top_k=2,compression_max_new_tokens=250,context_chars=1800,answer_max_new_tokens=1500,strict_generation=True)\n'
            if cell["id"] == "save":
                namespace["project_dir"] = tmp_path
            exec(compile(source, cell["id"], "exec"), namespace)
    record = namespace["node_run"]
    assert retrieval_options == [{"top_k": 7, "language": "fi"}]
    assert calls[0]["max_new_tokens"] == 150
    assert calls[-1]["max_new_tokens"] == 1500
    assert all(c["max_new_tokens"] == 250 for c in calls[1:-1])
    assert record["settings"]["context_chars"] == 1800
    namespace["SETTINGS"]["answer_max_new_tokens"] = 100
    try:
        namespace["check_settings"]()
    except ValueError:
        pass
    else:
        raise AssertionError("Changed settings were silently used")
    namespace["SETTINGS"]["answer_max_new_tokens"] = 1500
    assert record["search_focus"] == "history"
    assert record["stages"]["hyde"]["search_focus"] == "history"
    assert record["stages"]["postprocessing"]["search_focus"] == "history"
    assert record["status"] == "draft" and record["claim_support_reviewed"] is False
    assert record["runtime"] == {"mocked": True, "closed": True}
    assert record["stages"]["generation"]["claims"][0]["evidence"][0][
        "provenance_checked"
    ]
    assert namespace["node_report_path"].is_file()
    assert len(list(tmp_path.glob("reports/*.json"))) == 1
    # A different question cannot reuse the current source context silently.
    namespace["question"] = "A different question"
    try:
        namespace["start_node_stage"]("generation")
    except ValueError:
        pass
    else:
        raise AssertionError("Mixed-question state was accepted")
