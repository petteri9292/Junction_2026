"""Fixture integrity, not pipeline quality or historical-claim evaluation."""

import json
from pathlib import Path
from urllib.parse import urlsplit, parse_qs

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "wikilens/recording_ui/fixtures/examples.json").read_text())


def test_recorded_provenance_and_citation_integrity():
    assert DATA["schema_version"] == 2
    assert DATA["source_languages"] == ["fi"]
    assert len(DATA["examples"]) == 2
    assert {f["search_focus"] for f in DATA["examples"]} == {"", "videopeli"}
    for fixture in DATA["examples"]:
        assert fixture["status"] == "recorded"
        assert fixture["verification"]["pipeline_verified"] is True
        assert fixture["verification"]["claim_support_reviewed"] is False
        assert fixture["captured_at"]
        assert fixture["timings"]["total_seconds"] > 0
        assert fixture["provenance"]["backend"] == "vLLM"
        evidence = {e["id"]: e for e in fixture["evidence"]}
        assert len(evidence) == len(fixture["evidence"]) >= 1
        assert {e["lang"] for e in evidence.values()} == {"fi"}
        for claim in fixture["answer"]:
            assert claim["text"]
            assert set(claim["citations"]) <= evidence.keys()
        for source in evidence.values():
            assert source["passage"] and source["title"] and source["page_id"]
            assert source["passage"] in source["original_passage"]
            url = urlsplit(source["revision_url"])
            assert url.scheme == "https"
            assert url.hostname == f"{source['lang']}.wikipedia.org"
            assert parse_qs(url.query)["oldid"] == [str(source["rev_id"])]
            assert fixture["citation_mappings"][source["id"]] == source["revision_url"]


def test_local_server_needs_only_standard_library():
    import subprocess
    import sys

    # -S hides site packages. Running --help proves no DB/model/Flask import.
    result = subprocess.run(
        [sys.executable, "-S", "-m", "wikilens.recording", "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--port" in result.stdout
