from unittest.mock import patch

from wikilens.web import create_app


def test_search_passes_unicode_query_and_language_to_retrieval():
    with patch("wikilens.web.search_articles", return_value=[]) as search:
        response = (
            create_app()
            .test_client()
            .get("/api/search", query_string={"q": "Suomen itsenäisyys", "lang": "fi"})
        )
    assert response.status_code == 200
    search.assert_called_once_with("Suomen itsenäisyys", mode="keyword", lang="fi")


def test_input_failure_is_reported_without_database_details():
    with patch(
        "wikilens.web.search_articles", side_effect=ValueError("Unknown language")
    ):
        response = create_app().test_client().get("/api/search?q=history&lang=bad")
    assert response.status_code == 400
    assert response.json == {"error": "Unknown language"}
