"""Small local demo over the same MariaDB functions used by the notebooks."""

import pymysql
import requests
from flask import Flask, jsonify, render_template, request

from .database import compare_topic, related_articles, search_articles


def create_app():
    app = Flask(__name__)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/evidence")
    def evidence_workspace():
        return render_template("evidence.html")

    @app.get("/api/research")
    def research():
        from .research import cached_vector, edition_dossier, retrieve_multilingual

        question = request.args.get("q", "")
        result = retrieve_multilingual(question, {"direct": cached_vector(question)})
        result["dossiers"] = [
            edition_dossier(result, topic) for topic in result["aligned_topics"]
        ]
        result.pop("articles")
        return jsonify(result)

    @app.get("/api/runs/<report_hash>")
    def inspected_run(report_hash):
        from .provenance import inspect_run

        result = inspect_run(report_hash)
        if result is None:
            return jsonify(error="Reviewed run not found"), 404
        return jsonify(result)

    @app.get("/api/search")
    def search():
        return jsonify(
            search_articles(
                request.args.get("q", ""),
                mode=request.args.get("mode", "keyword"),
                lang=request.args.get("lang") or None,
            )
        )

    @app.get("/api/topic/<wikidata_id>")
    def topic(wikidata_id):
        return jsonify(compare_topic(wikidata_id))

    @app.get("/api/related/<lang>/<int:page_id>")
    def related(lang, page_id):
        return jsonify(related_articles(lang, page_id, limit=4))

    @app.post("/api/ask")
    def ask():
        from .rag import answer_question

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict) or not isinstance(
            payload.get("question"), str
        ):
            return jsonify(error="Send a JSON question string"), 400
        return jsonify(answer_question(payload["question"]))

    @app.errorhandler(ValueError)
    def invalid_input(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(pymysql.Error)
    def database_error(error):
        app.logger.error("MariaDB request failed: %s", type(error).__name__)
        return jsonify(
            error="MariaDB is unavailable or not loaded. See the setup instructions."
        ), 503

    @app.errorhandler(requests.RequestException)
    def model_error(error):
        return jsonify(
            error="The BGE-M3 embedding service is unavailable. Use keyword search or start Ollama with bge-m3."
        ), 503

    return app
