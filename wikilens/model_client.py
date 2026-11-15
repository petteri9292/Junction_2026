"""Ollama locally or through a private Hugging Face Jobs endpoint."""

import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import load_dotenv
from huggingface_hub import get_token


def model_request(path, body):
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    endpoint = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlsplit(endpoint)
    headers = {}
    if re.fullmatch(r"[a-f0-9]{24}--11434\.hf\.jobs", parsed.hostname or ""):
        if parsed.scheme != "https":
            raise ValueError("Hugging Face Jobs requires HTTPS")
        token = get_token()
        if not token:
            raise ValueError(
                "Sign in with hf auth login to access the private model job"
            )
        headers["Authorization"] = f"Bearer {token}"
    response = requests.post(endpoint + path, headers=headers, json=body, timeout=180)
    response.raise_for_status()
    return response.json()
