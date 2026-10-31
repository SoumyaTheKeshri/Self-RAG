"""
Single wrapper around TypeSafe AI's Jev System One API. Every critique
module calls call_jev() instead of building HTTP requests itself — same
pattern as llm.py for the OpenAI/Groq side, so swapping providers later
only means editing one file.

Jev is not a text-generation model. You send a "state" (the context)
plus one or more typed "questions" (choice / score / noul), and it
returns calibrated, structured answers — no free text to parse.
Docs: https://docs.typesafe.ai/api
"""

import requests
import config

JEV_ENDPOINT = "https://api.typesafe.ai/v1/systemone"


def call_jev(state, questions: dict) -> dict:
    """
    state: str or dict — the context Jev judges (e.g. a query + passage).
    questions: dict matching Jev's schema, e.g.
        {"is_relevant": {"type": "noul", "instructions": "..."}}

    Returns the `answers` dict from Jev's response — e.g.
        {"is_relevant": {"type": "noul", "noul": 0.87}}
    Returns {} on failure so callers can fail safely instead of crashing
    the whole pipeline over one flaky request.
    """
    payload = {
        "state": state,
        "model": config.JEV_MODEL,
        "questions": questions,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.TYPESAFE_API_KEY}",
    }
    try:
        response = requests.post(JEV_ENDPOINT, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json().get("answers", {})
    except requests.exceptions.RequestException as e:
        print(f"[jev_client.py] Jev API call failed: {e}")
        return {}
