"""
ISREL — relevance filter, powered by TypeSafe AI's Jev model.

Uses Jev's Noul primitive: a calibrated probability (0-1) that a
statement is true, instead of an LLM's self-reported "RELEVANT" /
"IRRELEVANT" text guess. This gives a real number to threshold on,
and Jev is purpose-trained to be calibrated on exactly this kind of
judgment — which is the whole reason to prefer it over a general
prompted LLM here.

Public function signatures are unchanged from the original prompting
version, so nothing in pipeline/self_rag.py needs to change.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_client import call_jev

# Tune this against a small set of examples you've labeled yourself —
# see the README section on calibrating thresholds.
RELEVANCE_THRESHOLD = 0.5


def _ask_relevance(query: str, chunk: str) -> float:
    """Sends one Noul question to Jev and returns the raw probability."""
    state = {"question": query, "passage": chunk}
    questions = {
        "is_relevant": {
            "type": "noul",
            "instructions": (
                "The passage contains information that helps answer the question."
            ),
        }
    }
    answers = call_jev(state, questions)
    result = answers.get("is_relevant", {})
    # Fail open (treat as relevant) if Jev is unreachable, rather than
    # silently dropping content the pipeline can't recover.
    return result.get("noul", 1.0)


def relevance_score(query: str, chunk: str) -> float:
    """Raw Jev probability (0-1) — useful for logging and eval, not just True/False."""
    return _ask_relevance(query, chunk)


def check_relevance(query: str, chunk: str) -> bool:
    """Returns True if Jev's relevance probability clears RELEVANCE_THRESHOLD."""
    return _ask_relevance(query, chunk) >= RELEVANCE_THRESHOLD


def filter_relevant_chunks(query: str, chunks: list[dict]) -> list[dict]:
    """
    Takes the retriever's output and returns only chunks Jev judges
    relevant. Each kept chunk gets an added "isrel_score" field so you
    can see the calibrated probability later (in the demo or eval CSV).
    """
    kept = []
    for chunk in chunks:
        prob = relevance_score(query, chunk["text"])
        is_relevant = prob >= RELEVANCE_THRESHOLD
        status = "KEPT" if is_relevant else "DROPPED"
        print(f"[ISREL] {status} (jev_prob={prob:.2f}, retrieval_score={chunk['score']:.3f}) "
              f"— {chunk['text'][:60]}...")
        if is_relevant:
            enriched = dict(chunk)
            enriched["isrel_score"] = prob
            kept.append(enriched)
    return kept
