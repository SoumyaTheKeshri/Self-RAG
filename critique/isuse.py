"""
ISUSE — candidate answer scoring, powered by Jev's Score primitive.

Score returns a calibrated, probability-weighted float across an
ordered rubric, plus a genuine confidence value — a real upgrade over
asking an LLM to "rate this 1-5" and hoping the digit it returns means
what it says. Confidence lets you flag close calls (e.g. two
candidates scoring 2.4 vs 2.6 with low confidence) rather than
silently picking a "best" answer that was really a coin flip.

Public function signatures are unchanged from the original prompting
version, so nothing in pipeline/self_rag.py needs to change.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_client import call_jev

# Ordered low -> high. Score returns the probability-weighted mean of
# these indices (0 to len-1), not just the single most likely level.
USEFULNESS_LEVELS = [
    "Poor: incomplete, unclear, or does not address the question.",
    "Below average: partially addresses the question but is unclear or missing key points.",
    "Average: addresses the question reasonably but lacks polish or completeness.",
    "Good: clear, complete, and directly addresses the question.",
    "Excellent: thorough, well organized, and directly addresses the question with no gaps.",
]


def score_candidate(query: str, answer: str) -> dict:
    """Returns {"score": float 0-4, "confidence": float}."""
    state = {"question": query, "answer": answer}
    questions = {
        "usefulness": {
            "type": "score",
            "instructions": "How useful and complete is the answer for the question?",
            "criteria": USEFULNESS_LEVELS,
        }
    }
    answers = call_jev(state, questions)
    result = answers.get("usefulness", {})
    return {
        "score": result.get("score", 0.0),
        "confidence": result.get("confidence", 0.0),
    }


def select_best_candidate(query: str, candidates: list[str]) -> dict:
    """
    Scores every candidate and returns the winner. Returns
    {"best_answer": ..., "best_score": ..., "all_scores": [...]} —
    all_scores now also carries each candidate's Jev confidence, so you
    can show in a demo when the "best" pick was a close call.
    """
    scored = []
    for i, candidate in enumerate(candidates):
        result = score_candidate(query, candidate)
        print(f"[ISUSE] Candidate {i+1} scored {result['score']:.2f}/4 "
              f"(confidence={result['confidence']:.2f})")
        scored.append({
            "answer": candidate,
            "score": result["score"],
            "confidence": result["confidence"],
        })

    best = max(scored, key=lambda x: x["score"])
    return {
        "best_answer": best["answer"],
        "best_score": best["score"],
        "all_scores": scored,
    }
