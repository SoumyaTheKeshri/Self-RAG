"""
ISSUP — claim support / hallucination check, powered by Jev.

Uses Jev's Choice primitive: a calibrated probability for EACH of
three support levels (not just a single guessed label), so you can
see how close a call was — e.g. fully_supported at 0.51 vs 0.49 is a
very different situation than 0.95 vs 0.02, even though both pick the
same winning label.

Public function signatures are unchanged from the original prompting
version, so nothing in pipeline/self_rag.py or eval/run_eval.py needs
to change.
"""

import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from jev_client import call_jev

SUPPORT_CRITERIA = {
    "fully_supported": "Every part of the claim is directly backed by the sources.",
    "partially_supported": "Some part of the claim is backed by the sources, but not all of it.",
    "not_supported": "The sources do not back up the claim, or contradict it.",
}


def split_into_claims(answer: str) -> list[str]:
    """Naive sentence splitter. Good enough for a first pass."""
    sentences = re.split(r"(?<=[.!?])\s+", answer.strip())
    return [s for s in sentences if s.strip()]


def check_support(claim: str, chunks: list[dict]) -> str:
    """Returns one of: FULLY_SUPPORTED, PARTIALLY_SUPPORTED, NOT_SUPPORTED, UNKNOWN."""
    sources_text = "\n\n".join(c["text"] for c in chunks)
    state = {"claim": claim, "sources": sources_text}
    questions = {
        "support": {
            "type": "choice",
            "instructions": "How well is the claim supported by the sources?",
            "criteria": SUPPORT_CRITERIA,
        }
    }
    answers = call_jev(state, questions)
    choice = answers.get("support", {}).get("choice")
    return choice.upper() if choice else "UNKNOWN"


def check_support_detailed(claim: str, chunks: list[dict]) -> dict:
    """
    Same as check_support but returns the full probability distribution
    and confidence too — useful if you want to report "how close was
    this call" rather than just the winning label in your write-up.
    """
    sources_text = "\n\n".join(c["text"] for c in chunks)
    state = {"claim": claim, "sources": sources_text}
    questions = {
        "support": {
            "type": "choice",
            "instructions": "How well is the claim supported by the sources?",
            "criteria": SUPPORT_CRITERIA,
        }
    }
    answers = call_jev(state, questions)
    result = answers.get("support", {})
    return {
        "label": result.get("choice", "unknown").upper(),
        "probabilities": result.get("probabilities", {}),
        "confidence": result.get("confidence"),
    }


def check_answer_support(answer: str, chunks: list[dict]) -> list[dict]:
    """
    Splits an answer into claims and checks each one. Returns a list of
    {"claim": ..., "support": ...} dicts — your reasoning-trace artifact
    for the demo/write-up.
    """
    claims = split_into_claims(answer)
    results = []
    for claim in claims:
        support = check_support(claim, chunks)
        print(f"[ISSUP] {support} — {claim[:70]}")
        results.append({"claim": claim, "support": support})
    return results


def hallucination_rate(results: list[dict]) -> float:
    """Fraction of claims that were NOT_SUPPORTED. Used in your eval table."""
    if not results:
        return 0.0
    unsupported = sum(1 for r in results if r["support"] == "NOT_SUPPORTED")
    return unsupported / len(results)
