"""
Self-RAG-style pipeline:

  1. Retrieve chunks
  2. ISREL — filter out irrelevant chunks
  3. Generate N candidate answers (temperature sampling) from the
     filtered context
  4. ISSUP — check each candidate's claims against the sources
  5. ISUSE — score all candidates, pick the best one

This is the integration point between Person A's retrieval work and
Person B's critique functions — everything below just calls into
retrieval/retriever.py and critique/*.py.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retriever import retrieve
from critique.isrel import filter_relevant_chunks
from critique.issup import check_answer_support, hallucination_rate
from critique.isuse import select_best_candidate
from llm import call_llm
import config

ANSWER_PROMPT = """Answer the question using only the information in the \
context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""


def generate_candidates(query: str, context: str, n: int) -> list[str]:
    """Generate n candidate answers via temperature sampling."""
    prompt = ANSWER_PROMPT.format(context=context, query=query)
    candidates = []
    for _ in range(n):
        answer = call_llm(prompt, temperature=config.CANDIDATE_TEMPERATURE, max_tokens=400)
        candidates.append(answer)
    return candidates


def run_self_rag(query: str) -> dict:
    """
    Returns a dict with the final answer plus the full reasoning trace:
    which chunks were kept, all candidate scores, and per-claim support
    labels for the winning answer. Print/display all of this in a demo —
    it's the part that makes the project look rigorous.
    """
    # Step 1: retrieve
    raw_chunks = retrieve(query)

    # Step 2: ISREL filter
    relevant_chunks = filter_relevant_chunks(query, raw_chunks)
    if not relevant_chunks:
        # Fall back to the raw top chunk so we still have something to work with
        relevant_chunks = raw_chunks[:1]

    context = "\n\n".join(c["text"] for c in relevant_chunks)

    # Step 3: generate multiple candidates
    candidates = generate_candidates(query, context, config.NUM_CANDIDATES)

    # Step 4: ISUSE selects the best candidate
    selection = select_best_candidate(query, candidates)
    best_answer = selection["best_answer"]

    # Step 5: ISSUP checks the winning answer's claims against sources
    support_trace = check_answer_support(best_answer, relevant_chunks)
    h_rate = hallucination_rate(support_trace)

    return {
        "answer": best_answer,
        "chunks_used": relevant_chunks,
        "chunks_dropped": len(raw_chunks) - len(relevant_chunks),
        "candidate_scores": selection["all_scores"],
        "support_trace": support_trace,
        "hallucination_rate": h_rate,
    }


if __name__ == "__main__":
    q = input("Question: ")
    result = run_self_rag(q)
    print("\n--- Final Answer ---")
    print(result["answer"])
    print(f"\nHallucination rate: {result['hallucination_rate']:.0%}")
    print(f"Chunks dropped by ISREL: {result['chunks_dropped']}")
