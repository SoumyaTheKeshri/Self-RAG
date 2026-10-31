"""
Plain baseline RAG: retrieve top-k chunks, generate one answer, done.
No relevance filtering, no self-checking. This is what we compare
the self-rag-style pipeline against.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retrieval.retriever import retrieve
from llm import call_llm

ANSWER_PROMPT = """Answer the question using only the information in the \
context below. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {query}

Answer:"""


def run_baseline(query: str) -> dict:
    """
    Returns {"answer": ..., "chunks_used": [...]} so the eval script
    can run ISSUP/ISREL against this output afterward, externally,
    purely for grading (see README for why this isn't circular).
    """
    chunks = retrieve(query)
    context = "\n\n".join(c["text"] for c in chunks)
    prompt = ANSWER_PROMPT.format(context=context, query=query)
    answer = call_llm(prompt, temperature=0.0, max_tokens=400)

    return {"answer": answer, "chunks_used": chunks}


if __name__ == "__main__":
    q = input("Question: ")
    result = run_baseline(q)
    print("\n--- Answer ---")
    print(result["answer"])
