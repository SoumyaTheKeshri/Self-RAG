"""
Loads the FAISS index built by build_index.py and exposes retrieve().
"""

import os
import sys
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

_model = None
_index = None
_chunks = None


def _load():
    global _model, _index, _chunks
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL)
    if _index is None:
        index_path = os.path.join(config.INDEX_DIR, "index.faiss")
        chunks_path = os.path.join(config.INDEX_DIR, "chunks.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                "No index found. Run `python retrieval/build_index.py` first."
            )
        _index = faiss.read_index(index_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            _chunks = json.load(f)


def retrieve(query: str, top_k: int = None) -> list[dict]:
    """
    Returns a list of {"text": ..., "source": ..., "score": ...} dicts,
    ordered by descending similarity to the query.
    """
    _load()
    top_k = top_k or config.TOP_K

    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(query_vec)

    scores, indices = _index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = _chunks[idx]
        results.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "score": float(score),
        })
    return results
