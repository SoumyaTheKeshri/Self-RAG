"""
Run this once (and again whenever you add/change documents in corpus/):

    python retrieval/build_index.py

Reads every .txt file in corpus/, splits it into overlapping chunks,
embeds them locally with sentence-transformers, and saves a FAISS
index + the chunk texts to retrieval/index/.
"""

import os
import sys
import json
import glob
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def build_index():
    corpus_dir = config.CORPUS_DIR
    index_dir = config.INDEX_DIR
    os.makedirs(index_dir, exist_ok=True)

    txt_files = glob.glob(os.path.join(corpus_dir, "*.txt"))
    if not txt_files:
        print(f"No .txt files found in {corpus_dir}/. Add some documents first.")
        return

    all_chunks = []
    all_sources = []

    for path in txt_files:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunk_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
        all_chunks.extend(chunks)
        all_sources.extend([os.path.basename(path)] * len(chunks))

    print(f"Loaded {len(txt_files)} file(s), produced {len(all_chunks)} chunks.")

    print(f"Loading embedding model: {config.EMBEDDING_MODEL} ...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    print("Embedding chunks...")
    embeddings = model.encode(all_chunks, show_progress_bar=True, convert_to_numpy=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Normalize for cosine similarity via inner product
    faiss.normalize_L2(embeddings)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(index_dir, "index.faiss"))

    with open(os.path.join(index_dir, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(
            [{"text": c, "source": s} for c, s in zip(all_chunks, all_sources)],
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Index built and saved to {index_dir}/")


if __name__ == "__main__":
    build_index()
