"""
Central configuration. Tweak these instead of hunting through every file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- LLM settings (used for generating answers — not for critique anymore) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# --- Jev / TypeSafe AI settings (used for ISREL, ISSUP, ISUSE critique) ---
TYPESAFE_API_KEY = os.getenv("TYPESAFE_API_KEY", "")
# Pinned to an exact version rather than "jev-latest" so thresholds you
# tune don't silently drift if TypeSafe moves the alias.
JEV_MODEL = os.getenv("JEV_MODEL", "jev-1.13.0")

# --- Embedding model (runs locally, no API key needed) ---
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval settings ---
CHUNK_SIZE = 500          # characters per chunk (simple char-based chunking)
CHUNK_OVERLAP = 50
TOP_K = 5                 # how many chunks the retriever pulls initially

# --- Self-RAG pipeline settings ---
NUM_CANDIDATES = 3         # how many candidate answers ISUSE chooses between
CANDIDATE_TEMPERATURE = 0.7

# --- Paths ---
CORPUS_DIR = "corpus"
INDEX_DIR = "retrieval/index"
