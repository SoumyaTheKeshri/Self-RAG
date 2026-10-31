# Self-RAG Style Project

A Self-RAG-style pipeline: adaptive relevance filtering (ISREL),
claim-support checking (ISSUP), and candidate answer selection (ISUSE)
— layered on top of a plain baseline RAG pipeline for comparison.

**Two different AI systems are used, for two different jobs:**
- **Generating answers** (baseline + candidate generation) uses a normal
  text-generation LLM (OpenAI or Groq) via `llm.py`.
- **Critique** (ISREL / ISSUP / ISUSE) uses **TypeSafe AI's Jev** — a
  "System 1" decision model that returns calibrated probabilities for
  typed questions (Choice / Score / Noul) instead of generated text.
  This replaces an earlier version of this project that used prompted
  LLM calls for critique; Jev gives genuinely calibrated confidence
  instead of an LLM guessing a label in free text.

## Folder structure

```
self_rag_project/
├── corpus/                # put your .txt source documents here
├── retrieval/
│   ├── build_index.py      # chunk + embed + build FAISS index
│   └── retriever.py        # load index, retrieve top-k chunks
├── critique/
│   ├── isrel.py            # relevance filter — Jev Noul primitive
│   ├── issup.py            # support/hallucination check — Jev Choice primitive
│   └── isuse.py            # candidate answer scoring — Jev Score primitive
├── pipeline/
│   ├── baseline_rag.py     # plain retrieve -> generate
│   └── self_rag.py         # full self-rag-style pipeline
├── eval/
│   ├── test_questions.json # your Q&A ground-truth set
│   └── run_eval.py         # runs both pipelines, produces comparison table
├── llm.py                  # LLM call wrapper for answer generation (OpenAI/Groq)
├── jev_client.py            # Jev API call wrapper for critique (TypeSafe AI)
├── config.py                # model names, k, thresholds
├── app.py                   # optional Streamlit demo UI
├── requirements.txt
└── .env.example
```

## Setup

1. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and add **both** API keys:

```bash
cp .env.example .env
```

Then edit `.env`:
- `OPENAI_API_KEY` — your OpenAI key, or a Groq key if using Groq's free
  tier (see the base_url note in `llm.py` if switching to Groq)
- `TYPESAFE_API_KEY` — your Jev API key from the TypeSafe dashboard

**Note on cost:** Jev is paid, no free tier, but very cheap — $0.042 per
million input tokens, output tokens free. A full run of the eval script
on a small question set will typically cost well under $1.

3. Add your source documents (plain `.txt` files) into the `corpus/` folder.
   A couple of sample files are already there — replace them with your own.

## Usage

### Step 1 — Build the retrieval index

```bash
python retrieval/build_index.py
```

This chunks every `.txt` file in `corpus/`, embeds the chunks locally
with `sentence-transformers` (no API key needed for this step — this
part is entirely free and offline), and saves a FAISS index to
`retrieval/index/`.

### Step 2 — Try a single query on both pipelines

```bash
python -c "
from pipeline.baseline_rag import run_baseline
from pipeline.self_rag import run_self_rag

query = 'Your question here'
print('--- BASELINE ---')
print(run_baseline(query))
print('--- SELF-RAG STYLE ---')
print(run_self_rag(query))
"
```

Watch the console output during the self-rag run — you'll see
`[ISREL]`, `[ISUSE]`, and `[ISSUP]` log lines showing Jev's calibrated
probabilities/scores as they happen.

### Step 3 — Run the full evaluation

Fill in `eval/test_questions.json` with your own 10-15 questions and
their known correct answers, then run:

```bash
python eval/run_eval.py
```

This produces a comparison table (printed + saved as `eval/results.csv`)
showing, for baseline vs. self-rag-style pipelines: whether the answer
matched ground truth, and how many claims were flagged as unsupported
by the ISSUP checker.

### Optional — Streamlit demo

```bash
streamlit run app.py
```

Lets you type a query and see both pipelines' answers side by side,
including which chunks were kept/rejected by ISREL, ISUSE's calibrated
score + confidence per candidate, and which claims were flagged by ISSUP.

## Calibrating the ISREL threshold

`critique/isrel.py` has a `RELEVANCE_THRESHOLD = 0.5` constant. Jev's
Noul returns a real probability, so this is worth tuning rather than
leaving at the default: run a handful of known relevant/irrelevant
chunks through `relevance_score()`, see where Jev's probabilities
actually land for your corpus, and adjust the threshold accordingly.
This is also a legitimate thing to report in your write-up — "we
calibrated the threshold against N manually labeled examples" is a
real methodological step, not just a config value.

## Notes

- Answer **generation** (baseline answers, and the N candidates in the
  self-rag pipeline) still goes through a normal LLM via `llm.py` —
  Jev is a decision model, not a text generator, so it's used only for
  the three critique steps.
- This is a **lightweight approximation** of the Self-RAG paper
  (Asai et al., 2023), not a retrained model — see the paper's
  original approach (a trained critic model + fine-tuned reflection
  tokens) for what a full reproduction would involve.
- Swap `llm.py`'s `call_llm()` function if you want a different answer
  generation provider (Anthropic, local model via Ollama, etc.) —
  every other file only calls that one function.
- Swap `jev_client.py`'s `call_jev()` function if TypeSafe changes their
  API shape — every critique file only calls that one function.
