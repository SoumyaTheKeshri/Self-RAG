"""
Optional demo UI. Run with:

    streamlit run app.py

Type a question, see both pipelines' answers side by side, plus the
self-rag pipeline's reasoning trace (which chunks were dropped, which
claims were flagged as unsupported).
"""

import streamlit as st
from pipeline.baseline_rag import run_baseline
from pipeline.self_rag import run_self_rag

st.set_page_config(page_title="Self-RAG Demo", layout="wide")
st.title("Baseline RAG vs. Self-RAG-style Pipeline")

query = st.text_input("Ask a question about your corpus:")

if st.button("Run") and query:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Baseline RAG")
        with st.spinner("Running baseline..."):
            baseline_result = run_baseline(query)
        st.write(baseline_result["answer"])
        with st.expander("Chunks used"):
            for c in baseline_result["chunks_used"]:
                st.caption(f"({c['source']}, score={c['score']:.3f})")
                st.text(c["text"][:300])

    with col2:
        st.subheader("Self-RAG style (critique powered by Jev)")
        with st.spinner("Running self-rag pipeline (LLM generation + Jev critique calls)..."):
            self_rag_result = run_self_rag(query)
        st.write(self_rag_result["answer"])

        st.metric("Hallucination rate", f"{self_rag_result['hallucination_rate']:.0%}")
        st.metric("Chunks dropped by ISREL", self_rag_result["chunks_dropped"])

        with st.expander("Candidate scores (ISUSE via Jev Score)"):
            for i, c in enumerate(self_rag_result["candidate_scores"]):
                st.caption(f"Candidate {i+1}: {c['score']:.2f}/4  (Jev confidence: {c['confidence']:.2f})")
                st.text(c["answer"][:300])

        with st.expander("Claim support trace (ISSUP via Jev Choice)"):
            for item in self_rag_result["support_trace"]:
                st.caption(item["support"])
                st.text(item["claim"])
