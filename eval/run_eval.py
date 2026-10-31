"""
Runs every question in test_questions.json through BOTH pipelines,
grades both outputs with the same ISSUP-based checker (used here as an
external grader, not a live filter — see README for why this is fair),
and checks each answer against your ground truth string.

Produces a printed table and saves eval/results.csv.

Usage:
    python eval/run_eval.py
"""

import os
import sys
import json
import csv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.baseline_rag import run_baseline
from pipeline.self_rag import run_self_rag
from critique.issup import check_answer_support, hallucination_rate


def load_test_questions(path: str = None) -> list[dict]:
    path = path or os.path.join(os.path.dirname(__file__), "test_questions.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def matches_ground_truth(answer: str, expected_substring: str) -> bool:
    """Simple, honest check: does the answer contain the expected phrase?
    Good enough for a first pass — feel free to grade manually instead
    for a more accurate accuracy number."""
    return expected_substring.lower() in answer.lower()


def run_evaluation():
    questions = load_test_questions()
    rows = []

    for i, item in enumerate(questions):
        query = item["question"]
        expected = item.get("expected_answer_contains", "")
        print(f"\n===== Question {i+1}/{len(questions)}: {query} =====")

        # --- Baseline ---
        print("\n-- Baseline RAG --")
        baseline_result = run_baseline(query)
        baseline_answer = baseline_result["answer"]
        baseline_support = check_answer_support(baseline_answer, baseline_result["chunks_used"])
        baseline_h_rate = hallucination_rate(baseline_support)
        baseline_correct = matches_ground_truth(baseline_answer, expected)

        # --- Self-RAG style ---
        print("\n-- Self-RAG style --")
        self_rag_result = run_self_rag(query)
        self_rag_answer = self_rag_result["answer"]
        self_rag_h_rate = self_rag_result["hallucination_rate"]
        self_rag_correct = matches_ground_truth(self_rag_answer, expected)

        rows.append({
            "question": query,
            "baseline_answer": baseline_answer,
            "baseline_correct": baseline_correct,
            "baseline_hallucination_rate": round(baseline_h_rate, 2),
            "self_rag_answer": self_rag_answer,
            "self_rag_correct": self_rag_correct,
            "self_rag_hallucination_rate": round(self_rag_h_rate, 2),
        })

    # --- Print summary table ---
    print("\n\n===== SUMMARY =====")
    print(f"{'Question':<40} {'Baseline OK':<12} {'Base Halluc%':<14} {'SelfRAG OK':<12} {'SelfRAG Halluc%':<16}")
    for r in rows:
        print(f"{r['question'][:38]:<40} "
              f"{str(r['baseline_correct']):<12} "
              f"{r['baseline_hallucination_rate']*100:<14.0f} "
              f"{str(r['self_rag_correct']):<12} "
              f"{r['self_rag_hallucination_rate']*100:<16.0f}")

    baseline_accuracy = sum(r["baseline_correct"] for r in rows) / len(rows)
    self_rag_accuracy = sum(r["self_rag_correct"] for r in rows) / len(rows)
    avg_baseline_halluc = sum(r["baseline_hallucination_rate"] for r in rows) / len(rows)
    avg_self_rag_halluc = sum(r["self_rag_hallucination_rate"] for r in rows) / len(rows)

    print(f"\nOverall baseline accuracy: {baseline_accuracy:.0%}")
    print(f"Overall self-rag accuracy: {self_rag_accuracy:.0%}")
    print(f"Overall baseline hallucination rate: {avg_baseline_halluc:.0%}")
    print(f"Overall self-rag hallucination rate: {avg_self_rag_halluc:.0%}")

    # --- Save CSV ---
    out_path = os.path.join(os.path.dirname(__file__), "results.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    run_evaluation()
