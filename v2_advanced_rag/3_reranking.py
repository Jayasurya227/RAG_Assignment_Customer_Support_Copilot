"""
V2 Advanced RAG - Technique 3: Cross-Encoder Reranking
Section 4.3 Implementation

Two-Stage Retrieval Architecture:
Stage 1 (Broad Candidate Retrieval): Retrieve Top N=20 candidates using Hybrid Search (Dense + BM25 + RRF).
Stage 2 (Cross-Encoder Reranking): Jointly score each (query, document) pair using a deep Cross-Encoder
model ('cross-encoder/ms-marco-MiniLM-L-6-v2') and output the refined Top K=5.

Demo Query (from Section 4.3):
"A customer received a damaged phone and wants a refund instead of replacement. What are their options and what does policy say?"

Comparison:
- BEFORE (Initial Retrieval without Reranking): Bi-encoders and BM25 retrieve mixed candidates
  based on broad keyword matches ('phone', 'replacement', 'damaged'), sometimes favoring
  generic replacement workflows.
- AFTER (Cross-Encoder Reranking): The cross-attention mechanism evaluates the exact semantic
  condition ('refund instead of replacement for damaged goods'), promoting RR-003 (the damaged
  goods policy explicitly offering refunds) and TCK-1002 (resolved cracked screen case) to the top.
"""

import importlib.util
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Suppress framework warnings & prevent tensorflow protobuf conflict
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

from dotenv import load_dotenv
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CANDIDATE_POOL_SIZE = 20
FINAL_TOP_K = 5

_CROSS_ENCODER = None
_HYBRID_SEARCH_FN = None


def get_hybrid_search():
    """Dynamically loads the hybrid_search function from 2_hybrid_search.py."""
    global _HYBRID_SEARCH_FN
    if _HYBRID_SEARCH_FN is None:
        hybrid_path = BASE_DIR / "v2_advanced_rag" / "2_hybrid_search.py"
        spec = importlib.util.spec_from_file_location("v2_hybrid_module", hybrid_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _HYBRID_SEARCH_FN = mod.hybrid_search
    return _HYBRID_SEARCH_FN


def get_cross_encoder() -> CrossEncoder:
    """Initializes and caches the Cross-Encoder model in memory."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL_NAME)
    return _CROSS_ENCODER


def rerank_documents(query: str, docs: list[Document], top_k: int = FINAL_TOP_K) -> list[Document]:
    """
    Reranks a list of candidate documents against the query using a Cross-Encoder.
    Returns the top_k documents sorted by descending cross-encoder score.
    """
    if not docs:
        return []

    cross_encoder = get_cross_encoder()

    # Form (query, document_text) pairs for the cross-encoder
    pairs = [(query, f"{d.metadata.get('title', '')} {d.page_content}") for d in docs]
    scores = cross_encoder.predict(pairs)

    # Attach score to metadata and sort
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = round(float(score), 4)

    reranked_docs = sorted(docs, key=lambda d: d.metadata.get("rerank_score", -999.0), reverse=True)
    return reranked_docs[:top_k]


def two_stage_retrieval(query: str, candidate_k: int = CANDIDATE_POOL_SIZE, final_k: int = FINAL_TOP_K) -> tuple[list[Document], list[Document]]:
    """
    Executes complete Two-Stage Retrieval:
    Stage 1: Hybrid Search retrieves candidate_k documents.
    Stage 2: Cross-Encoder reranks them to return final_k documents.
    Returns (stage1_initial_top_k, stage2_reranked_top_k).
    """
    hybrid_search_fn = get_hybrid_search()
    candidates = hybrid_search_fn(query, k=candidate_k)
    stage1_top = candidates[:final_k]
    stage2_top = rerank_documents(query, candidates, top_k=final_k)
    return stage1_top, stage2_top


def format_doc_entry(rank: int, doc: Document, score_label: str = "", score_val: Any = "") -> str:
    """Helper to format document output cleanly."""
    doc_id = doc.metadata.get("id", "UNKNOWN")
    title = doc.metadata.get("title", doc.page_content.split("\n")[0][:40])
    doc_type = doc.metadata.get("doc_type", "None")
    score_str = f"| {score_label}: {score_val:<7} " if score_label else ""
    return f"  [{rank}] ID: {doc_id:<10} | Type: {doc_type:<16} {score_str}| Title: {title[:50]}"


def run_demo_comparison() -> None:
    """Executes the official Section 4.3 demonstration: Before vs After."""
    query = (
        "A customer received a damaged phone and wants a refund instead of replacement. "
        "What are their options and what does policy say?"
    )

    print("=" * 80)
    print("V2 ADVANCED RAG - TECHNIQUE 3: CROSS-ENCODER RERANKING")
    print("=" * 80)
    print(f"Query: \"{query}\"\n")

    # Execute two-stage retrieval
    stage1_docs, stage2_docs = two_stage_retrieval(query, candidate_k=CANDIDATE_POOL_SIZE, final_k=FINAL_TOP_K)

    # 1. BEFORE: Stage 1 Initial Retrieval (Top 5 before reranking)
    print("--------------------------------------------------------------------------------")
    print(f"1. BEFORE: Stage 1 Initial Hybrid Retrieval (Top {FINAL_TOP_K} without reranking)")
    print("--------------------------------------------------------------------------------")
    for rank, doc in enumerate(stage1_docs, 1):
        rrf = doc.metadata.get("rrf_score", "N/A")
        print(format_doc_entry(rank, doc, score_label="RRF Score", score_val=rrf))

    # 2. AFTER: Stage 2 Cross-Encoder Reranked (Top 5)
    print("\n--------------------------------------------------------------------------------")
    print(f"2. AFTER: Stage 2 Cross-Encoder Reranked (Top {FINAL_TOP_K} with {RERANKER_MODEL_NAME})")
    print("--------------------------------------------------------------------------------")
    for rank, doc in enumerate(stage2_docs, 1):
        score = doc.metadata.get("rerank_score", "N/A")
        print(format_doc_entry(rank, doc, score_label="CE Score", score_val=score))

    # 3. Technical Insights Summary
    print("\n" + "=" * 80)
    print("EVALUATION & TECHNICAL INSIGHTS:")
    print("=" * 80)
    print("- Why Bi-Encoder / First-Stage Retrieval Needs Reranking:")
    print("  First-stage retrieval (Dense / BM25) encodes queries and documents separately,")
    print("  making it computationally fast (O(N) vector index lookup) but unable to capture")
    print("  subtle relationships like 'refund INSTEAD OF replacement'.")
    print("- How the Cross-Encoder Solves This:")
    print("  The Cross-Encoder feeds query and document together through all BERT layers,")
    print("  performing cross-attention between every query word and document word.")
    print("  It directly promotes:")
    print("    1. RR-003  (Official policy specifying refund vs replacement options)")
    print("    2. TCK-1002 (Real resolved case where customer with cracked screen was given a refund)")
    print("  Filtering out less relevant noise and ensuring the highest-quality LLM context.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo_comparison()
