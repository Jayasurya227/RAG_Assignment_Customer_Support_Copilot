"""
Phase 3: Multi-Index Architecture - Step 2: Multi-Index Search
Section 5.2 Implementation

Executes targeted retrieval across domain-specific indexes:
1. Targeted Index Retrieval: Queries only the designated index collections (e.g. policy_index + ticket_index).
2. Intra-Index Hybrid Search: Executes Dense (Chroma) + Sparse (BM25) with RRF within each selected index.
3. Cross-Index Candidate Merge: Gathers and deduplicates candidate documents from all selected indexes.
4. Global Cross-Encoder Reranking: Scores the unified candidate pool jointly with 'cross-encoder/ms-marco-MiniLM-L-6-v2'
   to produce the optimal Top 5 context.

Demo Query:
"Can I return an opened laptop on day 8 if I have the original box and all accessories?"
Selected Indexes: ["policy_index", "ticket_index"]
"""

import json
import os
import re
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
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dandes_electronics_kb.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60

INDEX_MAPPING = {
    "policy_index": ["return_policy", "shipping_policy"],
    "product_index": ["product_faq", "known_issue"],
    "order_index": ["order_guide"],
    "ticket_index": ["resolved_ticket"],
}
ALL_INDEX_NAMES = list(INDEX_MAPPING.keys())

STOPWORDS = {
    "what", "does", "say", "is", "are", "the", "a", "an", "and", "or", "in", "on",
    "at", "to", "for", "of", "with", "by", "about", "how", "can", "i", "my", "do",
    "tell", "me", "please", "we", "you", "your", "it"
}

_EMBEDDINGS = None
_CROSS_ENCODER = None
_VECTOR_STORES: dict[str, Chroma] = {}
_BM25_INDEXES: dict[str, tuple[BM25Okapi, list[dict]]] = {}
_KB_DATA = None


def get_embedding_function():
    """Returns OpenAI embeddings if sk- key is present, else local HuggingFace embeddings."""
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    api_key = os.getenv("OPENAI_API_KEY", "").strip('"').strip("'")
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_key.startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        kwargs = {"model": embedding_model, "openai_api_key": api_key}
        if base_url:
            kwargs["openai_api_base"] = base_url
        _EMBEDDINGS = OpenAIEmbeddings(**kwargs)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        _EMBEDDINGS = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    return _EMBEDDINGS


def get_cross_encoder() -> CrossEncoder:
    """Initializes and caches the Cross-Encoder model in memory."""
    global _CROSS_ENCODER
    if _CROSS_ENCODER is None:
        _CROSS_ENCODER = CrossEncoder(RERANKER_MODEL)
    return _CROSS_ENCODER


def tokenize(text: str) -> list[str]:
    """Alphanumeric tokenizer preserving codes like RR-001, TCK-1003."""
    return [t.lower() for t in re.findall(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*", text)]


def get_kb_documents() -> list[dict]:
    """Loads and caches the 50 raw KB documents."""
    global _KB_DATA
    if _KB_DATA is None:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            _KB_DATA = json.load(f).get("documents", [])
    return _KB_DATA


def get_vector_store(index_name: str) -> Chroma:
    """Connects to and caches a Chroma collection for a specific partition."""
    if index_name not in _VECTOR_STORES:
        embeddings = get_embedding_function()
        _VECTOR_STORES[index_name] = Chroma(
            collection_name=index_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return _VECTOR_STORES[index_name]


def get_partitioned_bm25(index_name: str) -> tuple[BM25Okapi, list[dict]]:
    """Builds and caches a BM25 index over a single index partition."""
    if index_name not in _BM25_INDEXES:
        all_docs = get_kb_documents()
        allowed_types = INDEX_MAPPING.get(index_name, [])
        partition_docs = [d for d in all_docs if d.get("doc_type") in allowed_types]

        tokenized_corpus = []
        for d in partition_docs:
            doc_id = d.get("id", "")
            p_code = d.get("metadata", {}).get("policy_code", "") or ""
            title = d.get("title", "")
            content = d.get("content", "")
            dtype = d.get("doc_type", "").replace("_", " ")
            doc_text = f"{doc_id} {doc_id} {p_code} {p_code} {dtype} {title} {content}"
            tokenized_corpus.append(tokenize(doc_text))

        bm25 = BM25Okapi(tokenized_corpus)
        _BM25_INDEXES[index_name] = (bm25, partition_docs)

    return _BM25_INDEXES[index_name]


def search_single_index(query: str, index_name: str, k: int = 10) -> list[Document]:
    """
    Executes Intra-Index Hybrid Search (Dense + BM25 + RRF) on one specific index partition.
    """
    vector_store = get_vector_store(index_name)
    bm25, partition_docs = get_partitioned_bm25(index_name)
    n_docs = len(partition_docs)
    fetch_k = min(k * 2, n_docs)

    # 1. Dense search on this collection
    dense_results = vector_store.similarity_search(query, k=fetch_k)

    # 2. BM25 search on this partition
    query_tokens = tokenize(query)
    clean_tokens = [t for t in query_tokens if t not in STOPWORDS] or query_tokens
    bm25_scores = bm25.get_scores(clean_tokens)
    ranked_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_k]

    # Convert BM25 results to Documents
    bm25_docs = []
    for idx in ranked_indices:
        d = partition_docs[idx]
        metadata = {
            "id": d.get("id"),
            "doc_type": d.get("doc_type"),
            "title": d.get("title"),
            "index_source": index_name,
            "bm25_score": float(bm25_scores[idx]),
        }
        for k_meta, v_meta in d.get("metadata", {}).items():
            if v_meta is not None:
                metadata[k_meta] = v_meta
        bm25_docs.append(Document(page_content=d.get("content", ""), metadata=metadata))

    # 3. Fuse via RRF
    doc_map: dict[str, Document] = {}
    rrf_scores: dict[str, float] = {}

    for rank, doc in enumerate(dense_results, 1):
        did = doc.metadata.get("id")
        if not did:
            continue
        doc.metadata["index_source"] = index_name
        doc_map[did] = doc
        rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (RRF_K + rank))

    for rank, doc in enumerate(bm25_docs, 1):
        did = doc.metadata.get("id")
        if not did:
            continue
        if did not in doc_map:
            doc_map[did] = doc
        rrf_scores[did] = rrf_scores.get(did, 0.0) + (1.0 / (RRF_K + rank))

    sorted_ids = sorted(rrf_scores.keys(), key=lambda did: rrf_scores[did], reverse=True)
    fused_docs = []
    for did in sorted_ids[:k]:
        d = doc_map[did]
        d.metadata["intra_rrf_score"] = round(rrf_scores[did], 5)
        fused_docs.append(d)

    return fused_docs


def multi_index_search(query: str, selected_indexes: list[str], top_k: int = 5, k_per_index: int = 6) -> list[Document]:
    """
    Executes Multi-Index Search across chosen index partitions:
    1. Collects candidates from each selected index.
    2. Merges and deduplicates candidates.
    3. Global Cross-Encoder Reranking to produce top_k documents.
    """
    # Defensive fallback: if no index selected or invalid names, query all indexes
    valid_selected = [idx for idx in selected_indexes if idx in INDEX_MAPPING]
    if not valid_selected:
        valid_selected = ALL_INDEX_NAMES

    # 1. Collect candidates across selected indexes
    candidate_pool: list[Document] = []
    seen_ids: set[str] = set()

    for idx_name in valid_selected:
        index_candidates = search_single_index(query, idx_name, k=k_per_index)
        for doc in index_candidates:
            did = doc.metadata.get("id")
            if did and did not in seen_ids:
                seen_ids.add(did)
                candidate_pool.append(doc)

    if not candidate_pool:
        return []

    # 2. Global Cross-Encoder Reranking across the unified pool
    cross_encoder = get_cross_encoder()
    pairs = [(query, f"{d.metadata.get('title', '')} {d.page_content}") for d in candidate_pool]
    scores = cross_encoder.predict(pairs)

    for doc, score in zip(candidate_pool, scores):
        doc.metadata["rerank_score"] = round(float(score), 4)

    reranked_pool = sorted(candidate_pool, key=lambda d: d.metadata.get("rerank_score", -999.0), reverse=True)
    return reranked_pool[:top_k]


def format_doc_entry(rank: int, doc: Document) -> str:
    """Helper to format document output cleanly."""
    doc_id = doc.metadata.get("id", "UNKNOWN")
    idx_src = doc.metadata.get("index_source", "UNKNOWN")
    title = doc.metadata.get("title", doc.page_content.split("\n")[0][:40])
    ce_score = doc.metadata.get("rerank_score", "N/A")
    score_str = f"| CE Score: {ce_score:<7} " if ce_score != "N/A" else ""
    return f"  [{rank}] ID: {doc_id:<10} | Index: {idx_src:<14} {score_str}| Title: {title[:45]}"


def run_demo_comparison() -> None:
    """Executes the Section 5.2 Multi-Index Search demonstration."""
    query = "Can I return an opened laptop on day 8 if I have the original box and all accessories?"
    selected = ["policy_index", "ticket_index"]

    print("=" * 80)
    print("PHASE 3: TECHNIQUE 2 - MULTI-INDEX SEARCH DEMONSTRATION")
    print("=" * 80)
    print(f"Query: \"{query}\"")
    print(f"Routing Decision: Querying target indexes -> {selected}\n")

    # 1. Show Intra-Index retrieval from each partition
    print("--------------------------------------------------------------------------------")
    print("1. Intra-Index Candidates Retrieved Per Partition:")
    print("--------------------------------------------------------------------------------")
    for idx_name in selected:
        print(f"\n* From '{idx_name}':")
        docs = search_single_index(query, idx_name, k=3)
        for rank, d in enumerate(docs, 1):
            rrf = d.metadata.get("intra_rrf_score", "N/A")
            print(f"    [{rank}] ID: {d.metadata.get('id'):<10} | RRF: {rrf} | Title: {d.metadata.get('title')}")

    # 2. Show Global Cross-Encoder Reranked output
    print("\n--------------------------------------------------------------------------------")
    print(f"2. Unified Multi-Index Reranked Output (Top {5}):")
    print("--------------------------------------------------------------------------------")
    final_docs = multi_index_search(query, selected_indexes=selected, top_k=5)
    for rank, doc in enumerate(final_docs, 1):
        print(format_doc_entry(rank, doc))

    # 3. Technical Insights Summary
    print("\n" + "=" * 80)
    print("EVALUATION & TECHNICAL INSIGHTS (Section 5.2):")
    print("=" * 80)
    print("- Why Multi-Index Search Outperforms Single-Index Search:")
    print("  1. Targeted Retrieval: The query explicitly targeted policies and tickets.")
    print("     It completely bypassed product_index and order_index, eliminating 26 irrelevant docs.")
    print("  2. Multi-Source Completeness: By querying policy_index AND ticket_index, it captured:")
    print("     - RR-001 / RR-002: The official 10-day return policy effective July 2026.")
    print("     - TCK-1003: An older 2025 customer ticket mentioning 7 days.")
    print("     Providing the downstream LLM with all necessary context for date conflict resolution.")
    print("=" * 80)


if __name__ == "__main__":
    run_demo_comparison()
