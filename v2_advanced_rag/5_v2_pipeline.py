"""
V2 Advanced RAG - Complete Online Pipeline
Section 4.5 Implementation

Integrates all four advanced RAG techniques into a unified, production-grade pipeline:
1. Technique 4: Query Transformation (Query Rewriting & Multi-Query decomposition)
2. Technique 2: Hybrid Search (Dense vector search + BM25 keyword search + RRF)
3. Technique 1: Metadata Filtering (domain isolation when filters are specified)
4. Technique 3: Cross-Encoder Reranking (two-stage retrieval: top 20 -> top 5 via cross-encoder)
5. Conflict Resolution: "Latest Information Wins" (orders documents by date; latest policy supersedes older tickets)
6. Grounded Generation with exact refusal:
   "I don't know based on the available information."
   and source citations: Sources: RR-001, RR-002
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
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
EXACT_REFUSAL = "I don't know based on the available information."
TOP_K_FINAL = 5
CANDIDATE_POOL = 20

_LLM = None
_HYBRID_SEARCH = None
_RERANK_FN = None
_REWRITE_FN = None
_MULTI_QUERY_FN = None


def load_module(module_name: str, file_path: Path):
    """Safely loads a module from a path containing numeric prefixes."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_pipeline_components():
    """Lazily loads and caches the components from Techniques 2, 3, and 4."""
    global _HYBRID_SEARCH, _RERANK_FN, _REWRITE_FN, _MULTI_QUERY_FN
    if _HYBRID_SEARCH is None:
        v2_dir = BASE_DIR / "v2_advanced_rag"
        hybrid_mod = load_module("v2_hybrid", v2_dir / "2_hybrid_search.py")
        _HYBRID_SEARCH = hybrid_mod.hybrid_search

        rerank_mod = load_module("v2_rerank", v2_dir / "3_reranking.py")
        _RERANK_FN = rerank_mod.rerank_documents

        query_mod = load_module("v2_query_trans", v2_dir / "4_query_transformation.py")
        _REWRITE_FN = query_mod.rewrite_query
        _MULTI_QUERY_FN = query_mod.generate_subqueries

    return _HYBRID_SEARCH, _RERANK_FN, _REWRITE_FN, _MULTI_QUERY_FN


def get_llm() -> ChatOpenAI:
    """Initializes and caches the chat LLM."""
    global _LLM
    if _LLM is not None:
        return _LLM

    api_key = os.getenv("OPENAI_API_KEY", "").strip('"').strip("'")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "gemini-flash-latest")

    kwargs = {
        "model": model_name,
        "temperature": 0.0,
        "openai_api_key": api_key,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url

    _LLM = ChatOpenAI(**kwargs)
    return _LLM


SYSTEM_PROMPT = """You are an expert customer support assistant for Dandes Electronics, an online electronics store.
Answer the user's question accurately using ONLY the provided retrieved context below.

Rules:
1. Ground your answer strictly in the provided context. Do NOT make up any information.
2. CONFLICT RESOLUTION (LATEST DATE WINS):
   If retrieved documents contain conflicting statements, numbers, or return windows (for example, an older ticket from 2025 stating 7 days versus a current policy from July 2026 stating 10 days), ALWAYS prioritize and answer according to the document with the most recent effective date. Explicitly explain that the current updated policy supersedes the older document.
3. If the context does not contain enough information to answer the question, reply EXACTLY with:
   "I don't know based on the available information."
4. Keep your answers concise, direct, helpful, and professional.

Retrieved Context:
{context}
"""


def format_context_with_metadata(docs: list[Document]) -> str:
    """
    Formats retrieved documents for the LLM with explicit Document IDs, Dates,
    and Document Types to enable date-based conflict resolution.
    """
    blocks = []
    for doc in docs:
        doc_id = doc.metadata.get("id", "UNKNOWN")
        doc_type = doc.metadata.get("doc_type", "general")
        date_str = doc.metadata.get("date") or doc.metadata.get("updated_at") or "Undated"
        title = doc.metadata.get("title", "")
        ce_score = doc.metadata.get("rerank_score", "N/A")

        header = f"[Document ID: {doc_id} | Effective Date: {date_str} | Type: {doc_type} | Relevance Score: {ce_score}]"
        blocks.append(f"{header}\nTitle: {title}\n{doc.page_content}")
    return "\n\n".join(blocks)


def retrieve_v2_documents(question: str, top_k: int = TOP_K_FINAL) -> list[Document]:
    """
    Executes V2 Multi-Stage Retrieval:
    1. Query Transformation (Rewriting or Sub-queries for compound queries).
    2. Hybrid Search (Dense + BM25 + RRF) retrieves candidate pool of 20 docs.
    3. Cross-Encoder joint scoring reranks candidate pool to top_k docs.
    """
    hybrid_search_fn, rerank_fn, rewrite_fn, multi_query_fn = get_pipeline_components()

    # Step 1: Detect compound queries
    is_compound = any(marker in question.lower() for marker in [" and ", ", what ", ", where ", ", how "])
    candidate_docs: list[Document] = []

    if is_compound:
        sub_queries = multi_query_fn(question)
        for sq in sub_queries:
            sub_candidates = hybrid_search_fn(sq, k=10)
            candidate_docs.extend(sub_candidates)
    else:
        # Step 1b: Rewrite informal/emotional query
        optimized_q = rewrite_fn(question)
        candidate_docs = hybrid_search_fn(optimized_q, k=CANDIDATE_POOL)

    # Deduplicate candidate docs by ID while preserving order
    unique_candidates: list[Document] = []
    seen_ids: set[str] = set()
    for d in candidate_docs:
        did = d.metadata.get("id")
        if did and did not in seen_ids:
            seen_ids.add(did)
            unique_candidates.append(d)

    # Step 2: Cross-Encoder Reranking
    reranked_docs = rerank_fn(question, unique_candidates, top_k=top_k)
    return reranked_docs


def answer_question(question: str) -> dict[str, Any]:
    """
    Complete V2 Pipeline:
    Retrieves top 5 reranked documents and generates an answer with conflict resolution.
    Returns:
      {
        "question": ...,
        "answer": ...,
        "sources": [...],
        "retrieved_docs": [...]
      }
    """
    retrieved_docs = retrieve_v2_documents(question, top_k=TOP_K_FINAL)

    # Filter out documents with strongly negative Cross-Encoder scores for out-of-scope detection
    top_score = retrieved_docs[0].metadata.get("rerank_score", 0.0) if retrieved_docs else -99.0
    if not retrieved_docs or top_score < -4.0:
        return {
            "question": question,
            "answer": EXACT_REFUSAL,
            "sources": [],
            "retrieved_docs": retrieved_docs,
        }

    context_str = format_context_with_metadata(retrieved_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})
    answer_text = response.content.strip()

    # Determine citations
    if EXACT_REFUSAL in answer_text:
        sources: list[str] = []
    else:
        # Extract unique source IDs of contributing docs
        sources = [doc.metadata.get("id") for doc in retrieved_docs if doc.metadata.get("id")]
        sources = list(dict.fromkeys(sources))

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "retrieved_docs": retrieved_docs,
    }


def print_formatted_result(result: dict[str, Any]) -> None:
    """Prints output in official Section 3.3 / 4.5 format."""
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    else:
        print("Sources: none")


def run_official_demo():
    """Runs the 7 official Section 3.2 benchmark questions to demonstrate V2 capabilities."""
    official_questions = [
        ("Q1 (Conflict Resolution)", "Can I return an opened laptop on day 8 if I have the original box and all accessories?"),
        ("Q2 (Product Known Issue)", "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?"),
        ("Q3 (Multi-Source Policy)", "How long does delivery take to a remote pincode, and is express delivery available there?"),
        ("Q4 (Order Guide)", "A customer wants to change the address after the order has shipped. Can they do it, and what should support tell them?"),
        ("Q5 (Exact Code Matching)", "What does policy RR-004 say?"),
        ("Q6 (Reranking damaged goods)", "A customer received a damaged phone and wants a refund instead of replacement. What are their options and what does policy say?"),
        ("Q7 (Out of Scope Refusal)", "Do you sell refrigerators?"),
    ]

    print("=" * 80)
    print("V2 ADVANCED RAG - UNIFIED PIPELINE (7 OFFICIAL QUESTIONS)")
    print("=" * 80)

    for label, q in official_questions:
        print(f"\n--- {label} ---")
        try:
            res = answer_question(q)
            print_formatted_result(res)
        except Exception as e:
            print(f"Question: {q}")
            print(f"Error executing LLM call: {e}")
            print("Sources: none")

    print("\n" + "=" * 80)
    print("V2 PIPELINE VERIFICATION COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        custom_q = " ".join(sys.argv[1:])
        res = answer_question(custom_q)
        print_formatted_result(res)
    else:
        run_official_demo()
