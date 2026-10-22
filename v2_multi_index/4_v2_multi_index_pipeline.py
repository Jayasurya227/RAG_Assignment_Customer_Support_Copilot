"""
Phase 3: Multi-Index Architecture - Step 4: Unified Multi-Index Pipeline
Section 6.2 Implementation

The complete production pipeline integrating Intent Routing + Multi-Index Targeted Search:
1. Step 1 (Intent Routing): Evaluates question with 3_intent_router.py and prints the routing decision.
2. Step 2 (Direct Refusal): If routed to ["none"], immediately returns:
   "I don't know based on the available information." with Sources: none.
3. Step 3 (Targeted Multi-Index Search): Queries only designated partitions using 2_multi_index_search.py,
   running intra-index hybrid search and global Cross-Encoder reranking to obtain the top 5 documents.
4. Step 4 (Conflict Resolution - "Latest Date Wins"): Formats context with effective dates so current
   July 2026 policies supersede older 2025 tickets.
5. Step 5 (Grounded LLM Generation): Generates accurate customer response citing source document IDs.
6. Step 6 (Deliverable Generation): Automatically saves benchmark execution outputs to 'demo_output.txt'
   fulfilling Section 9.1 requirements.
"""

import importlib.util
import json
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
DEMO_OUTPUT_FILE = BASE_DIR / "demo_output.txt"
EXACT_REFUSAL = "I don't know based on the available information."
TOP_K = 5

_LLM = None
_ROUTER_FN = None
_MULTI_SEARCH_FN = None


def load_module(module_name: str, file_path: Path):
    """Safely loads a module from a path containing numeric prefixes."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def get_pipeline_components():
    """Lazily loads and caches the router and multi-index search functions."""
    global _ROUTER_FN, _MULTI_SEARCH_FN
    if _ROUTER_FN is None or _MULTI_SEARCH_FN is None:
        multi_dir = BASE_DIR / "v2_multi_index"
        router_mod = load_module("v2_router_mod", multi_dir / "3_intent_router.py")
        _ROUTER_FN = router_mod.route_query

        search_mod = load_module("v2_multi_search_mod", multi_dir / "2_multi_index_search.py")
        _MULTI_SEARCH_FN = search_mod.multi_index_search

    return _ROUTER_FN, _MULTI_SEARCH_FN


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
   If retrieved documents contain conflicting statements, return windows, or fees (e.g. an older ticket from 2025 stating 7 days versus a current policy from July 2026 stating 10 days), ALWAYS prioritize and answer according to the document with the most recent effective date. Explicitly explain that the current updated policy supersedes the older document.
3. If the context does not contain enough information to answer the question, reply EXACTLY with:
   "I don't know based on the available information."
4. Keep your answers concise, direct, helpful, and professional.

Retrieved Context:
{context}
"""


def format_context_with_metadata(docs: list[Document]) -> str:
    """Formats context with Document ID, Effective Date, Type, and Cross-Encoder relevance score."""
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


def answer_question(question: str, print_routing: bool = True) -> dict[str, Any]:
    """
    Executes complete V2 Multi-Index Pipeline:
    1. Intent classification.
    2. Targeted multi-index search.
    3. Conflict-aware grounded LLM generation.
    Returns:
      {
        "question": ...,
        "answer": ...,
        "sources": [...],
        "routing_decision": {...},
        "retrieved_docs": [...]
      }
    """
    route_fn, multi_search_fn = get_pipeline_components()

    # Step 1: Routing Decision
    routing_decision = route_fn(question)
    selected_indexes = routing_decision.get("indexes", ["policy_index", "product_index", "order_index", "ticket_index"])

    if print_routing:
        print(f"Routing Decision:")
        print(f"  Intents:        {routing_decision.get('intents')}")
        print(f"  Target Indexes: {selected_indexes}")
        print(f"  Reason:         {routing_decision.get('reason')}")

    # Step 2: Immediate refusal for out-of-scope queries
    if selected_indexes == ["none"]:
        return {
            "question": question,
            "answer": EXACT_REFUSAL,
            "sources": [],
            "routing_decision": routing_decision,
            "retrieved_docs": [],
        }

    # Step 3: Targeted Multi-Index Search & Global Reranking
    retrieved_docs = multi_search_fn(question, selected_indexes=selected_indexes, top_k=TOP_K)

    # Out-of-scope check based on cross-encoder score
    top_score = retrieved_docs[0].metadata.get("rerank_score", 0.0) if retrieved_docs else -99.0
    if not retrieved_docs or top_score < -4.0:
        return {
            "question": question,
            "answer": EXACT_REFUSAL,
            "sources": [],
            "routing_decision": routing_decision,
            "retrieved_docs": retrieved_docs,
        }

    # Step 4: Grounded Answer Generation
    context_str = format_context_with_metadata(retrieved_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})
    answer_text = response.content.strip()

    if EXACT_REFUSAL in answer_text:
        sources = []
    else:
        sources = [d.metadata.get("id") for d in retrieved_docs if d.metadata.get("id")]
        sources = list(dict.fromkeys(sources))

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "routing_decision": routing_decision,
        "retrieved_docs": retrieved_docs,
    }


def format_output_block(res: dict[str, Any]) -> str:
    """Formats output block strictly matching Section 3.3 and 6.2."""
    routing = res.get("routing_decision", {})
    indexes_str = ", ".join(routing.get("indexes", []))
    sources_str = ", ".join(res.get("sources", [])) if res.get("sources") else "none"

    lines = [
        f"Question: {res['question']}",
        f"Routing Decision: Indexes: [{indexes_str}] | Reason: {routing.get('reason', '')}",
        f"Answer: {res['answer']}",
        f"Sources: {sources_str}",
    ]
    return "\n".join(lines)


def run_benchmark_and_generate_demo_output():
    """
    Executes the 7 official Section 3.2 benchmark questions,
    prints formatted outputs to the terminal, and writes them to demo_output.txt.
    """
    official_questions = [
        ("Q1 (Conflict Resolution)", "Can I return an opened laptop on day 8 if I have the original box and all accessories?"),
        ("Q2 (Hardware Issue + Ticket)", "My Galaxy Buds3 Pro left earbud won't connect. Is this a known issue, and has the same problem been resolved for another customer? What fixed it?"),
        ("Q3 (Shipping Policy)", "How long does delivery take to a remote pincode, and is express delivery available there?"),
        ("Q4 (Order Guide)", "A customer wants to change the address after the order has shipped. Can they do it, and what should support tell them?"),
        ("Q5 (Exact Code Matching)", "What does policy RR-004 say?"),
        ("Q6 (Reranking Damaged Goods)", "A customer received a damaged phone and wants a refund instead of replacement. What are their options and what does policy say?"),
        ("Q7 (Out of Scope Refusal)", "Do you sell refrigerators?"),
    ]

    print("=" * 80)
    print("PHASE 3: UNIFIED MULTI-INDEX PIPELINE BENCHMARK (7 OFFICIAL QUESTIONS)")
    print("=" * 80)

    output_blocks = [
        "================================================================================",
        "DANDES ELECTRONICS CUSTOMER SUPPORT COPILOT - MULTI-INDEX V2 PIPELINE DEMO",
        "Section 6.2 & Section 9.1 Official Execution Output",
        "================================================================================\n",
    ]

    for label, q in official_questions:
        print(f"\n--- {label} ---")
        try:
            res = answer_question(q, print_routing=True)
            block = format_output_block(res)
            print(f"Answer: {res['answer']}")
            print(f"Sources: {', '.join(res['sources']) if res['sources'] else 'none'}")
            output_blocks.append(f"--- {label} ---\n{block}\n")
        except Exception as e:
            err_msg = f"Error during generation: {e}"
            print(f"Question: {q}")
            print(err_msg)
            output_blocks.append(f"--- {label} ---\nQuestion: {q}\n{err_msg}\n")

    # Save to demo_output.txt
    with open(DEMO_OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(output_blocks))

    print("\n" + "=" * 80)
    print(f"Saved complete benchmark execution log to: {DEMO_OUTPUT_FILE.relative_to(BASE_DIR)}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        custom_q = " ".join(sys.argv[1:])
        res = answer_question(custom_q, print_routing=True)
        print(f"\nAnswer: {res['answer']}")
        print(f"Sources: {', '.join(res['sources']) if res['sources'] else 'none'}")
    else:
        run_benchmark_and_generate_demo_output()
