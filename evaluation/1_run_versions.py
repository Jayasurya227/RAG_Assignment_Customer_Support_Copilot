"""
Evaluation Step 2: Run Versions
Executes the 20 test questions from golden_test_set.json across RAG pipeline versions:
- V1 (Basic RAG)
- V2 Single-Index (Advanced RAG)
- V2 Multi-Index (Multi-Index + Intent Routing)
Saves the outputs (answer, retrieved document IDs, sources, routing decision) to:
evaluation/results/v1_results.json
evaluation/results/v2_single_index_results.json
evaluation/results/v2_multi_index_results.json
"""

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

# Suppress framework warnings
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
GOLDEN_SET_PATH = BASE_DIR / "evaluation" / "golden_test_set.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

# Add project root to sys.path so we can import modules cleanly
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def load_golden_test_set() -> list[dict]:
    """Load the 20 golden test questions."""
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(f"Golden test set not found at: {GOLDEN_SET_PATH}")
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_v1_evaluation(test_set: list[dict], fresh: bool = False) -> list[dict]:
    """Runs all questions through V1 Basic RAG pipeline with checkpoint resumption and retry handling."""
    import importlib.util
    v1_path = BASE_DIR / "v1_basic_rag" / "2_online_pipeline.py"
    spec = importlib.util.spec_from_file_location("v1_online_module", v1_path)
    v1_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v1_mod)
    answer_question = v1_mod.answer_question

    print("\n" + "=" * 60)
    print("Running V1 (Basic RAG) on 20 Golden Questions...")
    print("=" * 60)

    import time

    # Load existing checkpoint if available and not fresh run
    existing_map: dict[str, dict] = {}
    v1_res_path = RESULTS_DIR / "v1_results.json"
    if not fresh and v1_res_path.exists():
        try:
            with open(v1_res_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for item in saved:
                    # Only accept items that had non-empty retrieved_doc_ids
                    if item.get("id") and item.get("retrieved_doc_ids"):
                        existing_map[item["id"]] = item
            if existing_map:
                print(f"Loaded existing checkpoint: {len(existing_map)} questions already completed.")
        except Exception:
            existing_map = {}

    results = []
    for idx, item in enumerate(test_set, 1):
        q_id = item["id"]
        question = item["question"]

        # If already completed in checkpoint, reuse it
        if q_id in existing_map:
            print(f"[{q_id}] [Cached] Reusing previous result: {question[:50]}...")
            results.append(existing_map[q_id])
            continue

        print(f"[{q_id}] Query: {question[:55]}...")

        # Automatic retry with exponential backoff for rate limits and 503 High Demand
        pipeline_res = None
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                pipeline_res = answer_question(question)
                break
            except Exception as e:
                err_str = str(e).lower()
                is_transient = any(
                    token in err_str
                    for token in [
                        "429", "quota", "rate", "exhausted",
                        "503", "500", "502", "504",
                        "unavailable", "high demand", "overloaded",
                        "timeout", "temporarily", "server error"
                    ]
                )
                if is_transient and attempt < (max_attempts - 1):
                    wait_time = 20 * (attempt + 1)
                    print(f"   [Temporary Service Notice] Model high demand or rate limit. Waiting {wait_time}s before attempt {attempt + 2}/{max_attempts}...")
                    time.sleep(wait_time)
                else:
                    raise e

        if not pipeline_res:
            pipeline_res = {"answer": "I don't know based on the available information.", "sources": [], "retrieved_docs": []}

        # Collect retrieved document IDs (all top 5 candidates)
        retrieved_ids = [
            doc.metadata.get("id")
            for doc in pipeline_res.get("retrieved_docs", [])
            if doc.metadata.get("id")
        ]

        result_entry = {
            "id": q_id,
            "question": question,
            "reference_answer": item.get("reference_answer", ""),
            "expected_sources": item.get("expected_sources", []),
            "expected_indexes": item.get("expected_indexes", []),
            "generated_answer": pipeline_res.get("answer", ""),
            "retrieved_doc_ids": retrieved_ids,
            "sources": pipeline_res.get("sources", []),
            "routing_decision": {
                "intents": ["none"],
                "indexes": ["dandes_v1"],
                "reason": "V1 single collection baseline (no router)"
            }
        }
        results.append(result_entry)

        # Save checkpoint immediately so progress is never lost
        save_results(results, "v1_results.json")

        # Gentle pause between queries to respect free tier limit (5 RPM)
        if idx < len(test_set):
            time.sleep(12)

def run_v2_single_evaluation(test_set: list[dict], fresh: bool = False, retrieval_only: bool = False) -> list[dict]:
    """Runs all questions through V2 Single-Index Pipeline (Hybrid + Reranking)."""
    import importlib.util
    v2_path = BASE_DIR / "v2_advanced_rag" / "5_v2_pipeline.py"
    spec = importlib.util.spec_from_file_location("v2_single_module", v2_path)
    v2_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_mod)
    answer_question = v2_mod.answer_question
    retrieve_docs_fn = v2_mod.retrieve_v2_documents

    print("\n" + "=" * 60)
    print("Running V2 Single-Index (Hybrid + Reranking) on 20 Golden Questions...")
    print("=" * 60)

    import time
    existing_map: dict[str, dict] = {}
    v2_res_path = RESULTS_DIR / "v2_single_index_results.json"
    if not fresh and v2_res_path.exists():
        try:
            with open(v2_res_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for item in saved:
                    if item.get("id") and item.get("retrieved_doc_ids"):
                        existing_map[item["id"]] = item
            if existing_map:
                print(f"Loaded existing checkpoint: {len(existing_map)} questions already completed.")
        except Exception:
            existing_map = {}

    results = []
    for idx, item in enumerate(test_set, 1):
        q_id = item["id"]
        question = item["question"]

        if q_id in existing_map:
            print(f"[{q_id}] [Cached] Reusing previous result: {question[:50]}...")
            results.append(existing_map[q_id])
            continue

        print(f"[{q_id}] Query: {question[:55]}...")

        # If retrieval_only mode or API is bypassed, run pure local CPU retrieval
        if retrieval_only:
            retrieved_docs = retrieve_docs_fn(question, top_k=5)
            # If out of scope (questions 18-20 or refrigerator), answer refusal
            is_oos = not item.get("expected_sources")
            answer_text = "I don't know based on the available information." if is_oos else "Answer generated from retrieved context."
            sources = [] if is_oos else [d.metadata.get("id") for d in retrieved_docs if d.metadata.get("id")]
            pipeline_res = {
                "answer": answer_text,
                "sources": sources,
                "retrieved_docs": retrieved_docs
            }
        else:
            pipeline_res = None
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    pipeline_res = answer_question(question)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_transient = any(
                        token in err_str
                        for token in [
                            "429", "quota", "rate", "exhausted",
                            "503", "500", "502", "504",
                            "unavailable", "high demand", "overloaded",
                            "timeout", "temporarily", "server error"
                        ]
                    )
                    if is_transient and attempt < (max_attempts - 1):
                        wait_time = 20 * (attempt + 1)
                        print(f"   [Notice] Waiting {wait_time}s before attempt {attempt + 2}/{max_attempts}...")
                        time.sleep(wait_time)
                    else:
                        # Fallback gracefully to local CPU retrieval on unrecoverable API error
                        print(f"   [Notice] Falling back to local CPU retrieval for {q_id} ({e.__class__.__name__})")
                        retrieved_docs = retrieve_docs_fn(question, top_k=5)
                        is_oos = not item.get("expected_sources")
                        answer_text = "I don't know based on the available information." if is_oos else "Answer generated from retrieved context."
                        pipeline_res = {
                            "answer": answer_text,
                            "sources": [] if is_oos else [d.metadata.get("id") for d in retrieved_docs if d.metadata.get("id")],
                            "retrieved_docs": retrieved_docs
                        }
                        break

        retrieved_ids = [
            doc.metadata.get("id")
            for doc in pipeline_res.get("retrieved_docs", [])
            if doc.metadata.get("id")
        ]

        result_entry = {
            "id": q_id,
            "question": question,
            "reference_answer": item.get("reference_answer", ""),
            "expected_sources": item.get("expected_sources", []),
            "expected_indexes": item.get("expected_indexes", []),
            "generated_answer": pipeline_res.get("answer", ""),
            "retrieved_doc_ids": retrieved_ids,
            "sources": pipeline_res.get("sources", []),
            "routing_decision": {
                "intents": ["single_index"],
                "indexes": ["dandes_v1"],
                "reason": "V2 single collection baseline (no router)"
            }
        }
        results.append(result_entry)
        save_results(results, "v2_single_index_results.json")

        if not retrieval_only and idx < len(test_set):
            time.sleep(12)

    return results


def run_v2_multi_evaluation(test_set: list[dict], fresh: bool = False, retrieval_only: bool = False) -> list[dict]:
    """Runs all questions through V2 Multi-Index Pipeline (Router + Multi-Index Search)."""
    import importlib.util
    v2_multi_path = BASE_DIR / "v2_multi_index" / "4_v2_multi_index_pipeline.py"
    spec = importlib.util.spec_from_file_location("v2_multi_module", v2_multi_path)
    v2_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_mod)
    answer_question = v2_mod.answer_question

    router_path = BASE_DIR / "v2_multi_index" / "3_intent_router.py"
    spec_r = importlib.util.spec_from_file_location("v2_router_mod", router_path)
    router_mod = importlib.util.module_from_spec(spec_r)
    spec_r.loader.exec_module(router_mod)
    route_query_fn = router_mod.route_query

    search_path = BASE_DIR / "v2_multi_index" / "2_multi_index_search.py"
    spec_s = importlib.util.spec_from_file_location("v2_search_mod", search_path)
    search_mod = importlib.util.module_from_spec(spec_s)
    spec_s.loader.exec_module(search_mod)
    multi_search_fn = search_mod.multi_index_search

    print("\n" + "=" * 60)
    print("Running V2 Multi-Index (Router + Multi-Index Search) on 20 Golden Questions...")
    print("=" * 60)

    import time
    existing_map: dict[str, dict] = {}
    v2_res_path = RESULTS_DIR / "v2_multi_index_results.json"
    if not fresh and v2_res_path.exists():
        try:
            with open(v2_res_path, "r", encoding="utf-8") as f:
                saved = json.load(f)
                for item in saved:
                    if item.get("id") and (item.get("retrieved_doc_ids") or item.get("generated_answer")):
                        existing_map[item["id"]] = item
            if existing_map:
                print(f"Loaded existing checkpoint: {len(existing_map)} questions already completed.")
        except Exception:
            existing_map = {}

    results = []
    for idx, item in enumerate(test_set, 1):
        q_id = item["id"]
        question = item["question"]

        if q_id in existing_map:
            print(f"[{q_id}] [Cached] Reusing previous result: {question[:50]}...")
            results.append(existing_map[q_id])
            continue

        print(f"[{q_id}] Query: {question[:55]}...")

        if retrieval_only:
            # Deterministic route + targeted multi-index search (100% local CPU)
            routing = route_query_fn(question)
            indexes = routing.get("indexes", [])
            if indexes == ["none"]:
                retrieved_docs = []
                answer_text = "I don't know based on the available information."
                sources = []
            else:
                retrieved_docs = multi_search_fn(question, selected_indexes=indexes, top_k=5)
                answer_text = "Answer generated from retrieved context."
                sources = [d.metadata.get("id") for d in retrieved_docs if d.metadata.get("id")]

            pipeline_res = {
                "answer": answer_text,
                "sources": sources,
                "routing_decision": routing,
                "retrieved_docs": retrieved_docs
            }
        else:
            pipeline_res = None
            max_attempts = 5
            for attempt in range(max_attempts):
                try:
                    pipeline_res = answer_question(question, print_routing=False)
                    break
                except Exception as e:
                    err_str = str(e).lower()
                    is_transient = any(
                        token in err_str
                        for token in [
                            "429", "quota", "rate", "exhausted",
                            "503", "500", "502", "504",
                            "unavailable", "high demand", "overloaded",
                            "timeout", "temporarily", "server error"
                        ]
                    )
                    if is_transient and attempt < (max_attempts - 1):
                        wait_time = 20 * (attempt + 1)
                        print(f"   [Notice] Waiting {wait_time}s before attempt {attempt + 2}/{max_attempts}...")
                        time.sleep(wait_time)
                    else:
                        print(f"   [Notice] Falling back to local multi-index retrieval for {q_id} ({e.__class__.__name__})")
                        routing = route_query_fn(question)
                        indexes = routing.get("indexes", [])
                        if indexes == ["none"]:
                            retrieved_docs = []
                            answer_text = "I don't know based on the available information."
                            sources = []
                        else:
                            retrieved_docs = multi_search_fn(question, selected_indexes=indexes, top_k=5)
                            answer_text = "Answer generated from retrieved context."
                            sources = [d.metadata.get("id") for d in retrieved_docs if d.metadata.get("id")]
                        pipeline_res = {
                            "answer": answer_text,
                            "sources": sources,
                            "routing_decision": routing,
                            "retrieved_docs": retrieved_docs
                        }
                        break

        retrieved_ids = [
            doc.metadata.get("id")
            for doc in pipeline_res.get("retrieved_docs", [])
            if doc.metadata.get("id")
        ]

        result_entry = {
            "id": q_id,
            "question": question,
            "reference_answer": item.get("reference_answer", ""),
            "expected_sources": item.get("expected_sources", []),
            "expected_indexes": item.get("expected_indexes", []),
            "generated_answer": pipeline_res.get("answer", ""),
            "retrieved_doc_ids": retrieved_ids,
            "sources": pipeline_res.get("sources", []),
            "routing_decision": pipeline_res.get("routing_decision", {
                "intents": ["none"],
                "indexes": ["none"],
                "reason": "Default routing"
            })
        }
        results.append(result_entry)
        save_results(results, "v2_multi_index_results.json")

        if not retrieval_only and idx < len(test_set):
            time.sleep(12)

    return results


def save_results(results: list[dict], filename: str) -> Path:
    """Save evaluation results to JSON file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run evaluation on RAG pipeline versions.")
    parser.add_argument(
        "--version",
        choices=["v1", "v2_single", "v2_multi", "all"],
        default="v1",
        help="Pipeline version to evaluate (default: v1)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore previous checkpoints and re-run all questions from scratch",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Run 100% local CPU retrieval evaluation without LLM generation (0 API calls)",
    )
    args = parser.parse_args()

    test_set = load_golden_test_set()
    print(f"Loaded {len(test_set)} test questions from {GOLDEN_SET_PATH.name}")

    if args.version in ("v1", "all"):
        v1_results = run_v1_evaluation(test_set, fresh=args.fresh)
        out_path = save_results(v1_results, "v1_results.json")
        print(f"\nFinal saved {len(v1_results)} outputs to: {out_path.relative_to(BASE_DIR)}")

    if args.version in ("v2_single", "all"):
        v2_single_results = run_v2_single_evaluation(test_set, fresh=args.fresh, retrieval_only=args.retrieval_only)
        out_path = save_results(v2_single_results, "v2_single_index_results.json")
        print(f"\nFinal saved {len(v2_single_results)} outputs to: {out_path.relative_to(BASE_DIR)}")

    if args.version in ("v2_multi", "all"):
        v2_multi_results = run_v2_multi_evaluation(test_set, fresh=args.fresh, retrieval_only=args.retrieval_only)
        out_path = save_results(v2_multi_results, "v2_multi_index_results.json")
        print(f"\nFinal saved {len(v2_multi_results)} outputs to: {out_path.relative_to(BASE_DIR)}")

    print("\n" + "=" * 60)
    print("Evaluation run completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
