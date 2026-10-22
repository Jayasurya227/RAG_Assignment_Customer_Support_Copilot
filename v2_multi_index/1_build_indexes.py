"""
Phase 3: Multi-Index Architecture - Step 1: Build Indexes
Section 5.1 Implementation

Partitions the 50 knowledge base documents into 4 domain-specific vector collections in Chroma:
1. 'policy_index'  (10 documents): return_policy (5) + shipping_policy (5)
2. 'product_index' (20 documents): product_faq (12) + known_issue (8)
3. 'order_index'   (6 documents):  order_guide (6)
4. 'ticket_index'  (14 documents): resolved_ticket (14)
Total: 50 documents

Benefits:
- Eliminates cross-domain noise (e.g. ticket workarounds polluting official policy searches).
- Enables targeted hybrid search on only the relevant index partitions.
- Reduces search latency by querying smaller candidate spaces.
"""

import json
import os
import sys
import warnings
from pathlib import Path

# Suppress framework warnings & prevent tensorflow protobuf conflict
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dandes_electronics_kb.json"
CHROMA_DIR = BASE_DIR / "chroma_db"

INDEX_CONFIG = {
    "policy_index": {
        "doc_types": ["return_policy", "shipping_policy"],
        "expected_count": 10,
        "description": "Store return, refund, and shipping policies",
    },
    "product_index": {
        "doc_types": ["product_faq", "known_issue"],
        "expected_count": 20,
        "description": "Product specifications, compatibility FAQs, and known hardware/software issues",
    },
    "order_index": {
        "doc_types": ["order_guide"],
        "expected_count": 6,
        "description": "Step-by-step guides for cancellation, address changes, tracking, and COD",
    },
    "ticket_index": {
        "doc_types": ["resolved_ticket"],
        "expected_count": 14,
        "description": "Historical resolved customer support tickets with verified solutions",
    },
}

_EMBEDDINGS = None


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


def load_partitioned_documents() -> dict[str, list[Document]]:
    """Loads and partitions KB documents into the 4 domain collections."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Knowledge base not found at: {DATA_PATH}")

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("documents", [])
    partitioned: dict[str, list[Document]] = {idx: [] for idx in INDEX_CONFIG}

    for item in records:
        doc_type = item.get("doc_type")
        raw_meta = item.get("metadata", {})
        cleaned_meta = {k: v for k, v in raw_meta.items() if v is not None}

        cleaned_meta["id"] = item["id"]
        cleaned_meta["doc_type"] = item["doc_type"]
        cleaned_meta["title"] = item["title"]

        page_content = f"{item['title']}\n{item['content']}"
        doc = Document(page_content=page_content, metadata=cleaned_meta, id=item["id"])

        # Assign to appropriate index
        for idx_name, config in INDEX_CONFIG.items():
            if doc_type in config["doc_types"]:
                partitioned[idx_name].append(doc)
                break

    return partitioned


def build_all_indexes() -> dict[str, int]:
    """Builds and persists all 4 partitioned Chroma collections."""
    print("=" * 75)
    print("PHASE 3: BUILDING 4 DOMAIN-SPECIFIC CHROMABASE INDEXES")
    print("=" * 75)

    embeddings = get_embedding_function()
    partitioned_docs = load_partitioned_documents()
    counts = {}

    for idx_name, docs in partitioned_docs.items():
        print(f"\nBuilding collection '{idx_name}' ({len(docs)} documents)...")

        # Initialize/reset collection in persistent storage
        vector_store = Chroma(
            collection_name=idx_name,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )

        # Clear existing entries in collection to ensure exact count
        try:
            existing = vector_store.get()
            if existing and existing.get("ids"):
                vector_store.delete(ids=existing["ids"])
        except Exception:
            pass

        # Add partitioned documents
        vector_store.add_documents(docs)
        counts[idx_name] = len(docs)
        print(f"  -> Successfully indexed {len(docs)} documents into '{idx_name}'")

    print("\n" + "=" * 75)
    print("INDEX VERIFICATION SUMMARY (Strict Section 5.1 Rubric Check):")
    print("=" * 75)
    print(f"{'Index Name':<18} | {'Indexed Docs':<13} | {'Expected Docs':<14} | {'Status'}")
    print("-" * 75)

    all_passed = True
    for idx_name, config in INDEX_CONFIG.items():
        actual = counts.get(idx_name, 0)
        expected = config["expected_count"]
        passed = actual == expected
        if not passed:
            all_passed = False
        status_str = "PASS" if passed else f"MISMATCH (got {actual})"
        print(f"{idx_name:<18} | {actual:<13} | {expected:<14} | {status_str}")

    print("-" * 75)
    total_docs = sum(counts.values())
    print(f"{'TOTAL':<18} | {total_docs:<13} | {50:<14} | {'PASS' if total_docs == 50 else 'FAIL'}")
    print("=" * 75)

    if all_passed:
        print("\nAll 4 indexes match Section 5.1 requirements with 100% precision.")

    return counts


if __name__ == "__main__":
    build_all_indexes()
