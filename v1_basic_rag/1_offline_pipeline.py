"""
V1: Offline Ingestion Pipeline
Loads the knowledge base JSON, cleans metadata by removing null values,
embeds records using OpenAI text-embedding-3-small (or local HuggingFace embeddings
for free local verification), and persists them into the Chroma collection 'dandes_v1'.
"""

import json
import os
import sys
import warnings
from pathlib import Path
from dotenv import load_dotenv

# Suppress unnecessary framework deprecation notices
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_core.documents import Document
from langchain_chroma import Chroma

# Load environment variables from .env
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "dandes_electronics_kb.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "dandes_v1"


def get_embedding_function():
    """
    Returns the embedding model:
    - Uses OpenAI text-embedding-3-small when an official OpenAI key is configured.
    - Seamlessly falls back to local all-MiniLM-L6-v2 for 100% free offline execution.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip('"').strip("'")
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_key.startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        kwargs = {"model": embedding_model, "openai_api_key": api_key}
        if base_url:
            kwargs["openai_api_base"] = base_url
        print(f"Using OpenAI Embeddings ({embedding_model})...")
        return OpenAIEmbeddings(**kwargs)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        print("Using local embedding model (all-MiniLM-L6-v2) - 100% free CPU execution...")
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def load_and_prepare_documents(data_path: Path) -> list[Document]:
    """
    Load raw JSON records and convert to LangChain Documents.
    Each record is treated as one chunk.
    Drops any metadata field whose value is null as required by Section 3.2.
    """
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("documents", [])
    documents: list[Document] = []

    for item in records:
        # Retain non-null metadata fields from the nested metadata dictionary
        raw_meta = item.get("metadata", {})
        cleaned_meta = {k: v for k, v in raw_meta.items() if v is not None}

        # Add top-level identifiers to metadata
        cleaned_meta["id"] = item["id"]
        cleaned_meta["doc_type"] = item["doc_type"]
        cleaned_meta["title"] = item["title"]

        # Content formatted with Title and Content for optimal retrieval
        page_content = f"{item['title']}\n{item['content']}"

        doc = Document(
            page_content=page_content,
            metadata=cleaned_meta,
            id=item["id"],
        )
        documents.append(doc)

    return documents


def run_offline_pipeline() -> int:
    """Builds and persists the dandes_v1 Chroma collection."""
    print("=" * 60)
    print("V1 Offline Pipeline: Ingesting Knowledge Base into Chroma")
    print("=" * 60)

    # 1. Load documents
    docs = load_and_prepare_documents(DATA_PATH)
    print(f"Loaded {len(docs)} documents from {DATA_PATH.name}")

    # 2. Initialize Embeddings
    embeddings = get_embedding_function()

    # 3. Create persistent Chroma store
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )

    # 4. Add documents (reset collection if it exists to ensure exact count)
    existing = vector_store.get()
    if existing and existing["ids"]:
        vector_store.delete(ids=existing["ids"])

    vector_store.add_documents(docs)

    # 5. Verify and print stored count (Must be 50 as per Section 3.2)
    stored_count = len(vector_store.get()["ids"])
    print(f"\nSuccessfully stored in collection '{COLLECTION_NAME}':")
    print(f"Total documents stored: {stored_count}")
    print("=" * 60)

    return stored_count


def search_offline(query: str, k: int = 1):
    """Searches the local dandes_v1 Chroma collection 100% offline (no LLM, no internet)."""
    embeddings = get_embedding_function()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )
    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        print("No documents found.")
        return

    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    for i, doc in enumerate(docs, 1):
        doc_id = doc.metadata.get("id", "UNKNOWN")
        title = doc.metadata.get("title", "")
        doc_type = doc.metadata.get("doc_type", "")
        print(f"\n[Match #{i}] ID: {doc_id} ({doc_type})")
        print(f"Title: {title}")
        print(f"Content:\n{doc.page_content}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    # If a question was passed directly via CLI: python 1_offline_pipeline.py "query"
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        search_offline(query)
    else:
        # 1. Run the ingestion check
        run_offline_pipeline()

        # 2. Interactive Offline Search (Zero API calls, Zero LLM)
        print("\n" + "-" * 60)
        print("Offline Search Mode (Pure local search - Zero API / Zero LLM)")
        print("Paste any question below to test offline (or type 'exit' to quit):")
        print("-" * 60)
        while True:
            try:
                user_q = input("\nEnter Question to search offline: ").strip()
                if not user_q or user_q.lower() in ("exit", "quit", "q"):
                    print("Exiting offline search.")
                    break
                search_offline(user_q, k=1)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
