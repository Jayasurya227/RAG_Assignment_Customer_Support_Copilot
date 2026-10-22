"""
V1: Online Query Pipeline
Takes a user question, retrieves the top 5 documents from the 'dandes_v1' Chroma collection,
and generates a grounded answer citing the source document IDs.
If information is not available in the retrieved context, replies with the exact refusal string:
"I don't know based on the available information."
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

# Suppress framework warnings & prevent tensorflow protobuf conflict
warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "dandes_v1"
TOP_K = 5
EXACT_REFUSAL = "I don't know based on the available information."

SYSTEM_PROMPT = """You are a helpful customer support assistant for Dandes Electronics, an online electronics store.
Answer the user's question accurately using ONLY the provided retrieved context below.

Rules:
1. Ground your answer strictly in the provided context. Do NOT make up any information.
2. If the context does not contain enough information to answer the question, reply EXACTLY with:
   "I don't know based on the available information."
3. Keep your answers concise, direct, and professional.

Retrieved Context:
{context}
"""


_EMBEDDINGS = None
_VECTOR_STORE = None


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


def get_vector_store() -> Chroma:
    """Connect to the persistent dandes_v1 collection (cached in-memory)."""
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        embeddings = get_embedding_function()
        _VECTOR_STORE = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
            persist_directory=str(CHROMA_DIR),
        )
    return _VECTOR_STORE


def get_llm() -> ChatOpenAI:
    """Initialize the chat LLM with optional custom base URL."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip('"').strip("'")
    base_url = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    kwargs = {
        "model": model_name,
        "temperature": 0.0,
        "openai_api_key": api_key,
    }
    if base_url:
        kwargs["openai_api_base"] = base_url

    return ChatOpenAI(**kwargs)


def format_context(docs: list[Document]) -> str:
    """Format retrieved documents into a clean string for the LLM prompt."""
    context_blocks = []
    for doc in docs:
        doc_id = doc.metadata.get("id", "UNKNOWN")
        context_blocks.append(f"[Document ID: {doc_id}]\n{doc.page_content}")
    return "\n\n".join(context_blocks)


def answer_question(question: str) -> dict[str, Any]:
    """
    Executes V1 retrieval (Top 5) and generates an answer with sources.
    Returns a dictionary with 'question', 'answer', 'sources', and 'retrieved_docs'.
    """
    vector_store = get_vector_store()
    retriever = vector_store.as_retriever(search_kwargs={"k": TOP_K})
    retrieved_docs = retriever.invoke(question)

    if not retrieved_docs:
        return {
            "question": question,
            "answer": EXACT_REFUSAL,
            "sources": [],
            "retrieved_docs": [],
        }

    context_str = format_context(retrieved_docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}"),
    ])

    llm = get_llm()
    chain = prompt | llm
    response = chain.invoke({"context": context_str, "question": question})
    answer_text = response.content.strip()

    # Extract unique source IDs from retrieved documents that contributed
    if EXACT_REFUSAL in answer_text:
        sources: list[str] = []
    else:
        # Collect IDs of retrieved docs
        sources = [doc.metadata.get("id") for doc in retrieved_docs if doc.metadata.get("id")]
        # Deduplicate while preserving order
        sources = list(dict.fromkeys(sources))

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "retrieved_docs": retrieved_docs,
    }


def print_formatted_result(result: dict[str, Any]) -> None:
    """Prints output matching the required format in Section 3.3."""
    print(f"\nQuestion: {result['question']}")
    print(f"Answer: {result['answer']}")
    if result["sources"]:
        print(f"Sources: {', '.join(result['sources'])}")
    else:
        print("Sources: none")


if __name__ == "__main__":
    import sys
    print("=" * 60)
    print("V1 Online Pipeline")
    print("=" * 60)

    # If question passed via command line argument, answer it directly
    if len(sys.argv) > 1:
        custom_q = " ".join(sys.argv[1:])
        res = answer_question(custom_q)
        print_formatted_result(res)
    else:
        # 1. Run the official demo question required by Section 3.3
        demo_question = "How long do UPI refunds take?"
        res = answer_question(demo_question)
        print_formatted_result(res)
        print("=" * 60)

        # 2. Interactive prompt: Type or paste any question directly
        print("\nInteractive Mode: Paste any question below (type 'exit' to stop):")
        while True:
            try:
                user_q = input("\nEnter Question: ").strip()
                if not user_q or user_q.lower() in ("exit", "quit", "q"):
                    print("Exiting interactive mode.")
                    break
                result = answer_question(user_q)
                print_formatted_result(result)
            except (KeyboardInterrupt, EOFError):
                print("\nExiting.")
                break
