"""
V1: Conversational RAG Pipeline
Extends the online query pipeline with conversation history.
Follow-up questions are contextualized using chat history so that references
(e.g., "What if the box is opened?") are resolved correctly to the subject laptop/product.
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
os.environ["USE_TF"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "dandes_v1"
TOP_K = 5
EXACT_REFUSAL = "I don't know based on the available information."

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

QA_SYSTEM_PROMPT = """You are a helpful customer support assistant for Dandes Electronics, an online electronics store.
Answer the user's question accurately using ONLY the provided retrieved context below.

Rules:
1. Ground your answer strictly in the provided context. Do NOT make up any information.
2. If the context does not contain enough information to answer the question, reply EXACTLY with:
   "I don't know based on the available information."
3. Keep your answers concise, direct, and professional.

Retrieved Context:
{context}
"""


def get_embedding_function():
    """Returns OpenAI embeddings if sk- key is present, else local HuggingFace embeddings."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip('"').strip("'")
    base_url = os.getenv("OPENAI_BASE_URL")

    if api_key.startswith("sk-"):
        from langchain_openai import OpenAIEmbeddings
        embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        kwargs = {"model": embedding_model, "openai_api_key": api_key}
        if base_url:
            kwargs["openai_api_base"] = base_url
        return OpenAIEmbeddings(**kwargs)
    else:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def get_vector_store() -> Chroma:
    """Connect to persistent dandes_v1 collection."""
    embeddings = get_embedding_function()
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR),
    )


def get_llm() -> ChatOpenAI:
    """Initialize ChatOpenAI LLM."""
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
    blocks = []
    for doc in docs:
        doc_id = doc.metadata.get("id", "UNKNOWN")
        blocks.append(f"[Document ID: {doc_id}]\n{doc.page_content}")
    return "\n\n".join(blocks)


class ConversationalRAG:
    """Conversational RAG session maintaining chat history across multiple turns."""

    def __init__(self):
        self.vector_store = get_vector_store()
        self.retriever = self.vector_store.as_retriever(search_kwargs={"k": TOP_K})
        self.llm = get_llm()
        self.chat_history: list[BaseMessage] = []

        self.contextualize_prompt = ChatPromptTemplate.from_messages([
            ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.question_rewriter = self.contextualize_prompt | self.llm

        self.qa_prompt = ChatPromptTemplate.from_messages([
            ("system", QA_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        self.qa_chain = self.qa_prompt | self.llm

    def ask(self, user_question: str) -> dict[str, Any]:
        """Processes a single conversational turn."""
        if self.chat_history:
            rewritten_response = self.question_rewriter.invoke({
                "chat_history": self.chat_history,
                "input": user_question,
            })
            standalone_query = rewritten_response.content.strip()
        else:
            standalone_query = user_question

        retrieved_docs = self.retriever.invoke(standalone_query)

        if not retrieved_docs:
            answer = EXACT_REFUSAL
            sources = []
        else:
            context_str = format_context(retrieved_docs)
            response = self.qa_chain.invoke({
                "context": context_str,
                "chat_history": self.chat_history,
                "input": user_question,
            })
            answer = response.content.strip()

            if EXACT_REFUSAL in answer:
                sources = []
            else:
                sources = [doc.metadata.get("id") for doc in retrieved_docs if doc.metadata.get("id")]
                sources = list(dict.fromkeys(sources))

        self.chat_history.append(HumanMessage(content=user_question))
        self.chat_history.append(AIMessage(content=answer))

        return {
            "question": user_question,
            "standalone_query": standalone_query,
            "answer": answer,
            "sources": sources,
            "retrieved_docs": retrieved_docs,
        }


def run_demo():
    """Demonstrates follow-up question understanding as required by Section 3.2."""
    print("=" * 60)
    print("V1 Conversational RAG Demo")
    print("=" * 60)

    session = ConversationalRAG()

    # Turn 1
    q1 = "What is the return window for laptops?"
    res1 = session.ask(q1)
    print(f"\nQ1: {res1['question']}")
    print(f"Answer: {res1['answer']}")
    print(f"Sources: {', '.join(res1['sources']) if res1['sources'] else 'none'}")

    # Turn 2 (Follow-up)
    q2 = "What if the box is opened?"
    res2 = session.ask(q2)
    print(f"\nQ2: {res2['question']}")
    print(f"(Understood as: '{res2['standalone_query']}')")
    print(f"Answer: {res2['answer']}")
    print(f"Sources: {', '.join(res2['sources']) if res2['sources'] else 'none'}")
    print("=" * 60)


if __name__ == "__main__":
    run_demo()
