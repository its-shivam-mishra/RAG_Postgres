"""
query.py
--------
RAG query orchestration.

Flow
----
1. Retrieve + rerank documents from the vector store.
2. If no documents found → fall back to Tavily immediately.
3. Run the LLM over the reranked documents.
4. If the LLM signals it cannot answer → fall back to Tavily.
5. Return (answer, sources, source_type) where source_type is "rag" | "tavily".
"""

import logging
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from app.models import llm
from app.retriever import retrieve_and_rerank
from app.search import tavily_search, is_uncertain_answer

logger = logging.getLogger(__name__)

# ── RAG prompt ────────────────────────────────────────────────────────────────
_RAG_SYSTEM_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If the answer is not contained in the context, respond with exactly: "
    "'I don't know based on the provided documents.' "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "{context}"
)

_rag_prompt = ChatPromptTemplate.from_messages([
    ("system", _RAG_SYSTEM_PROMPT),
    ("human", "{input}"),
])

_rag_chain = create_stuff_documents_chain(llm, _rag_prompt)


def answer_question(
    question: str,
    user_id: str,
    filename: str | None = None,
) -> tuple[str, list[str], str]:
    """
    Answer *question* using the RAG pipeline, falling back to Tavily Search
    when the documents cannot provide a satisfactory answer.

    Returns
    -------
    tuple[str, list[str], str]
        (answer_text, sources, source_type)
        source_type is ``"rag"`` or ``"tavily"``.
    """
    # ── Step 1: retrieve + rerank ─────────────────────────────────────────────
    docs = retrieve_and_rerank(question, user_id, filename)

    # ── Step 2: no docs → immediate web fallback ──────────────────────────────
    if not docs:
        logger.info(
            "[Query] No docs for user '%s'. Falling back to Tavily.", user_id
        )
        answer, sources = tavily_search(question)
        return answer, sources, "tavily"

    # ── Step 3: answer from RAG context ───────────────────────────────────────
    answer_text: str = _rag_chain.invoke({"input": question, "context": docs})

    # ── Step 4: uncertain answer → web fallback ───────────────────────────────
    if is_uncertain_answer(answer_text):
        logger.info(
            "[Query] Uncertain RAG answer. Falling back to Tavily."
        )
        answer, sources = tavily_search(question)
        return answer, sources, "tavily"

    # ── Step 5: return RAG answer with unique source filenames ────────────────
    sources: list[str] = []
    for doc in docs:
        src = doc.metadata.get("source", "Unknown")
        if src not in sources:
            sources.append(src)

    return answer_text, sources, "rag"
