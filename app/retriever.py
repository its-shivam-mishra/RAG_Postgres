"""
retriever.py
------------
Vector-store retrieval + FlashRank reranking.

Responsibilities:
  - Build/return the PGVector store.
  - Retrieve candidate documents by similarity search.
  - Rerank candidates with FlashRank and return the top-N docs.
"""

import logging
from langchain_postgres.vectorstores import PGVector
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest

from app.config import (
    DATABASE_URL_PGVECTOR,
    COLLECTION_NAME,
    RETRIEVER_TOP_K,
    RERANKER_TOP_N,
)
from app.models import embeddings

logger = logging.getLogger(__name__)

# FlashRank Ranker — loaded once (ms-marco-TinyBERT-L-2-v2, ~15 MB)
_ranker = Ranker()


def get_vector_store() -> PGVector:
    """Return a PGVector store connected to the shared collection."""
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL_PGVECTOR,
        use_jsonb=True,
    )


def retrieve_and_rerank(
    question: str,
    user_id: str,
    filename: str | None = None,
) -> list[Document]:
    """
    Retrieve the top-K candidate documents for *question* then rerank them
    down to top-N using FlashRank.

    Parameters
    ----------
    question : str          The user's query.
    user_id  : str          Used to scope results to the user's own documents.
    filename : str | None   Optional: restrict to a single uploaded file.

    Returns
    -------
    list[Document]  Top-N reranked documents (may be empty).
    """
    store = get_vector_store()

    filter_dict: dict = {"user_id": user_id}
    if filename:
        filter_dict["source"] = filename

    retriever = store.as_retriever(
        search_kwargs={"k": RETRIEVER_TOP_K, "filter": filter_dict}
    )
    candidates = retriever.invoke(question)

    if not candidates:
        logger.info("[Retriever] No candidates found for user '%s'.", user_id)
        return []

    # ── Rerank ────────────────────────────────────────────────────────────────
    passages = [
        {"id": i, "text": doc.page_content, "meta": doc.metadata}
        for i, doc in enumerate(candidates)
    ]
    rerank_request = RerankRequest(query=question, passages=passages)
    reranked = _ranker.rerank(rerank_request)

    top_docs = [
        Document(page_content=r["text"], metadata=r["meta"])
        for r in reranked[:RERANKER_TOP_N]
    ]
    logger.info(
        "[Retriever] %d candidates → %d reranked docs returned.",
        len(candidates),
        len(top_docs),
    )
    return top_docs
