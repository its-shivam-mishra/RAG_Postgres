"""
document.py
-----------
Document ingestion pipeline:
  1. Load the file (PDF or TXT).
  2. Attach user metadata to each page.
  3. Split into chunks.
  4. Embed and store in the PGVector collection.
"""

import logging
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyMuPDFLoader, TextLoader

from app.config import (
    DATABASE_URL_PGVECTOR,
    COLLECTION_NAME,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)
from app.models import embeddings

logger = logging.getLogger(__name__)


def _get_vector_store() -> PGVector:
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL_PGVECTOR,
        use_jsonb=True,
    )


def ingest_document(file_path: str, filename: str, user_id: str) -> int:
    """
    Load, split, and store *file_path* in the vector store.

    Parameters
    ----------
    file_path : str   Absolute path to the file on disk.
    filename  : str   Original filename (stored as metadata).
    user_id   : str   Owner of the document.

    Returns
    -------
    int   Number of chunks indexed.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    if file_path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")

    docs = loader.load()

    # Attach user metadata so we can filter by user + file later
    for doc in docs:
        doc.metadata["source"] = filename
        doc.metadata["user_id"] = user_id

    # ── Split ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    # ── Store ─────────────────────────────────────────────────────────────────
    store = _get_vector_store()
    store.add_documents(chunks)

    logger.info(
        "Ingested '%s' for user '%s': %d chunks indexed.", filename, user_id, len(chunks)
    )
    return len(chunks)
