"""
config.py
---------
Centralised configuration.  All environment variables are read once here
and exposed as plain constants so the rest of the codebase never calls
os.getenv() directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL", "postgresql://postgres:12345@localhost:5432/vectordb"
)

# PGVector requires the psycopg driver prefix
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL_PGVECTOR: str = DATABASE_URL.replace(
        "postgresql://", "postgresql+psycopg://"
    )
else:
    DATABASE_URL_PGVECTOR: str = DATABASE_URL

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_KEY: str = os.getenv("AZURE_OPENAI_KEY", "")
AZURE_OPENAI_API_VERSION: str = os.getenv(
    "AZURE_OPENAI_API_VERSION", "2024-12-01-preview"
)
AZURE_OPENAI_CHAT_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4"
)
AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = os.getenv(
    "AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"
)

# ── Azure SSO ─────────────────────────────────────────────────────────────────
AZURE_CLIENT_ID: str = os.getenv("AZURE_CLIENT_ID", "")
AZURE_CLIENT_SECRET: str = os.getenv("AZURE_CLIENT_SECRET", "")
AZURE_TENANT_ID: str = os.getenv("AZURE_TENANT_ID", "common")
SESSION_SECRET_KEY: str = os.getenv(
    "SESSION_SECRET_KEY", "super-secret-key-123"
)

# ── Tavily Search ─────────────────────────────────────────────────────────────
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# ── RAG settings ──────────────────────────────────────────────────────────────
COLLECTION_NAME: str = "rag_docs"
CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 128
RETRIEVER_TOP_K: int = 20      # candidates fetched from vector store
RERANKER_TOP_N: int = 5        # documents kept after reranking
TAVILY_MAX_RESULTS: int = 5
