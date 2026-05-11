"""
routes/query.py
---------------
API route for querying the RAG pipeline.
  POST /api/query
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.auth import get_current_user
from app.query import answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["query"])


class QueryRequest(BaseModel):
    question: str
    filename: str | None = None


def _get_user_id(user: dict) -> str:
    return (
        user.get("preferred_username")
        or user.get("email")
        or user.get("oid")
        or "unknown_user"
    )


@router.post("/query")
async def query_endpoint(
    request: QueryRequest,
    user: dict = Depends(get_current_user),
):
    """
    Answer a question using the RAG pipeline.
    Falls back to Tavily web search when the documents cannot help.

    Response fields:
      - answer      (str)         The answer text.
      - sources     (list[str])   File names (RAG) or URLs (Tavily).
      - source_type (str)         "rag" or "tavily".
    """
    user_id = _get_user_id(user)
    try:
        answer, sources, source_type = answer_question(
            request.question, user_id, request.filename
        )
        return {"answer": answer, "sources": sources, "source_type": source_type}
    except Exception as exc:
        logger.exception("[Query route] Unhandled error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))
