"""
routes/documents.py
--------------------
API routes for document management:
  GET  /api/documents  — list the current user's uploaded files
  POST /api/upload     — upload + ingest a PDF or TXT file
"""

import os
import shutil
import logging
import traceback

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.auth import get_current_user
from app.database import record_document, list_documents
from app.document import ingest_document

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
UPLOAD_DIR = "temp_uploads"


def _get_user_id(user: dict) -> str:
    return (
        user.get("preferred_username")
        or user.get("email")
        or user.get("oid")
        or "unknown_user"
    )


@router.get("/documents")
async def get_documents(user: dict = Depends(get_current_user)):
    """Return the list of files the authenticated user has uploaded."""
    user_id = _get_user_id(user)
    try:
        docs = list_documents(user_id)
        return {"documents": docs}
    except Exception as exc:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    user: dict = Depends(get_current_user),
):
    """Upload a PDF or TXT file, ingest it into the vector store, and record it."""
    user_id = _get_user_id(user)

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are allowed.",
        )

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    try:
        # Save to disk temporarily
        with open(file_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        # Ingest into vector store
        chunks = ingest_document(file_path, file.filename, user_id)

        # Record in user_documents table
        try:
            record_document(user_id, file.filename)
        except Exception as db_exc:
            logger.warning("Failed to record document in DB: %s", db_exc)

        return {
            "filename": file.filename,
            "message": "File processed successfully",
            "chunks": chunks,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
