"""
main.py
-------
FastAPI application entry point.

Responsibilities
----------------
  - Create the FastAPI app instance.
  - Register middleware (session).
  - Mount static files.
  - Include all routers (auth, documents, query).
  - Initialise the database at startup.
  - Serve the SPA root.
"""

import logging
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from app.config import SESSION_SECRET_KEY
from app.database import init_db
from app.auth import router as auth_router
from app.routes.documents import router as documents_router
from app.routes.query import router as query_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Application",
    description="Document Q&A powered by Azure OpenAI + pgvector + Tavily Search",
    version="2.0.0",
)

# ── Middleware ────────────────────────────────────────────────────────────────
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET_KEY)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(documents_router)
app.include_router(query_router)

# ── Static files ──────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def on_startup():
    logger.info("Initialising database schema…")
    init_db()
    logger.info("Application ready.")


# ── SPA root ──────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()


# ── Dev runner ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
