"""
database.py
-----------
Database initialisation.
- Creates the pgvector extension (if not present).
- Creates the user_documents tracking table (if not present).

Called once at application startup.
"""

import logging
import psycopg
from app.config import DATABASE_URL

logger = logging.getLogger(__name__)


def init_db() -> None:
    """Bootstrap the database schema required by this application."""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_documents (
                        id        SERIAL PRIMARY KEY,
                        user_id   TEXT NOT NULL,
                        filename  TEXT NOT NULL,
                        UNIQUE(user_id, filename)
                    );
                    """
                )
                conn.commit()
        logger.info("Database schema initialised successfully.")
    except Exception as exc:
        logger.warning("Could not initialise database schema: %s", exc)


def record_document(user_id: str, filename: str) -> None:
    """Insert a (user_id, filename) row; silently ignore duplicate uploads."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO user_documents (user_id, filename)
                VALUES (%s, %s)
                ON CONFLICT (user_id, filename) DO NOTHING;
                """,
                (user_id, filename),
            )
            conn.commit()


def list_documents(user_id: str) -> list[str]:
    """Return all filenames uploaded by *user_id*, newest first."""
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT filename FROM user_documents WHERE user_id = %s ORDER BY id DESC;",
                (user_id,),
            )
            return [row[0] for row in cur.fetchall()]
