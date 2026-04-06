from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
import os
import shutil

from app.rag_engine import process_and_store_document, query_rag
from app.auth import router as auth_router, get_current_user

app = FastAPI(title="RAG Application")

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-key-123"))

app.include_router(auth_router)

# Mount the static folder at the root
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

class QueryRequest(BaseModel):
    question: str
    filename: str = None

@app.get("/api/documents")
async def get_documents(user: dict = Depends(get_current_user)):
    user_id = user.get('preferred_username') or user.get('email') or user.get('oid') or "unknown_user"
    from app.rag_engine import connection_string_original
    import psycopg
    try:
        with psycopg.connect(connection_string_original) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT filename FROM user_documents WHERE user_id = %s ORDER BY id DESC;", (user_id,))
                rows = cur.fetchall()
                docs = [row[0] for row in rows]
                return {"documents": docs}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    user_id = user.get('preferred_username') or user.get('email') or user.get('oid') or "unknown_user"
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")
    
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks_indexed = process_and_store_document(file_path, file.filename, user_id)
        
        from app.rag_engine import connection_string_original
        import psycopg
        try:
            with psycopg.connect(connection_string_original) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO user_documents (user_id, filename) VALUES (%s, %s) ON CONFLICT (user_id, filename) DO NOTHING;",
                        (user_id, file.filename)
                    )
                    conn.commit()
        except Exception as db_e:
            print(f"Failed to record user document: {db_e}")

        return {"filename": file.filename, "message": "File processed successfully", "chunks": chunks_indexed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/query")
async def query_endpoint(request: QueryRequest, user: dict = Depends(get_current_user)):
    user_id = user.get('preferred_username') or user.get('email') or user.get('oid') or "unknown_user"
    try:
        answer, sources = query_rag(request.question, user_id, request.filename)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # When running the file directly, start the Uvicorn server automatically
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)