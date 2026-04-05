from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import os
import shutil

from app.rag_engine import process_and_store_document, query_rag

app = FastAPI(title="RAG Application")

# Mount the static folder at the root
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

class QueryRequest(BaseModel):
    question: str

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")
    
    # Save the file temporarily
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # Process and index the document
        chunks_indexed = process_and_store_document(file_path, file.filename)
        return {"filename": file.filename, "message": "File processed successfully", "chunks": chunks_indexed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up the temp file
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/query")
async def query_endpoint(request: QueryRequest):
    try:
        answer, sources = query_rag(request.question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
