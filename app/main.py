from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
from authlib.integrations.starlette_client import OAuth
import os
import shutil

from app.rag_engine import process_and_store_document, query_rag

app = FastAPI(title="RAG Application")

app.add_middleware(SessionMiddleware, secret_key=os.getenv("SESSION_SECRET_KEY", "super-secret-key-123"))

oauth = OAuth()
oauth.register(
    name='azure',
    client_id=os.getenv("AZURE_CLIENT_ID", "dummy_client_id"),
    client_secret=os.getenv("AZURE_CLIENT_SECRET", "dummy_secret"),
    server_metadata_url=f'https://login.microsoftonline.com/{os.getenv("AZURE_TENANT_ID", "common")}/v2.0/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid profile email User.Read'},
)

async def get_current_user(request: Request):
    user = request.session.get('user')
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user

@app.get("/api/auth/login")
async def login(request: Request):
    # This must match the redirect URI added in Azure Portal
    redirect_uri = str(request.url_for('auth_callback'))
    # Azure explicitly blocks 127.0.0.1 for Web platforms, so we force it to localhost
    redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")
    return await oauth.azure.authorize_redirect(request, redirect_uri)

@app.get("/api/auth/callback", name="auth_callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(request)
        user = token.get('userinfo')
        if user:
            request.session['user'] = user
    except Exception as e:
        # Instead of failing, just redirect or show error
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")
    
    return RedirectResponse(url="/")

@app.get("/api/auth/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"user": user}

@app.get("/api/auth/logout")
async def logout(request: Request):
    request.session.clear()
    
    # Optional: Federated Logout to force Microsoft to clear its cookies too
    tenant_id = os.getenv("AZURE_TENANT_ID", "common")
    azure_logout_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/logout?post_logout_redirect_uri=http://localhost:8000/"
    
    return RedirectResponse(url=azure_logout_url)

# Mount the static folder at the root
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

class QueryRequest(BaseModel):
    question: str

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are allowed.")
    
    os.makedirs("temp_uploads", exist_ok=True)
    file_path = os.path.join("temp_uploads", file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks_indexed = process_and_store_document(file_path, file.filename)
        return {"filename": file.filename, "message": "File processed successfully", "chunks": chunks_indexed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/api/query")
async def query_endpoint(request: QueryRequest, user: dict = Depends(get_current_user)):
    try:
        answer, sources = query_rag(request.question)
        return {"answer": answer, "sources": sources}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # When running the file directly, start the Uvicorn server automatically
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)