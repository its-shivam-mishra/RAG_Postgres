from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth
import os

router = APIRouter(prefix="/api/auth", tags=["auth"])

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

@router.get("/login")
async def login(request: Request):
    # This must match the redirect URI added in Azure Portal
    redirect_uri = str(request.url_for('auth_callback'))
    # Azure explicitly blocks 127.0.0.1 for Web platforms, so we force it to localhost
    redirect_uri = redirect_uri.replace("127.0.0.1", "localhost")
    return await oauth.azure.authorize_redirect(request, redirect_uri)

@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request):
    try:
        token = await oauth.azure.authorize_access_token(
            request, 
            claims_options={} 
        )
        user = token.get('userinfo')
        if user:
            request.session['user'] = user
    except Exception as e:
        # Instead of failing, just redirect or show error
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")
    
    return RedirectResponse(url="/")

@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"user": user}

@router.get("/logout")
async def logout(request: Request):
    request.session.clear()
    
    # Optional: Federated Logout to force Microsoft to clear its cookies too
    tenant_id = os.getenv("AZURE_TENANT_ID", "common")
    #azure_logout_url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/logout?post_logout_redirect_uri=http://localhost:8000/"
    
    return RedirectResponse(url="/")
