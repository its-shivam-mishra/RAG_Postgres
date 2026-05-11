"""
auth.py
-------
Azure AD / Microsoft SSO via OAuth 2.0 (OpenID Connect).

Routes
------
  GET /api/auth/login     — redirect to Microsoft login
  GET /api/auth/callback  — handle the token exchange after login
  GET /api/auth/me        — return the current user's profile
  GET /api/auth/logout    — clear the session and redirect to home
"""

import logging
import traceback

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from authlib.integrations.starlette_client import OAuth

from app.config import AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ── OAuth client setup ────────────────────────────────────────────────────────
oauth = OAuth()
oauth.register(
    name="azure",
    client_id=AZURE_CLIENT_ID,
    client_secret=AZURE_CLIENT_SECRET,
    server_metadata_url=(
        f"https://login.microsoftonline.com/{AZURE_TENANT_ID}"
        "/v2.0/.well-known/openid-configuration"
    ),
    client_kwargs={"scope": "openid profile email User.Read"},
)


# ── Dependency ────────────────────────────────────────────────────────────────
async def get_current_user(request: Request) -> dict:
    """FastAPI dependency — raises 401 if the user is not authenticated."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return user


# ── Routes ────────────────────────────────────────────────────────────────────
@router.get("/login")
async def login(request: Request):
    """Initiate the Azure AD login flow."""
    # Redirect 127.0.0.1 → localhost so session cookies survive the round-trip
    if "127.0.0.1" in request.headers.get("host", ""):
        return RedirectResponse(
            url=str(request.url).replace("127.0.0.1", "localhost")
        )
    redirect_uri = str(request.url_for("auth_callback"))
    return await oauth.azure.authorize_redirect(request, redirect_uri)


@router.get("/callback", name="auth_callback")
async def auth_callback(request: Request):
    """Exchange the authorisation code for tokens and store the user session."""
    try:
        token = await oauth.azure.authorize_access_token(
            request, claims_options={}
        )
        user = token.get("userinfo")
        if user:
            request.session["user"] = user
    except Exception as exc:
        traceback.print_exc()
        logger.error("OAuth callback failed: %s", exc)
        raise HTTPException(
            status_code=400, detail=f"Authentication failed: {exc}"
        )
    return RedirectResponse(url="/")


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return {"user": user}


@router.get("/logout")
async def logout(request: Request):
    """Clear the session and return to the home page."""
    request.session.clear()
    return RedirectResponse(url="/")
