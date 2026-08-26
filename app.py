"""
FastAPI backend for the BNS Legal Intelligence application.

This version exposes JSON APIs for a separate Next.js frontend.
"""

import os
from urllib.parse import quote

from authlib.integrations.starlette_client import OAuth
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from auth import create_session_token, decode_session_token
from database import (
    create_chat_session,
    get_or_create_user,
    get_session_messages,
    get_user_by_id,
    get_user_sessions,
    delete_chat_session,
    save_message,
    update_session_title,
)
from legal_ai import build_statutory_context, generate_response, route_and_search

load_dotenv()

SECRET_KEY = os.environ.get("SECRET_KEY", "change-this-secret-in-production")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000").rstrip("/")

if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
    print("[WARNING] GOOGLE_CLIENT_ID / GOOGLE_CLIENT_SECRET not set. OAuth will not work.")

app = FastAPI(title="BNS Legal Intelligence API")
app.add_middleware(SessionMiddleware, secret_key=SECRET_KEY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

oauth = OAuth()
oauth.register(
    name="google",
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
    client_kwargs={"scope": "openid email profile", "prompt": "select_account"},
)


class QueryRequest(BaseModel):
    message: str
    act_filter: str = "All Acts"
    session_id: str | None = None


def get_current_user_or_401(request: Request) -> tuple[str, dict]:
    token = request.cookies.get("app_session")
    user_id = decode_session_token(token) if token else None
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required.")

    user = get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=401, detail="User session is invalid.")
    return user_id, user


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return RedirectResponse(url=FRONTEND_URL, status_code=302)


@app.get("/auth/google")
async def login_with_google(request: Request):
    redirect_uri = str(request.base_url).rstrip("/") + "/auth/callback"
    try:
        return await oauth.google.authorize_redirect(request, redirect_uri)
    except Exception as exc:
        message = quote(str(exc))
        return RedirectResponse(
            url=f"{FRONTEND_URL}/?error=oauth&message={message}",
            status_code=302,
        )


@app.get("/auth/callback")
async def google_auth_callback(request: Request):
    try:
        token = await oauth.google.authorize_access_token(request)
    except Exception as exc:
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=oauth&message={exc}", status_code=302)

    user_info = token.get("userinfo")
    if not user_info:
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=userinfo", status_code=302)

    try:
        user = get_or_create_user(
            google_id=user_info["sub"],
            email=user_info.get("email", ""),
            name=user_info.get("name", "User"),
            avatar_url=user_info.get("picture", ""),
        )
    except Exception as exc:
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=db&message={exc}", status_code=302)

    session_token = create_session_token(user["id"])
    response = RedirectResponse(url=f"{FRONTEND_URL}/chat", status_code=302)
    response.set_cookie(
        key="app_session",
        value=session_token,
        httponly=True,
        samesite="lax",
        max_age=30 * 86400,
        secure=False,
    )
    return response


@app.post("/auth/logout")
async def logout():
    response = JSONResponse({"ok": True})
    response.delete_cookie(key="app_session")
    return response


@app.get("/api/auth/session")
async def auth_session(request: Request):
    token = request.cookies.get("app_session")
    user_id = decode_session_token(token) if token else None
    if not user_id:
        return JSONResponse({"authenticated": False}, status_code=401)

    user = get_user_by_id(user_id)
    if not user:
        return JSONResponse({"authenticated": False}, status_code=401)

    return {
        "authenticated": True,
        "user": {
            "id": user_id,
            "name": user.get("name", "User"),
            "email": user.get("email", ""),
            "avatar_url": user.get("avatar_url", ""),
        },
    }


@app.get("/api/chat/sessions")
async def list_chat_sessions(request: Request):
    user_id, _ = get_current_user_or_401(request)
    sessions = get_user_sessions(user_id)
    return {"sessions": [{"id": session_id, "title": title} for title, session_id in sessions]}


@app.get("/api/chat/sessions/{session_id}")
async def read_chat_session(session_id: str, request: Request):
    user_id, _ = get_current_user_or_401(request)
    sessions = dict((sid, title) for title, sid in get_user_sessions(user_id))
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    messages = get_session_messages(session_id)
    return {
        "session": {
            "id": session_id,
            "title": sessions[session_id],
            "messages": [{"role": role, "content": content} for role, content in messages],
        }
    }


@app.post("/api/chat/sessions")
async def create_session(request: Request):
    user_id, _ = get_current_user_or_401(request)
    session_id = create_chat_session(user_id, "New Legal Brief")
    return {"session_id": session_id}


@app.delete("/api/chat/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    user_id, _ = get_current_user_or_401(request)
    owned_session_ids = {sid for _, sid in get_user_sessions(user_id)}
    if session_id not in owned_session_ids:
        raise HTTPException(status_code=404, detail="Session not found.")

    delete_chat_session(session_id)
    return {"ok": True}


@app.post("/api/chat/query")
async def chat_query(payload: QueryRequest, request: Request):
    user_id, _ = get_current_user_or_401(request)

    message = payload.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = payload.session_id
    if not session_id:
        title = message[:50] + ("..." if len(message) > 50 else "")
        session_id = create_chat_session(user_id, title)

    owned_session_ids = {sid for _, sid in get_user_sessions(user_id)}
    if session_id not in owned_session_ids:
        raise HTTPException(status_code=404, detail="Session not found.")

    prior_messages = get_session_messages(session_id)
    history = [{"role": role, "content": content} for role, content in prior_messages]

    save_message(session_id, user_id, "user", message)
    if not prior_messages:
        update_session_title(session_id, message)

    filter_target = "ALL" if payload.act_filter == "All Acts" else payload.act_filter
    retrieved = route_and_search(message, selected_act=filter_target)
    statutory_context = build_statutory_context(retrieved)
    response = generate_response(message, history, statutory_context)

    save_message(session_id, user_id, "assistant", response)

    return {
        "session_id": session_id,
        "message": {"role": "assistant", "content": response},
        "retrievals": [
            {
                "act": act,
                "section": section,
                "title": title_sec,
                "chapter": chapter,
                "text": text,
            }
            for act, section, title_sec, chapter, text in retrieved
        ],
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
