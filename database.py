import os
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_or_create_user(google_id: str, email: str, name: str, avatar_url: str) -> dict:
    """Upsert user on Google OAuth login. Returns the user record dict."""
    existing = supabase.table("users").select("*").eq("google_id", google_id).execute()
    if existing.data:
        result = supabase.table("users").update({
            "email": email,
            "name": name,
            "avatar_url": avatar_url,
        }).eq("google_id", google_id).execute()
        return result.data[0]
    else:
        result = supabase.table("users").insert({
            "google_id": google_id,
            "email": email,
            "name": name,
            "avatar_url": avatar_url,
        }).execute()
        return result.data[0]


def get_user_by_id(user_id: str) -> dict | None:
    """Fetch a user record by their UUID primary key."""
    result = supabase.table("users").select("name, email, avatar_url").eq("id", user_id).execute()
    return result.data[0] if result.data else None


def create_chat_session(user_id: str, title: str = "New Chat") -> str:
    """Creates a new chat session and returns its UUID."""
    # Truncate title to keep sidebar tidy
    title = title[:50] + ("..." if len(title) > 50 else "")
    result = supabase.table("chat_sessions").insert({
        "user_id": user_id,
        "title": title,
    }).execute()
    return result.data[0]["id"]


def update_session_title(session_id: str, title: str):
    """Update the display title of a session."""
    title = title[:50] + ("..." if len(title) > 50 else "")
    supabase.table("chat_sessions").update({"title": title}).eq("id", session_id).execute()


def save_message(session_id: str, user_id: str, role: str, content: str):
    """Persist a single chat message (user or assistant) to the database."""
    supabase.table("chat_messages").insert({
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
    }).execute()


def get_user_sessions(user_id: str, limit: int = 30) -> list[tuple[str, str]]:
    """
    Returns a list of (title, session_id) tuples for a user, newest first.
    Used to populate the chat history sidebar.
    """
    result = (
        supabase.table("chat_sessions")
        .select("id, title, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(limit)
        .execute()
    )
    return [(s["title"], s["id"]) for s in result.data]


def get_session_messages(session_id: str) -> list[tuple[str, str]]:
    """
    Returns a list of (role, content) tuples for a session in chronological order.
    Used to restore a conversation when a user clicks a past session.
    """
    result = (
        supabase.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return [(m["role"], m["content"]) for m in result.data]


def delete_chat_session(session_id: str):
    """Delete all messages for a session, then delete the session itself."""
    supabase.table("chat_messages").delete().eq("session_id", session_id).execute()
    supabase.table("chat_sessions").delete().eq("id", session_id).execute()

