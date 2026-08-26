-- ================================================================
--  BNS-GPT v2 — Supabase Database Setup
--  Run this SQL in: Supabase Dashboard → SQL Editor → New Query
-- ================================================================

-- 1. USERS TABLE
--    Stores one row per Google account that has signed in.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    google_id   TEXT UNIQUE NOT NULL,
    email       TEXT UNIQUE NOT NULL,
    name        TEXT,
    avatar_url  TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by google_id (used on every login)
CREATE INDEX IF NOT EXISTS idx_users_google_id ON public.users (google_id);


-- 2. CHAT SESSIONS TABLE
--    Each row is one conversation thread belonging to a user.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.chat_sessions (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    title       TEXT NOT NULL DEFAULT 'New Chat',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast retrieval of a user's sessions (sidebar query)
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON public.chat_sessions (user_id, created_at DESC);


-- 3. CHAT MESSAGES TABLE
--    Each row is one message (user or assistant) within a session.
-- ----------------------------------------------------------------
CREATE TABLE IF NOT EXISTS public.chat_messages (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES public.chat_sessions(id) ON DELETE CASCADE,
    user_id     UUID NOT NULL REFERENCES public.users(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast retrieval of messages in a session (chat history load)
CREATE INDEX IF NOT EXISTS idx_chat_messages_session_id ON public.chat_messages (session_id, created_at ASC);


-- ================================================================
--  Row Level Security (RLS)
--  Supabase enables RLS by default on new tables. Since we use the
--  server-side anon key (not user JWT tokens), we explicitly disable
--  RLS. The Python backend enforces access control in application code.
-- ================================================================

ALTER TABLE public.users         DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_sessions DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.chat_messages DISABLE ROW LEVEL SECURITY;
