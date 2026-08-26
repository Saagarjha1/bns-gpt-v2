"use client";

import { useEffect, useState } from "react";
import { API_BASE_URL, apiFetch } from "@/lib/api";

type User = {
  id: string;
  name: string;
  email: string;
  avatar_url: string;
};

type SessionItem = {
  id: string;
  title: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
};

type Retrieval = {
  act: string;
  section: string;
  title: string;
  chapter: string;
  text: string;
};

const corpusOptions = ["All Acts", "BNS", "BNSS", "BSA"];

export default function ChatPage() {
  const [user, setUser] = useState<User | null>(null);
  const [sessions, setSessions] = useState<SessionItem[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [corpus, setCorpus] = useState("All Acts");
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [retrievals, setRetrievals] = useState<Retrieval[]>([]);
  const [error, setError] = useState("");

  useEffect(() => {
    void boot();
  }, []);

  async function boot() {
    try {
      const auth = await apiFetch<{ authenticated: boolean; user: User }>("/api/auth/session");
      setUser(auth.user);

      const sessionPayload = await apiFetch<{ sessions: SessionItem[] }>("/api/chat/sessions");
      setSessions(sessionPayload.sessions);

      if (sessionPayload.sessions[0]) {
        await loadSession(sessionPayload.sessions[0].id);
      }
    } catch {
      window.location.href = "/";
    } finally {
      setLoading(false);
    }
  }

  async function loadSession(sessionId: string) {
    const payload = await apiFetch<{ session: { id: string; messages: Message[] } }>(
      `/api/chat/sessions/${sessionId}`
    );
    setActiveSessionId(payload.session.id);
    setMessages(payload.session.messages);
    setRetrievals([]);
    setError("");
  }

  async function handleCreateSession() {
    try {
      const payload = await apiFetch<{ session_id: string }>("/api/chat/sessions", {
        method: "POST",
        body: JSON.stringify({})
      });
      const nextSession: SessionItem = { id: payload.session_id, title: "New Legal Brief" };
      setSessions((current) => [nextSession, ...current]);
      setActiveSessionId(payload.session_id);
      setMessages([]);
      setRetrievals([]);
      setError("");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not start a new session.");
    }
  }

  async function handleDeleteSession(sessionId: string) {
    if (!window.confirm("Delete this case history permanently?")) return;

    try {
      await apiFetch(`/api/chat/sessions/${sessionId}`, { method: "DELETE" });
      const remainingSessions = sessions.filter((session) => session.id !== sessionId);
      setSessions(remainingSessions);

      if (activeSessionId === sessionId) {
        if (remainingSessions[0]) {
          await loadSession(remainingSessions[0].id);
        } else {
          setActiveSessionId(null);
          setMessages([]);
          setRetrievals([]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not delete this case history.");
    }
  }

  async function handleSend() {
    const message = input.trim();
    if (!message || sending) return;

    const optimisticUserMessage: Message = { role: "user", content: message };
    setMessages((current) => [...current, optimisticUserMessage]);
    setInput("");
    setSending(true);
    setError("");

    try {
      const payload = await apiFetch<{
        session_id: string;
        message: Message;
        retrievals: Retrieval[];
      }>("/api/chat/query", {
        method: "POST",
        body: JSON.stringify({
          message,
          act_filter: corpus,
          session_id: activeSessionId
        })
      });

      setActiveSessionId(payload.session_id);
      setMessages((current) => [...current, payload.message]);
      setRetrievals(payload.retrievals);

      const sessionPayload = await apiFetch<{ sessions: SessionItem[] }>("/api/chat/sessions");
      setSessions(sessionPayload.sessions);
    } catch (err) {
      setMessages((current) => current.slice(0, -1));
      setInput(message);
      setError(err instanceof Error ? err.message : "Query could not be analyzed.");
    } finally {
      setSending(false);
    }
  }

  async function handleLogout() {
    await apiFetch("/auth/logout", { method: "POST" });
    window.location.href = "/";
  }

  if (loading) {
    return <main className="loading-shell">Loading legal workspace...</main>;
  }

  return (
    <main className="chat-page-shell">
      <div className="chat-photo-wash" />

      <section className="chat-topbar">
        <div>
          <div className="eyebrow-pill">BNS Legal Intelligence Desk</div>
          <h1>Indian Police and Criminal Law Research Workspace</h1>
          <p>BNS, BNSS, and BSA legal analysis with session memory, corpus targeting, and AI brief generation.</p>
        </div>
        <div className="status-pill-row">
          <span className="status-pill">Institutional Access</span>
          <span className="status-pill status-pill-live">Live Retrieval</span>
        </div>
      </section>

      <section className="workspace-grid">
        <aside className="sidebar-panel">
          <div className="sidebar-head">
            <div className="seal-mark small">AI</div>
            <div>
              <strong>Investigation Desk</strong>
              <span>Session control and case memory</span>
            </div>
          </div>

          {user ? (
            <div className="profile-card">
              <div className="profile-name">{user.name}</div>
              <div className="profile-email">{user.email}</div>
              <button className="secondary-button" onClick={handleLogout} type="button">
                Sign Out
              </button>
            </div>
          ) : null}

          <button className="primary-light-button" onClick={handleCreateSession} type="button">
            New Legal Brief
          </button>

          <div className="section-kicker">Case History</div>
          <div className="session-list">
            {sessions.map((session) => (
              <div className={`session-row ${session.id === activeSessionId ? "active" : ""}`} key={session.id}>
                <button
                  className="session-item"
                  onClick={() => void loadSession(session.id)}
                  type="button"
                >
                  {session.title}
                </button>
                <button
                  aria-label={`Delete ${session.title}`}
                  className="delete-session-button"
                  onClick={() => void handleDeleteSession(session.id)}
                  title="Delete case history"
                  type="button"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </aside>

        <section className="chat-main-panel">
          <div className="overview-grid">
            <article className="overview-card main">
              <div className="mini-kicker">Command Overview</div>
              <h2>Ask like an investigator. Read like a legal researcher.</h2>
              <p>
                Search offences, procedure, and evidence law with a polished institutional interface
                inspired by Indian police review desks, statute files, and courtroom research systems.
              </p>
            </article>

            <article className="overview-card side">
              <strong>Acts in scope</strong>
              <span>BNS, BNSS, and BSA criminal-law analysis.</span>
              <strong>Best query style</strong>
              <span>Section number, offence scenario, arrest question, or evidence issue.</span>
            </article>
          </div>

          <div className="corpus-bar">
            <label htmlFor="corpus">Target Legal Corpus</label>
            <select id="corpus" value={corpus} onChange={(e) => setCorpus(e.target.value)}>
              {corpusOptions.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          </div>

          <div className="messages-panel">
            {messages.length === 0 ? (
              <div className="empty-state">
                State a criminal law question, cite a section, or describe a police-investigation scenario under BNS, BNSS, or BSA.
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`message-card ${message.role === "user" ? "user" : "assistant"}`}
                >
                  <div className="message-role">{message.role === "user" ? "You" : "Legal AI"}</div>
                  <div className="message-content">{message.content}</div>
                </article>
              ))
            )}
          </div>

          {retrievals.length > 0 ? (
            <div className="retrieval-panel">
              <div className="section-kicker">Retrieved Provisions</div>
              <div className="retrieval-grid">
                {retrievals.map((item, index) => (
                  <article key={`${item.act}-${item.section}-${index}`} className="retrieval-card">
                    <strong>
                      {item.act} Section {item.section}
                    </strong>
                    <span>{item.title}</span>
                    <p>{item.chapter}</p>
                  </article>
                ))}
              </div>
            </div>
          ) : null}

          {error ? <div className="error-banner compact">{error}</div> : null}

          <div className="composer">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void handleSend();
                }
              }}
              placeholder="e.g. 'BNS Section 103', 'What constitutes murder under BNS?', 'What is the arrest procedure for a cognizable offence?'"
              rows={3}
            />
            <button className="send-button" onClick={() => void handleSend()} type="button" disabled={sending}>
              {sending ? "Analyzing..." : "Analyze Query"}
            </button>
          </div>
        </section>
      </section>
    </main>
  );
}
