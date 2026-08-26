"""
chat.py — Gradio Chatbot UI for BNS Legal Intelligence Suite

This module contains:
  - Layer 1: SQLite FTS5 data ingestion (same as original app.py)
  - Layer 2: Query routing & matching engine (same as original app.py)
  - Layer 3: Groq LLM response generation (upgraded with conversation history)
  - Layer 4: Gradio Blocks UI (upgraded to chatbot with sidebar session history)
"""

import os
import re
import sqlite3
import pandas as pd
import gradio as gr
from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq

from auth import decode_session_token
from database import (
    get_user_by_id,
    get_user_sessions,
    get_session_messages,
    create_chat_session,
    save_message,
    update_session_title,
    delete_chat_session,
)

load_dotenv()

# =====================================================================
# CONFIGURATION
# =====================================================================
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

# =====================================================================
# LAYER 1: DATA INGESTION & LOCAL RETRIEVAL SUBSYSTEM (SQLite FTS5)
# =====================================================================
print("[*] Initializing local dataset & search indexes from Hugging Face...")
dataset = load_dataset("GSMS-B/indian-legal-sections-bns-bnss-bsa-2023", token=False)
full_df = dataset["train"].to_pandas()
full_df["clean_act"] = full_df["act"].astype(str).str.strip().str.upper()

bns_df  = full_df[full_df["clean_act"].str.contains("BNS",  na=False) & ~full_df["clean_act"].str.contains("BNSS", na=False)].reset_index(drop=True)
bnss_df = full_df[full_df["clean_act"].str.contains("BNSS", na=False)].reset_index(drop=True)
bsa_df  = full_df[full_df["clean_act"].str.contains("BSA",  na=False)].reset_index(drop=True)

_conn = sqlite3.connect("statutory_search.db", check_same_thread=False)
_cursor = _conn.cursor()

_acts_registry = {"bns_fts": bns_df, "bnss_fts": bnss_df, "bsa_fts": bsa_df}

for table_name, df in _acts_registry.items():
    _cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    _cursor.execute(f"""
        CREATE VIRTUAL TABLE {table_name} USING fts5(
            chunk_id, section_number, section_title, chapter, text,
            tokenize = 'unicode61'
        )
    """)
    for _, row in df.iterrows():
        _cursor.execute(f"""
            INSERT INTO {table_name} (chunk_id, section_number, section_title, chapter, text)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(row.get("chunk_id",       "") or "").strip(),
            str(row.get("section_number", "") or "").strip(),
            str(row.get("section_title",  "") or "").strip(),
            str(row.get("chapter",        "") or "").strip(),
            str(row.get("text",           "") or "").strip(),
        ))

_conn.commit()
print("[OK] SQLite FTS5 virtual tables built successfully.\n")


# =====================================================================
# LAYER 2: QUERY ROUTING & MATCHING ENGINE
# =====================================================================
def _execute_fts_query(table_name: str, clean_query: str, section_num: str = None):
    cur = _conn.cursor()
    if section_num:
        cur.execute(
            f"SELECT section_number, section_title, chapter, text FROM {table_name} WHERE section_number = ?",
            (section_num,),
        )
        records = cur.fetchall()
        if records:
            return records

    stopwords = {"section", "sec", "bns", "bnss", "bsa", "under", "for", "the", "act"}
    tokens = [w for w in clean_query.split() if len(w) > 2 and w.lower() not in stopwords]
    if not tokens:
        return []

    cleaned_terms = [re.sub(r"[^\w]", "", t) for t in tokens]
    fts_expression = " AND ".join([f'"{term}*"' for term in cleaned_terms if term])
    if not fts_expression:
        return []

    try:
        cur.execute(
            f"SELECT section_number, section_title, chapter, text FROM {table_name} WHERE {table_name} MATCH ? LIMIT 2",
            (fts_expression,),
        )
        return cur.fetchall()
    except Exception:
        return []


def route_and_search(user_query: str, selected_act: str = "ALL"):
    sanitized_q = re.sub(r"[^a-zA-Z0-9\s]", "", user_query).strip()
    sec_match = re.search(r"\b(?:section|sec)?\s*(\d+)\b", sanitized_q, re.IGNORECASE)
    target_sec = sec_match.group(1) if sec_match else None

    act_indicator = None
    if re.search(r"\bbnss\b", sanitized_q, re.IGNORECASE):
        act_indicator = "BNSS"
    elif re.search(r"\bbsa\b", sanitized_q, re.IGNORECASE):
        act_indicator = "BSA"
    elif re.search(r"\bbns\b", sanitized_q, re.IGNORECASE):
        act_indicator = "BNS"

    active_target = act_indicator if act_indicator else selected_act.upper()
    routing_map = {
        "BNS":  ["bns_fts"],
        "BNSS": ["bnss_fts"],
        "BSA":  ["bsa_fts"],
        "ALL":  ["bns_fts", "bnss_fts", "bsa_fts"],
    }
    target_tables = routing_map.get(active_target, routing_map["ALL"])

    results = []
    for tbl in target_tables:
        act_label = tbl.replace("_fts", "").upper()
        for m in _execute_fts_query(tbl, sanitized_q, target_sec):
            results.append((act_label, m[0], m[1], m[2], m[3]))
    return results


# =====================================================================
# LAYER 3: LLM RESPONSE GENERATION (WITH CONVERSATION HISTORY)
# =====================================================================
_SYSTEM_PROMPT = (
    "You are an expert Indian legal assistant specializing in the Bharatiya Nyaya Sanhita (BNS), "
    "Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA). "
    "You help users understand Indian criminal law provisions with clear, concise, and well-structured "
    "explanations. When statutory text is provided, analyze it thoroughly. When asked follow-up questions, "
    "refer back to the previous context in the conversation. Always format responses in clean markdown. "
    "Use **bold** (double asterisks) for emphasis and headings — never use *italic* single asterisks for emphasis. "
    "Use ## for section headings, **bold** for key terms, and bullet points for lists."
)


def _generate_response(user_message: str, history: list, statutory_context: str = None) -> str:
    """
    Calls Groq with the full conversation history for multi-turn awareness.
    history: list of {"role": "user"|"assistant", "content": str} dicts.
    """
    try:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        # Inject conversation history (last 10 turns to stay within context limits)
        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        if statutory_context:
            user_content = f"""Based on the following retrieved statutory provisions, answer the user's question.

**Retrieved Statutory Text:**
{statutory_context}

---

**User Question:** {user_message}

Provide a structured response using this markdown format:

⚖️ Legal Intelligence Brief

Explanation
[Clear summary of what the provision means]

Key Statutory Elements
- [Each key element]

Statutory Exceptions
- [Applicable exceptions, or "None specified."]

Penalties / Consequences
- [Punishments or legal consequences]

Statutory Text
> [Exact statutory text from the retrieved provisions]"""
        else:
            user_content = f"""The user's query did not match a specific section in the local index.
Analyze this under Indian criminal law (BNS/BNSS/BSA): "{user_message}"

⚖️ Legal Intelligence Brief (Semantic Analysis)

Explanation
[Legal implications of this scenario]

Key Statutory Elements
- [Likely applicable provisions]

Statutory Exceptions
- [Conditions or exceptions, or "None specified."]

Penalties / Consequences
- [Legal consequences]

Statutory Guidance
> [Analytical breakdown under Indian law]"""

        messages.append({"role": "user", "content": user_content})

        completion = groq_client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.1,
        )
        return completion.choices[0].message.content

    except Exception as e:
        return f"⚠️ **Error generating response:** {e}"


# =====================================================================
# LAYER 4: GRADIO CHATBOT UI
# =====================================================================
_CHAT_CSS = """
:root {
    --page-bg: #f7f8fa;
    --page-bg-soft: #f1f4f7;
    --surface: rgba(255, 255, 255, 0.82);
    --surface-strong: rgba(255, 255, 255, 0.94);
    --surface-solid: #ffffff;
    --ink: #17212b;
    --ink-soft: #263746;
    --muted: #61717f;
    --navy: #17365d;
    --navy-strong: #214d87;
    --khaki: #b79c62;
    --gold: #cfb06f;
    --gold-soft: rgba(207, 176, 111, 0.18);
    --line: rgba(23, 33, 43, 0.09);
    --line-strong: rgba(23, 54, 93, 0.14);
    --shadow: 0 24px 60px rgba(23, 33, 43, 0.10);
    --shadow-soft: 0 12px 30px rgba(23, 33, 43, 0.06);
}

.gradio-container {
    font-family: 'Source Sans 3', sans-serif !important;
    color: var(--ink) !important;
    background:
        radial-gradient(circle at top left, rgba(33, 77, 135, 0.10), transparent 28%),
        linear-gradient(180deg, #fbfcfd 0%, #f2f5f8 100%) !important;
    position: relative;
    overflow: hidden;
}

.gradio-container,
.gradio-container * {
    color-scheme: light !important;
}

.gradio-container::before {
    content: "";
    position: fixed;
    inset: 0;
    background:
        linear-gradient(180deg, rgba(247,248,250,0.72), rgba(247,248,250,0.9)),
        url('/static/images/chat_bg.jpg') center top / cover no-repeat;
    opacity: 0.58;
    pointer-events: none;
    z-index: 0;
}

.gradio-container::after {
    content: "";
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(23, 54, 93, 0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(23, 54, 93, 0.03) 1px, transparent 1px);
    background-size: 56px 56px;
    mask-image: linear-gradient(180deg, rgba(0,0,0,0.55), transparent 90%);
    pointer-events: none;
    z-index: 0;
}

.gradio-container > .main,
.gradio-container .contain {
    position: relative;
    z-index: 1;
}

.gradio-container .block,
.gradio-container .form,
.gradio-container .wrap,
.gradio-container .panel,
.gradio-container .container,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-group {
    color: var(--ink) !important;
}

.command-header {
    display: flex;
    align-items: stretch;
    justify-content: space-between;
    gap: 20px;
    padding: 24px 26px;
    border-radius: 26px;
    background:
        linear-gradient(135deg, rgba(255,255,255,0.88), rgba(241,244,247,0.96)),
        linear-gradient(120deg, rgba(23,54,93,0.06), rgba(183,156,98,0.05));
    border: 1px solid var(--line-strong);
    box-shadow: var(--shadow);
    margin-bottom: 18px;
    position: relative;
    overflow: hidden;
}

.command-header::before {
    content: "";
    position: absolute;
    inset: 0 0 auto;
    height: 4px;
    background: linear-gradient(90deg, rgba(255,153,51,0.9), rgba(255,255,255,0.98) 52%, rgba(19,136,8,0.75));
}

.command-header::after {
    content: "";
    position: absolute;
    right: -64px;
    top: -70px;
    width: 220px;
    height: 220px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(207,176,111,0.18), transparent 70%);
}

.header-left {
    display: flex;
    align-items: center;
    gap: 16px;
    position: relative;
    z-index: 1;
}

.header-emblem {
    width: 54px;
    height: 54px;
    border-radius: 18px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(183,156,98,0.18), rgba(33,77,135,0.08));
    border: 1px solid rgba(183,156,98,0.2);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.75);
    flex-shrink: 0;
}

.header-emblem svg {
    width: 40px;
    height: 40px;
}

.header-titles h1 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.15rem;
    line-height: 0.96;
    letter-spacing: -0.02em;
    color: #122235;
    margin: 0;
}

.header-titles p {
    color: var(--muted);
    font-size: 14px;
    margin: 6px 0 0 0;
    line-height: 1.45;
}

.header-badges {
    display: flex;
    gap: 8px;
    align-items: flex-start;
    flex-wrap: wrap;
    justify-content: flex-end;
    position: relative;
    z-index: 1;
}

.status-badge {
    font-size: 11px;
    padding: 8px 12px;
    border-radius: 999px;
    font-weight: 700;
    letter-spacing: 0.06em;
    display: inline-flex;
    align-items: center;
    gap: 6px;
    text-transform: uppercase;
}

.badge-live {
    background: rgba(20, 138, 8, 0.08);
    border: 1px solid rgba(20, 138, 8, 0.14);
    color: #246f20;
}

.badge-secure {
    background: rgba(23,54,93,0.06);
    border: 1px solid rgba(23,54,93,0.10);
    color: var(--navy);
}

.badge-dot-green {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #2f8f2e;
    animation: pulse-dot 2s ease-in-out infinite;
}

@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.45; transform: scale(0.9); }
}

.sidebar-col {
    background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(247,248,250,0.9)) !important;
    border: 1px solid var(--line) !important;
    border-radius: 24px !important;
    padding: 16px 14px !important;
    min-height: 640px;
    box-shadow: var(--shadow-soft);
    position: relative;
    overflow: hidden;
}

.sidebar-col::before {
    content: "";
    position: absolute;
    inset: 0 0 auto;
    height: 3px;
    background: linear-gradient(90deg, rgba(255,153,51,0.75), rgba(255,255,255,0.95) 52%, rgba(19,136,8,0.6));
}

.sidebar-emblem {
    padding: 8px 4px 14px;
    margin-bottom: 12px;
    border-bottom: 1px solid rgba(23,54,93,0.08);
}

.sidebar-emblem-badge {
    display: inline-flex;
    align-items: center;
    gap: 12px;
    width: 100%;
}

.sidebar-emblem-mark {
    width: 42px;
    height: 42px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(183,156,98,0.16), rgba(33,77,135,0.08));
    border: 1px solid rgba(183,156,98,0.16);
}

.sidebar-emblem-mark svg {
    width: 28px;
    height: 28px;
}

.sidebar-emblem-copy {
    display: flex;
    flex-direction: column;
    gap: 2px;
}

.sidebar-emblem-text {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: var(--navy);
    font-weight: 700;
}

.sidebar-emblem-sub {
    font-size: 12px;
    color: var(--muted);
}

.user-profile-card {
    padding: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,244,247,0.95));
    border: 1px solid rgba(23,54,93,0.09);
    border-radius: 18px;
    margin-bottom: 14px;
    box-shadow: 0 10px 24px rgba(23,33,43,0.05);
}

.section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    color: var(--navy) !important;
    margin: 16px 0 8px 2px !important;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-label::before {
    content: '';
    width: 20px;
    height: 2px;
    border-radius: 999px;
    background: linear-gradient(90deg, var(--khaki), var(--gold));
}

.session-list {
    display: flex;
    flex-direction: column;
    gap: 6px;
    max-height: 440px;
    overflow-y: auto;
    padding-right: 2px;
}

.session-item {
    display: flex;
    align-items: center;
    gap: 6px;
    border-radius: 14px;
    padding: 10px 10px 10px 12px;
    font-size: 13px;
    color: var(--ink-soft);
    cursor: pointer;
    background: rgba(255,255,255,0.7);
    border: 1px solid rgba(23,54,93,0.06);
    transition: all 0.18s ease;
    box-shadow: 0 4px 12px rgba(23,33,43,0.03);
}

.session-item:hover {
    background: rgba(255,255,255,0.98);
    color: var(--ink);
    border-color: rgba(23,54,93,0.12);
    transform: translateY(-1px);
}

.session-item.active {
    background: linear-gradient(135deg, rgba(33,77,135,0.08), rgba(207,176,111,0.12));
    color: var(--navy);
    border-color: rgba(183,156,98,0.22);
}

.session-item-name {
    flex: 1;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    font-weight: 600;
    line-height: 1.3;
}

.session-delete-btn {
    flex-shrink: 0;
    width: 28px;
    height: 28px;
    border-radius: 8px;
    border: 1px solid rgba(200,50,50,0.15);
    background: rgba(220,50,50,0.07);
    color: #c0392b;
    cursor: pointer;
    font-size: 13px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.15s ease;
    opacity: 1;
    padding: 0;
    line-height: 1;
}

.session-delete-btn:hover {
    background: rgba(220,50,50,0.18);
    border-color: rgba(200,50,50,0.35);
    color: #a93226;
    transform: scale(1.08);
}

.session-list::-webkit-scrollbar { width: 4px; }
.session-list::-webkit-scrollbar-track { background: transparent; }
.session-list::-webkit-scrollbar-thumb { background: rgba(23,54,93,0.12); border-radius: 999px; }

.session-empty {
    font-size: 12px;
    color: var(--muted);
    text-align: center;
    padding: 18px 8px;
    line-height: 1.5;
}


.chat-col {
    background: linear-gradient(180deg, rgba(255,255,255,0.78), rgba(247,248,250,0.9)) !important;
    border: 1px solid var(--line) !important;
    border-radius: 28px !important;
    padding: 16px !important;
    box-shadow: var(--shadow);
    overflow: hidden;
}

.chat-col > div,
.chat-col .block,
.chat-col .form,
.chat-col .wrap,
.chat-col .panel,
.chat-col .container {
    background: transparent !important;
    color: var(--ink) !important;
}

.intel-banner {
    display: grid;
    grid-template-columns: minmax(0, 1.6fr) minmax(220px, 0.8fr);
    gap: 14px;
    margin-bottom: 14px;
}

.intel-banner-main,
.intel-banner-side {
    background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(241,244,247,0.95));
    border: 1px solid rgba(23,54,93,0.08);
    border-radius: 22px;
    padding: 18px 18px 16px;
}

.intel-banner-main {
    position: relative;
    overflow: hidden;
}

.intel-banner-main::after {
    content: "";
    position: absolute;
    right: -40px;
    top: -42px;
    width: 160px;
    height: 160px;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(207,176,111,0.16), transparent 70%);
}

.intel-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--navy);
    margin-bottom: 10px;
}

.intel-eyebrow::before {
    content: "";
    width: 18px;
    height: 2px;
    background: linear-gradient(90deg, var(--khaki), var(--gold));
    border-radius: 999px;
}

.intel-banner-main h2 {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    line-height: 0.95;
    letter-spacing: -0.02em;
    color: #122235;
    margin: 0 0 8px 0;
}

.intel-banner-main p {
    font-size: 14px;
    line-height: 1.55;
    color: #415465;
    margin: 0;
    max-width: 640px;
    font-weight: 600;
}

.intel-metrics {
    display: grid;
    gap: 10px;
}

.intel-metric {
    padding: 12px 14px;
    border-radius: 16px;
    background: rgba(255,255,255,0.82);
    border: 1px solid rgba(23,54,93,0.08);
}

.intel-metric strong {
    display: block;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--navy);
    margin-bottom: 4px;
}

.intel-metric span {
    font-size: 13px;
    line-height: 1.45;
    color: #3f5263;
    font-weight: 600;
}

.jurisdiction-dropdown {
    margin-bottom: 12px !important;
}

.jurisdiction-dropdown,
.jurisdiction-dropdown > div,
.jurisdiction-dropdown .wrap,
.jurisdiction-dropdown .form,
.jurisdiction-dropdown .container {
    background: transparent !important;
}

.jurisdiction-dropdown select,
.jurisdiction-dropdown input {
    background: #ffffff !important;
    border: 1px solid rgba(23,54,93,0.10) !important;
    color: #152738 !important;
    border-radius: 16px !important;
    min-height: 46px !important;
    box-shadow: 0 8px 18px rgba(23,33,43,0.04);
    font-weight: 600 !important;
}

.jurisdiction-dropdown label {
    color: var(--navy) !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
}

.chatbot {
    background: rgba(255,255,255,0.96) !important;
    border: 1px solid rgba(23,54,93,0.08) !important;
    border-radius: 24px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.95);
}

.chatbot,
.chatbot > div,
.chatbot .wrap,
.chatbot .message-wrap,
.chatbot .bubble-wrap,
.chatbot [class*="message"],
.chatbot [class*="bubble"],
.chatbot [class*="placeholder"],
.chatbot [class*="empty"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.99), rgba(245,248,251,0.98)) !important;
    color: #17212b !important;
}

.chatbot .message {
    border-radius: 18px !important;
}

.chatbot .user-message,
.chatbot .bot-message {
    font-size: 14px !important;
    line-height: 1.68 !important;
}

.chatbot .user-message {
    background: linear-gradient(135deg, rgba(33,77,135,0.10), rgba(33,77,135,0.06)) !important;
    color: var(--ink) !important;
}

.chatbot .bot-message {
    background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(241,244,247,0.92)) !important;
    color: var(--ink-soft) !important;
    border: 1px solid rgba(23,54,93,0.06) !important;
}

.chatbot .message p,
.chatbot .message li,
.chatbot .message strong,
.chatbot .message span,
.chatbot .placeholder,
.chatbot .placeholder p {
    color: #1f3243 !important;
}

.chatbot .placeholder,
.chatbot .placeholder p {
    font-weight: 600 !important;
}

.input-row {
    align-items: flex-end !important;
    gap: 10px !important;
    margin-top: 12px;
}

.input-row textarea {
    background: #ffffff !important;
    border: 1px solid rgba(23,54,93,0.10) !important;
    color: #132234 !important;
    border-radius: 18px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    box-shadow: 0 10px 22px rgba(23,33,43,0.05);
    transition: border-color 0.2s, box-shadow 0.2s !important;
}

.input-row textarea:focus {
    border-color: rgba(33,77,135,0.35) !important;
    box-shadow: 0 0 0 4px rgba(33,77,135,0.08) !important;
}

.input-row textarea::placeholder {
    color: #586b7d !important;
    opacity: 1 !important;
}

#new-chat-btn {
    background: linear-gradient(180deg, #ffffff, #f5f7fa) !important;
    border: 1px solid rgba(23,54,93,0.10) !important;
    color: var(--navy) !important;
    font-size: 13px !important;
    font-weight: 700 !important;
    border-radius: 16px !important;
    min-height: 42px !important;
    box-shadow: 0 10px 24px rgba(23,33,43,0.05) !important;
}

#new-chat-btn:hover {
    border-color: rgba(183,156,98,0.28) !important;
    background: linear-gradient(180deg, #ffffff, #eef3f8) !important;
}

#send-btn {
    border-radius: 18px !important;
    font-weight: 700 !important;
    background: linear-gradient(135deg, #17365d, #214d87) !important;
    color: #ffffff !important;
    border: none !important;
    font-size: 13px !important;
    letter-spacing: 0.04em;
    min-height: 48px !important;
    box-shadow: 0 16px 28px rgba(23,54,93,0.18) !important;
}

#send-btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 20px 34px rgba(23,54,93,0.24) !important;
}

::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: transparent;
}

::-webkit-scrollbar-thumb {
    background: rgba(23,54,93,0.14);
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: rgba(23,54,93,0.22);
}

.gradio-container .block {
    background: transparent !important;
    border: none !important;
}

.gradio-container [class*="dark"],
.gradio-container [style*="background-color: rgb(17, 24, 39)"],
.gradio-container [style*="background-color: rgb(15, 23, 42)"] {
    background: #f7f9fb !important;
    color: #17212b !important;
}

.gradio-container footer {
    display: none !important;
}

@media (max-width: 900px) {
    .command-header {
        flex-direction: column;
    }

    .header-badges {
        justify-content: flex-start;
    }

    .intel-banner {
        grid-template-columns: 1fr;
    }

    .chat-col,
    .sidebar-col {
        border-radius: 22px !important;
    }
}
"""



# ── Event Handlers ─────────────────────────────────────────
def load_user_data(request: gr.Request):
    """Called on page load. Reads session cookie and populates sidebar."""
    token = request.cookies.get("app_session")
    if not token:
        user_html = "<p style='color:#17365d; font-size:13px;'>Access required. <a href='/' style='color:#214d87; font-weight:700;'>Sign in</a></p>"
        return user_html, gr.update(value=build_session_list_html([])), None

    user_id = decode_session_token(token)
    if not user_id:
        user_html = "<p style='color:#17365d; font-size:13px;'>Session expired. <a href='/' style='color:#214d87; font-weight:700;'>Sign in again</a></p>"
        return user_html, gr.update(value=build_session_list_html([])), None

    try:
        user = get_user_by_id(user_id) or {}
        sessions = get_user_sessions(user_id)  # [(title, session_id), ...]

        name       = user.get("name", "User")
        email      = user.get("email", "")
        avatar_url = user.get("avatar_url", "")

        user_html = f"""
        <div class="user-profile-card">
            <div style="display:flex; align-items:center; gap:10px; margin-bottom:10px;">
                <img src="{avatar_url}"
                     style="width:34px; height:34px; border-radius:50%; object-fit:cover; border:2px solid rgba(183,156,98,0.22);"
                     onerror="this.style.display='none'" />
                <div>
                    <div style="font-weight:700; font-size:13px; color:#17212b; line-height:1.2;">{name}</div>
                    <div style="font-size:11px; color:#61717f; margin-top:1px;">{email}</div>
                </div>
            </div>
            <a href="/logout" style="display:block; text-align:center; padding:6px 10px;
               background:rgba(23,54,93,0.05); color:#17365d; border-radius:10px;
               font-size:12px; text-decoration:none; border:1px solid rgba(23,54,93,0.1);
               transition:all 0.15s; font-weight:700;">
               Sign Out
            </a>
        </div>
        """
        return user_html, gr.update(value=build_session_list_html(sessions)), user_id

    except Exception as e:
        return f"<p style='color:#f87171; font-size:12px;'>Error: {e}</p>", gr.update(value=build_session_list_html([])), None



def respond(message: str, history: list, act_filter: str, session_id, user_id, request: gr.Request):
    """
    Main chat handler. Performs FTS retrieval → LLM generation → DB persistence.
    Returns: (updated_history, session_id, updated_session_list_html, cleared_input)
    """
    if not message.strip():
        return history, session_id, gr.update(), ""

    # Fallback: re-read user_id from cookie if State was not set
    if not user_id:
        token = request.cookies.get("app_session")
        user_id = decode_session_token(token) if token else None

    is_new_session = not session_id

    # Create a new session on the first message
    if not session_id and user_id:
        title = message[:50] + ("..." if len(message) > 50 else "")
        session_id = create_chat_session(user_id, title)
    elif session_id and user_id and not history:
        # Session exists but no history yet (e.g. page reload) — update title on first real message
        update_session_title(session_id, message)

    # Persist the user's message
    if user_id and session_id:
        save_message(session_id, user_id, "user", message)

    # FTS5 retrieval
    filter_target = "ALL" if act_filter == "All Acts" else act_filter
    retrieved = route_and_search(message, selected_act=filter_target)

    # Build statutory context block if records found
    statutory_context = None
    if retrieved:
        payloads = [
            f"[{act}] Section {section}: {title_sec}\nChapter: {chapter}\n\n{text}"
            for act, section, title_sec, chapter, text in retrieved
        ]
        statutory_context = "\n\n---\n\n".join(payloads)

    # Generate LLM response (passes full conversation history for multi-turn)
    response = _generate_response(message, history, statutory_context)

    # Persist the assistant's response
    if user_id and session_id:
        save_message(session_id, user_id, "assistant", response)

    # Update chat history
    new_history = history + [
        {"role": "user",      "content": message},
        {"role": "assistant", "content": response},
    ]

    # Refresh the session sidebar HTML
    sessions = get_user_sessions(user_id) if user_id else []
    sidebar_html = build_session_list_html(sessions, session_id)

    return new_history, session_id, gr.update(value=sidebar_html), ""


def load_session_history(session_id, user_id):
    """
    Loads a past session's messages into the chat when the user clicks
    a session in the sidebar. Allows them to continue the conversation.
    """
    if not session_id or not user_id:
        return [], None, gr.update()

    messages = get_session_messages(session_id)
    history = [{"role": role, "content": content} for role, content in messages]

    # Refresh sidebar to highlight the newly active session
    sessions = get_user_sessions(user_id)
    sidebar_html = build_session_list_html(sessions, session_id)

    return history, session_id, gr.update(value=sidebar_html)



def start_new_chat():
    """Resets chat state for a fresh session (session_id will be created on first message)."""
    return [], None


def build_session_list_html(sessions: list[tuple[str, str]], active_id: str = None) -> str:
    """
    Renders the sidebar session list as HTML.
    sessions: list of (title, session_id) tuples.
    active_id: currently active session_id to highlight.
    Uses a <script> block for event delegation (more reliable than inline onclick in Gradio).
    """
    if not sessions:
        return "<div class='session-empty'>No case history yet.<br>Start a new Legal Brief to begin.</div>"

    items = []
    for title, sid in sessions:
        active_cls = " active" if sid == active_id else ""
        safe_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
        items.append(f"""
<div class="session-item{active_cls}" data-sid="{sid}">
  <span class="session-item-name" data-action="load" data-sid="{sid}">{safe_title}</span>
  <button class="session-delete-btn" data-action="delete" data-sid="{sid}" title="Delete this case">🗑</button>
</div>""")

    list_html = f"<div class='session-list' id='bns-session-list'>{''.join(items)}</div>"

    # Inline script for event delegation — fires each time HTML is updated
    script = """
<script>
(function() {
  function dispatch(elemId, value) {
    var wrap = document.getElementById(elemId);
    if (!wrap) return;
    var ta = wrap.querySelector('textarea') || wrap.querySelector('input');
    if (!ta) return;
    var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value') ||
                                 Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value');
    nativeInputValueSetter.set.call(ta, value);
    ta.dispatchEvent(new Event('input', {bubbles: true}));
    ta.dispatchEvent(new Event('change', {bubbles: true}));
  }

  function attachListeners() {
    var list = document.getElementById('bns-session-list');
    if (!list) return;
    list.addEventListener('click', function(e) {
      var btn = e.target.closest('[data-action]');
      if (!btn) return;
      var action = btn.getAttribute('data-action');
      var sid = btn.getAttribute('data-sid');
      if (action === 'delete') {
        e.stopPropagation();
        dispatch('del-session-box', sid);
      } else if (action === 'load') {
        dispatch('sel-session-box', sid);
      }
    });
  }

  // Run immediately and also after short delay for Gradio hydration
  attachListeners();
  setTimeout(attachListeners, 300);
  setTimeout(attachListeners, 800);
})();
</script>"""

    return list_html + script



def delete_session(session_id_to_delete: str, current_session_id, user_id):
    """
    Deletes a session from the DB, refreshes the sidebar HTML.
    If the deleted session was active, clears the chat.
    """
    if not session_id_to_delete or not user_id:
        sessions = get_user_sessions(user_id) if user_id else []
        return (
            gr.update(value=build_session_list_html(sessions, current_session_id)),
            current_session_id,
            [],
        )

    try:
        delete_chat_session(session_id_to_delete)
    except Exception:
        pass

    sessions = get_user_sessions(user_id)

    # If the deleted session was the active one, clear the chat
    if session_id_to_delete == current_session_id:
        new_session_id = None
        new_history = []
    else:
        new_session_id = current_session_id
        new_history = gr.update()  # no change to chat

    return (
        gr.update(value=build_session_list_html(sessions, new_session_id)),
        new_session_id,
        new_history,
    )


# ── Build Gradio Interface ─────────────────────────────────
_GRADIO_THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.blue,
    secondary_hue=gr.themes.colors.slate,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Source Sans 3"), "ui-sans-serif", "system-ui"],
)

# ── Ashoka Chakra SVG (reusable) ───────────────────────────
_ASHOKA_SVG_SMALL = """<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
<circle cx="50" cy="50" r="46" fill="none" stroke="rgba(196,163,90,0.3)" stroke-width="1.5"/>
<g stroke="rgba(196,163,90,0.5)" stroke-width="1.1" stroke-linecap="round">
<line x1="50" y1="50" x2="50" y2="8"/><line x1="50" y1="50" x2="60.9" y2="8.7"/>
<line x1="50" y1="50" x2="70.7" y2="12.6"/><line x1="50" y1="50" x2="78.5" y2="19.4"/>
<line x1="50" y1="50" x2="83.8" y2="28.5"/><line x1="50" y1="50" x2="86.2" y2="39.1"/>
<line x1="50" y1="50" x2="92" y2="50"/><line x1="50" y1="50" x2="86.2" y2="60.9"/>
<line x1="50" y1="50" x2="83.8" y2="71.5"/><line x1="50" y1="50" x2="78.5" y2="80.6"/>
<line x1="50" y1="50" x2="70.7" y2="87.4"/><line x1="50" y1="50" x2="60.9" y2="91.3"/>
<line x1="50" y1="50" x2="50" y2="92"/><line x1="50" y1="50" x2="39.1" y2="91.3"/>
<line x1="50" y1="50" x2="29.3" y2="87.4"/><line x1="50" y1="50" x2="21.5" y2="80.6"/>
<line x1="50" y1="50" x2="16.2" y2="71.5"/><line x1="50" y1="50" x2="13.8" y2="60.9"/>
<line x1="50" y1="50" x2="8" y2="50"/><line x1="50" y1="50" x2="13.8" y2="39.1"/>
<line x1="50" y1="50" x2="16.2" y2="28.5"/><line x1="50" y1="50" x2="21.5" y2="19.4"/>
<line x1="50" y1="50" x2="29.3" y2="12.6"/><line x1="50" y1="50" x2="39.1" y2="8.7"/>
</g>
<circle cx="50" cy="50" r="5" fill="rgba(196,163,90,0.2)" stroke="rgba(196,163,90,0.45)" stroke-width="1"/>
<circle cx="50" cy="50" r="2" fill="rgba(196,163,90,0.5)"/>
</svg>"""


with gr.Blocks(
    title="BNS Legal Command Center",
) as gradio_app:

    # ── Persistent State ──────────────────────────────────
    current_session_id = gr.State(None)
    user_id_state      = gr.State(None)

    # ── Load Google Fonts for header ──────────────────────
    gr.HTML("""
    <link rel="preconnect" href="https://fonts.googleapis.com"/>
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
    <link href="https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Source+Sans+3:wght@400;500;600;700&display=swap" rel="stylesheet"/>
    """)

    # ── Command Center Header ─────────────────────────────
    gr.HTML(f"""
    <div class="command-header">
        <div class="header-left">
            <div class="header-emblem">{_ASHOKA_SVG_SMALL}</div>
            <div class="header-titles">
                <h1>BNS Legal Intelligence Desk</h1>
                <p>Indian Police and Criminal Law Research Workspace for BNS, BNSS, and BSA analysis.</p>
            </div>
        </div>
        <div class="header-badges">
            <span class="status-badge badge-secure">&#128274; Institutional Access</span>
            <span class="status-badge badge-live">
                <span class="badge-dot-green"></span> Live Retrieval
            </span>
        </div>
    </div>
    """)

    # ── Main Layout ───────────────────────────────────────
    with gr.Row(equal_height=True):

        # ── Sidebar ───────────────────────────────────────
        with gr.Column(scale=1, min_width=220, elem_classes=["sidebar-col"]):

            # Sidebar emblem
            gr.HTML(f"""
            <div class="sidebar-emblem">
                <div class="sidebar-emblem-badge">
                    <div class="sidebar-emblem-mark">{_ASHOKA_SVG_SMALL}</div>
                    <div class="sidebar-emblem-copy">
                        <div class="sidebar-emblem-text">Investigation Desk</div>
                        <div class="sidebar-emblem-sub">Session control and case memory</div>
                    </div>
                </div>
            </div>
            """)

            user_info_html = gr.HTML("<p style='color:#61717f; font-size:12px;'>Loading workspace...</p>")

            new_chat_btn = gr.Button(
                "New Legal Brief",
                elem_id="new-chat-btn",
                size="sm",
            )

            gr.HTML("<div class='section-label'>Case History</div>")

            # Custom HTML session list (replaces gr.Radio — supports delete buttons)
            session_list_html = gr.HTML(
                value=build_session_list_html([]),
                elem_id="session-list-container",
            )

            # Hidden textboxes tunnelled from JS clicks
            selected_session_box = gr.Textbox(
                value="",
                visible=False,
                elem_id="sel-session-box",
            )
            delete_session_box = gr.Textbox(
                value="",
                visible=False,
                elem_id="del-session-box",
            )

        # ── Chat Area ─────────────────────────────────────
        with gr.Column(scale=4, elem_classes=["chat-col"]):
            gr.HTML("""
            <div class="intel-banner">
                <div class="intel-banner-main">
                    <div class="intel-eyebrow">Command Overview</div>
                    <h2>Ask like an investigator. Read like a legal researcher.</h2>
                    <p>
                        Search offences, procedure, and evidence law with a brighter institutional interface
                        inspired by Indian police review desks, legal records, and courtroom research systems.
                    </p>
                </div>
                <div class="intel-banner-side">
                    <div class="intel-metrics">
                        <div class="intel-metric">
                            <strong>Acts in scope</strong>
                            <span>BNS, BNSS, and BSA criminal-law analysis.</span>
                        </div>
                        <div class="intel-metric">
                            <strong>Best query style</strong>
                            <span>Section number, offence scenario, arrest question, or evidence issue.</span>
                        </div>
                    </div>
                </div>
            </div>
            """)

            act_dropdown = gr.Dropdown(
                choices=[
                    ("All Acts", "All Acts"),
                    ("BNS (Substantive Law)", "BNS"),
                    ("BNSS (Procedural Law)", "BNSS"),
                    ("BSA (Evidence Law)", "BSA"),
                ],
                value="All Acts",
                label="Target Legal Corpus",
                interactive=True,
                scale=1,
                elem_classes=["jurisdiction-dropdown"],
            )

            chatbot = gr.Chatbot(
                value=[],
                label="",
                height=500,
                show_label=False,
                render_markdown=True,
                placeholder="State a criminal law question, cite a section, or describe a police-investigation scenario under BNS, BNSS, or BSA.",
                avatar_images=(
                    None,
                    "https://api.dicebear.com/9.x/bottts-neutral/svg?seed=bns-legal-desk&backgroundColor=f1f4f7&textColor=17365d",
                ),
            )

            with gr.Row(elem_classes=["input-row"]):
                msg_box = gr.Textbox(
                    placeholder="e.g. 'BNS Section 103', 'What constitutes murder under BNS?', 'What is the arrest procedure for a cognizable offence?'",
                    lines=2,
                    scale=5,
                    show_label=False,
                    container=False,
                    autofocus=True,
                )
                send_btn = gr.Button(
                    "Analyze Query",
                    variant="primary",
                    scale=1,
                    min_width=110,
                    elem_id="send-btn",
                )

    # ── Wire up New Chat button ──────────────────────────────
    new_chat_btn.click(
        fn=start_new_chat,
        outputs=[chatbot, current_session_id],
    )

    # ── Page Load ─────────────────────────────────────────
    gradio_app.load(
        fn=load_user_data,
        outputs=[user_info_html, session_list_html, user_id_state],
    )

    # ── Send Message ──────────────────────────────────────
    _send_inputs  = [msg_box, chatbot, act_dropdown, current_session_id, user_id_state]
    _send_outputs = [chatbot, current_session_id, session_list_html, msg_box]

    send_btn.click(fn=respond, inputs=_send_inputs, outputs=_send_outputs)
    msg_box.submit(fn=respond, inputs=_send_inputs, outputs=_send_outputs)

    # ── Load Past Session (triggered by JS click on session name) ─────
    selected_session_box.change(
        fn=load_session_history,
        inputs=[selected_session_box, user_id_state],
        outputs=[chatbot, current_session_id, session_list_html],
    )

    # ── Delete Session (triggered by JS click on delete button) ───────
    delete_session_box.change(
        fn=delete_session,
        inputs=[delete_session_box, current_session_id, user_id_state],
        outputs=[session_list_html, current_session_id, chatbot],
    )
