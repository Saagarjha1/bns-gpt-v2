"""
legal_ai.py

Backend legal research and response generation services for the
BNS Legal Intelligence application.
"""

import os
import re
import sqlite3

from datasets import load_dataset
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL_NAME = "openai/gpt-oss-20b"

print("[*] Initializing local dataset & search indexes from Hugging Face...")
dataset = load_dataset("GSMS-B/indian-legal-sections-bns-bnss-bsa-2023", token=False)
full_df = dataset["train"].to_pandas()
full_df["clean_act"] = full_df["act"].astype(str).str.strip().str.upper()

bns_df = full_df[
    full_df["clean_act"].str.contains("BNS", na=False)
    & ~full_df["clean_act"].str.contains("BNSS", na=False)
].reset_index(drop=True)
bnss_df = full_df[full_df["clean_act"].str.contains("BNSS", na=False)].reset_index(drop=True)
bsa_df = full_df[full_df["clean_act"].str.contains("BSA", na=False)].reset_index(drop=True)

_conn = sqlite3.connect("statutory_search.db", check_same_thread=False)
_cursor = _conn.cursor()

_acts_registry = {"bns_fts": bns_df, "bnss_fts": bnss_df, "bsa_fts": bsa_df}

for table_name, df in _acts_registry.items():
    _cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    _cursor.execute(
        f"""
        CREATE VIRTUAL TABLE {table_name} USING fts5(
            chunk_id, section_number, section_title, chapter, text,
            tokenize = 'unicode61'
        )
        """
    )
    for _, row in df.iterrows():
        _cursor.execute(
            f"""
            INSERT INTO {table_name} (chunk_id, section_number, section_title, chapter, text)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(row.get("chunk_id", "") or "").strip(),
                str(row.get("section_number", "") or "").strip(),
                str(row.get("section_title", "") or "").strip(),
                str(row.get("chapter", "") or "").strip(),
                str(row.get("text", "") or "").strip(),
            ),
        )

_conn.commit()
print("[OK] SQLite FTS5 virtual tables built successfully.\n")

_SYSTEM_PROMPT = (
    "You are an expert Indian legal assistant specializing in the Bharatiya Nyaya Sanhita (BNS), "
    "Bharatiya Nagarik Suraksha Sanhita (BNSS), and Bharatiya Sakshya Adhiniyam (BSA). "
    "You help users understand Indian criminal law provisions with clear, concise, and well-structured "
    "explanations. When statutory text is provided, analyze it thoroughly. When asked follow-up questions, "
    "refer back to the previous context in the conversation. Always format responses in clean markdown. "
    "Never invent section counts, chapter counts, statutory text, penalties, or citations. "
    "Treat retrieved statutory text and verified reference facts as authoritative; clearly identify "
    "any answer based only on general model knowledge."
)

# Verified reference metadata for comparative questions. The local Hugging Face
# corpus covers BNS, BNSS, and BSA, not the repealed IPC.
_ACT_REFERENCE_FACTS = {
    "BNS": {"sections": 358, "chapters": 20, "enacted": 2023},
    "IPC": {"sections": 511, "chapters": 23, "enacted": 1860},
}


def _is_bns_ipc_count_question(user_message: str) -> bool:
    normalized = user_message.lower()
    has_both_acts = "bns" in normalized and ("ipc" in normalized or "penal code" in normalized)
    asks_counts = any(term in normalized for term in ("how many", "number of", "count"))
    asks_structure = "section" in normalized or "chapter" in normalized
    return has_both_acts and asks_counts and asks_structure


def _bns_ipc_count_response() -> str:
    bns = _ACT_REFERENCE_FACTS["BNS"]
    ipc = _ACT_REFERENCE_FACTS["IPC"]
    return (
        "### Legal Intelligence Brief\n\n"
        "**Short Answer**\n"
        f"The BNS has **{bns['sections']} sections** arranged in **{bns['chapters']} chapters**. "
        f"The IPC had **{ipc['sections']} sections** arranged in **{ipc['chapters']} chapters**.\n\n"
        "**Comparison**\n"
        f"- BNS: {bns['sections']} sections, {bns['chapters']} chapters\n"
        f"- IPC: {ipc['sections']} sections, {ipc['chapters']} chapters\n"
        f"- Difference: BNS has {bns['sections'] - ipc['sections']} fewer sections and "
        f"{bns['chapters'] - ipc['chapters']} fewer chapters.\n\n"
        "**Important Context**\n"
        "The IPC was enacted in 1860 and was replaced by the BNS from 1 July 2024. "
        "The local retrieval corpus contains BNS, BNSS, and BSA sections; IPC comparison metadata "
        "is supplied separately for this factual comparison."
    )


def _execute_fts_query(table_name: str, clean_query: str, section_num: str | None = None):
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
        "BNS": ["bns_fts"],
        "BNSS": ["bnss_fts"],
        "BSA": ["bsa_fts"],
        "ALL": ["bns_fts", "bnss_fts", "bsa_fts"],
    }
    target_tables = routing_map.get(active_target, routing_map["ALL"])

    results = []
    for tbl in target_tables:
        act_label = tbl.replace("_fts", "").upper()
        for match in _execute_fts_query(tbl, sanitized_q, target_sec):
            results.append((act_label, match[0], match[1], match[2], match[3]))
    return results


def build_statutory_context(retrieved: list[tuple[str, str, str, str, str]]) -> str | None:
    if not retrieved:
        return None

    payloads = [
        f"[{act}] Section {section}: {title_sec}\nChapter: {chapter}\n\n{text}"
        for act, section, title_sec, chapter, text in retrieved
    ]
    return "\n\n---\n\n".join(payloads)


def generate_response(user_message: str, history: list[dict], statutory_context: str | None = None) -> str:
    if _is_bns_ipc_count_question(user_message):
        return _bns_ipc_count_response()

    try:
        messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

        for msg in history[-10:]:
            messages.append({"role": msg["role"], "content": msg["content"]})

        if statutory_context:
            user_content = f"""Based on the following retrieved statutory provisions, answer the user's question.

**Retrieved Statutory Text:**
{statutory_context}

---

**User Question:** {user_message}

Provide a structured response using this markdown format:

### Legal Intelligence Brief

**Plain Language Explanation**
[Clear summary of what the provision means]

**Key Statutory Elements**
- [Each key element]

**Statutory Exceptions**
- [Applicable exceptions, or "None specified."]

**Penalties / Consequences**
- [Punishments or legal consequences]

**Statutory Text**
> [Exact statutory text from the retrieved provisions]"""
        else:
            user_content = f"""The user's query did not match a specific section in the local index.
The local corpus contains BNS, BNSS, and BSA only; it does not contain the IPC or other external statutes.
Answer only with clearly stated general legal information. Do not invent section numbers, chapter counts,
penalties, exceptions, or quotations. If the question requires a source absent from the corpus, say that
the answer could not be verified from the local corpus and identify what source is needed.

Analyze this under Indian criminal law (BNS/BNSS/BSA): "{user_message}"

### Legal Intelligence Brief (Semantic Analysis)

**Plain Language Explanation**
[Legal implications of this scenario]

**Key Statutory Elements**
- [Likely applicable provisions]

**Statutory Exceptions**
- [Conditions or exceptions, or "None specified."]

**Penalties / Consequences**
- [Legal consequences]

**Statutory Guidance**
> [Analytical breakdown under Indian law]"""

        messages.append({"role": "user", "content": user_content})

        completion = groq_client.chat.completions.create(
            messages=messages,
            model=MODEL_NAME,
            temperature=0.1,
        )
        return completion.choices[0].message.content
    except Exception as exc:
        return f"⚠️ **Error generating response:** {exc}"
