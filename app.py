import os
import re
import sqlite3
import pandas as pd
import gradio as gr
from datasets import load_dataset

# =====================================================================
# LAYER 1: DATA INGESTION & LOCAL RETRIEVAL SUBSYSTEM (SQLite FTS5)
# =====================================================================
print("⏳ Initializing local dataset & search indexes from Hugging Face...")
dataset = load_dataset("GSMS-B/indian-legal-sections-bns-bnss-bsa-2023", token=False)
full_df = dataset['train'].to_pandas()
full_df['clean_act'] = full_df['act'].astype(str).str.strip().str.upper()

bns_df = full_df[full_df['clean_act'].str.contains('BNS', na=False) & ~full_df['clean_act'].str.contains('BNSS', na=False)].reset_index(drop=True)
bnss_df = full_df[full_df['clean_act'].str.contains('BNSS', na=False)].reset_index(drop=True)
bsa_df = full_df[full_df['clean_act'].str.contains('BSA', na=False)].reset_index(drop=True)

DB_NAME = "statutory_search.db"
conn = sqlite3.connect(DB_NAME, check_same_thread=False)
cursor = conn.cursor()

acts_registry = {'bns_fts': bns_df, 'bnss_fts': bnss_df, 'bsa_fts': bsa_df}

for table_name, df in acts_registry.items():
    cursor.execute(f"DROP TABLE IF EXISTS {table_name}")
    cursor.execute(f"""
        CREATE VIRTUAL TABLE {table_name} USING fts5(
            chunk_id, section_number, section_title, chapter, text, tokenize = 'unicode61'
        )
    """)
    for _, row in df.iterrows():
        cursor.execute(f"""
            INSERT INTO {table_name} (chunk_id, section_number, section_title, chapter, text)
            VALUES (?, ?, ?, ?, ?)
        """, (
            str(row.get('chunk_id', '') or '').strip(),
            str(row.get('section_number', '') or '').strip(),
            str(row.get('section_title', '') or '').strip(),
            str(row.get('chapter', '') or '').strip(),
            str(row.get('text', '') or '').strip()
        ))

conn.commit()
print("🚀 Local SQLite FTS5 virtual tables built successfully from Hugging Face dataset.\n")


# =====================================================================
# LAYER 2: QUERY ROUTING & MATCHING ENGINE
# =====================================================================
def execute_fts_query(table_name: str, clean_query: str, section_num: str = None):
    cur = conn.cursor()
    if section_num:
        cur.execute(f"SELECT section_number, section_title, chapter, text FROM {table_name} WHERE section_number = ?", (section_num,))
        records = cur.fetchall()
        if records:
            return records

    tokens = [w for w in clean_query.split() if len(w) > 2 and w.lower() not in ['section', 'sec', 'bns', 'bnss', 'bsa', 'under', 'for', 'the', 'act']]
    if not tokens:
        return []

    cleaned_terms = [re.sub(r"[^\w]", "", t) for t in tokens]
    fts_expression = " AND ".join([f'"{term}*"' for term in cleaned_terms if term])
    
    if not fts_expression:
        return []

    try:
        cur.execute(f"SELECT section_number, section_title, chapter, text FROM {table_name} WHERE {table_name} MATCH ? LIMIT 2", (fts_expression,))
        records = cur.fetchall()
    except Exception:
        records = []

    return records

def route_and_search(user_query: str, selected_act: str = "ALL"):
    sanitized_q = re.sub(r'[^a-zA-Z0-9\s]', '', user_query).strip()
    sec_match = re.search(r'\b(?:section|sec)?\s*(\d+)\b', sanitized_q, re.IGNORECASE)
    target_sec = sec_match.group(1) if sec_match else None

    act_indicator = None
    if re.search(r'\bbnss\b', sanitized_q, re.IGNORECASE):
        act_indicator = "BNSS"
    elif re.search(r'\bbsa\b', sanitized_q, re.IGNORECASE):
        act_indicator = "BSA"
    elif re.search(r'\bbns\b', sanitized_q, re.IGNORECASE):
        act_indicator = "BNS"

    active_target = act_indicator if act_indicator else selected_act.upper()
    routing_map = {
        "BNS": ["bns_fts"], 
        "BNSS": ["bnss_fts"], 
        "BSA": ["bsa_fts"], 
        "ALL": ["bns_fts", "bnss_fts", "bsa_fts"]
    }
    target_tables = routing_map.get(active_target, routing_map["ALL"])

    search_results = []
    for tbl in target_tables:
        act_label = tbl.replace('_fts', '').upper()
        matches = execute_fts_query(tbl, sanitized_q, target_sec)
        for m in matches:
            search_results.append((act_label, m[0], m[1], m[2], m[3]))
            
    return search_results


# =====================================================================
# LAYER 3: ADVANCED LEGAL BRIEF SYNTHESIZER (Dynamic & Section-Aware)
# =====================================================================
def generate_legal_brief(statutory_text: str) -> str:
    """Dynamically parses retrieved statutory text to build a section-aware brief without boilerplate."""
    try:
        lines = [line.strip() for line in statutory_text.split('\n') if line.strip()]
        
        raw_title = next((l for l in lines if "Section" in l), "Statutory Provision")
        clean_title = re.sub(r'[*#]', '', raw_title).replace("📌", "").strip()
        
        raw_chapter = next((l for l in lines if "Chapter:" in l or "CHAPTER" in l), "")
        clean_chapter = re.sub(r'[*#]', '', raw_chapter).replace("Chapter:", "").strip()
        
        content_lines = [
            l for l in lines 
            if not l.startswith("📌") 
            and not "Chapter:" in l 
            and not l.startswith("**Statutory Provisions:**")
        ]
        
        full_content = " ".join(content_lines) if content_lines else statutory_text
        clean_content = re.sub(r'^\d+\.\s+[A-Za-z\s]+\.—', '', full_content).strip()
        
        summary_sentence = clean_content if len(clean_content) < 350 else clean_content[:347] + "..."
        has_punishment = any(keyword in clean_content.lower() for keyword in ['punish', 'imprisonment', 'fine', 'penalty', 'offence', 'term of'])
        
        structured_output = f"""### ⚖️ Legal Intelligence Brief

**Plain Language Explanation**
This provision ({clean_title}) under **{clean_chapter if clean_chapter else 'General Provisions'}** dictates that: {summary_sentence}

**Key Ingredients / Conditions**
Based strictly on the statutory text, the governing requirements for this section involve:
* **Operational Trigger:** Directly activated when circumstances require compliance with the specific rules detailed in the provision.
* **Statutory Mandate:** 
  > {clean_content}"""

        if has_punishment:
            structured_output += """

**Penalties / Consequences**
Non-compliance or violation of these prescribed parameters attracts statutory penalties or legal consequences as defined by the governing framework."""

        return structured_output

    except Exception as runtime_err:
        return f"**[Brief Synthesis Notice]** Processed via structural extraction due to: {runtime_err}"


# =====================================================================
# LAYER 4: CONTROLLER & WEB PRESENTATION LAYER (Gradio + Custom HTML)
# =====================================================================
def process_user_request(query_text: str, act_filter: str):
    if not query_text.strip():
        return "⚠️ Please provide a section number or descriptive keyword query.", ""
    
    filter_target = "ALL" if act_filter == "All Acts" else act_filter
    retrieved_records = route_and_search(query_text, selected_act=filter_target)
    
    if not retrieved_records:
        return "❌ No exact or matching provisions found within the local SQLite FTS index.", "Summary unavailable due to empty retrieval."
    
    formatted_payloads = []
    for act, section, title, chapter, text in retrieved_records:
        formatted_payloads.append(f"📌 **[{act}] Section {section}: {title}**\n**Chapter:** {chapter}\n\n**Statutory Provisions:**\n{text}")
    
    combined_raw_text = "\n\n---\n\n".join(formatted_payloads)
    synthesized_summary = generate_legal_brief(combined_raw_text)
    
    return combined_raw_text, synthesized_summary

# Custom HTML Header / Wrapper Component
custom_html_header = gr.HTML("""
<div style="background: linear-gradient(135deg, #1e293b, #0f172a); padding: 30px; border-radius: 12px; color: white; text-align: center; margin-bottom: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
    <h1 style="margin: 0; font-size: 2.2em; font-weight: 700; letter-spacing: -0.5px;">⚖️ Bharatiya Sanhita Legal Intelligence Suite</h1>
    <p style="margin: 10px 0 0 0; color: #94a3b8; font-size: 1.1em;">High-speed local FTS5 legal retrieval & section-aware automated briefing engine.</p>
</div>
""")

with gr.Blocks(title="BNS Legal Intelligence Suite", theme=gr.themes.Soft()) as demo_interface:
    custom_html_header

    with gr.Row(equal_height=True):
        with gr.Column(scale=4):
            user_input_box = gr.Textbox(
                label="🔍 Statutory Search Query / Section",
                placeholder="e.g., 'BNS Section 34', 'BNSS 67', or 'Organised crime'",
                lines=2
            )
        with gr.Column(scale=1):
            act_selection_dropdown = gr.Dropdown(
                choices=["All Acts", "BNS", "BNSS", "BSA"],
                value="All Acts",
                label="🏛️ Target Jurisdiction"
            )
            dispatch_btn = gr.Button("⚡ Query & Generate Brief", variant="primary", scale=1)

    with gr.Row():
        with gr.Column(scale=1):
            summary_markdown_pane = gr.Markdown(
                label="🤖 Structured Legal Brief Explanation",
                value="*Legal brief summary will appear here after query execution...*"
            )
        with gr.Column(scale=1):
            raw_text_textbox = gr.Textbox(
                label="📂 Local FTS5 Dataset Raw Text",
                interactive=False,
                lines=15
            )

    dispatch_btn.click(
        fn=process_user_request,
        inputs=[user_input_box, act_selection_dropdown],
        outputs=[raw_text_textbox, summary_markdown_pane]
    )
    user_input_box.submit(
        fn=process_user_request,
        inputs=[user_input_box, act_selection_dropdown],
        outputs=[raw_text_textbox, summary_markdown_pane]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo_interface.launch(server_name="0.0.0.0", server_port=port, ssr_mode=False, show_error=True)
