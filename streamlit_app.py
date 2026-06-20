import logging
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s - %(message)s")

st.set_page_config(
    page_title="MultiAgent RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

*, html, body { font-family: 'Inter', -apple-system, sans-serif !important; box-sizing: border-box; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ══════════════════ SIDEBAR ══════════════════ */
[data-testid="stSidebar"] {
    background: #171717 !important;
    border-right: 1px solid #2a2a2a !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div { color: #ececec !important; }

.sb-top {
    padding: 1rem 0.9rem 0.7rem;
    border-bottom: 1px solid #2a2a2a;
}
.sb-logo {
    font-size: 0.88rem;
    font-weight: 700;
    color: #fff !important;
    letter-spacing: -0.2px;
    display: flex;
    align-items: center;
    gap: 7px;
    margin-bottom: 0.8rem;
}

.new-chat-btn {
    display: flex;
    align-items: center;
    gap: 8px;
    width: 100%;
    background: #202020;
    border: 1px solid #3a3a3a;
    border-radius: 8px;
    padding: 8px 12px;
    color: #ececec !important;
    font-size: 0.83rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.15s;
    text-decoration: none;
}
.new-chat-btn:hover { background: #2a2a2a; }

/* Session history list */
.sb-section-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #666 !important;
    font-weight: 700;
    padding: 0.8rem 0.9rem 0.3rem;
    display: block;
}

/* Style ALL sidebar buttons to look like flat sidebar list items */
[data-testid="stSidebar"] button {
    background: transparent !important;
    border: none !important;
    border-radius: 6px !important;
    color: #c5c5c5 !important;
    font-size: 0.82rem !important;
    font-weight: 400 !important;
    text-align: left !important;
    padding: 6px 10px !important;
    margin: 1px 0 !important;
    transition: background 0.12s !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] button:hover {
    background: #2a2a2a !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] button:focus {
    outline: none !important;
    box-shadow: none !important;
}
/* New Chat button — slightly outlined to stand out */
[data-testid="stSidebar"] button:first-of-type,
[data-testid="stSidebar"] .new-chat-row button {
    border: 1px solid #3a3a3a !important;
    background: #202020 !important;
    color: #ececec !important;
    font-weight: 500 !important;
    margin-bottom: 4px !important;
}
/* Delete (✕) buttons — keep them small and dim */
[data-testid="stSidebar"] [data-testid="column"] button {
    border: none !important;
    background: transparent !important;
    color: #555 !important;
    font-size: 0.75rem !important;
    padding: 2px 6px !important;
}
[data-testid="stSidebar"] [data-testid="column"] button:hover {
    color: #ef4444 !important;
    background: transparent !important;
}

/* Docs section in sidebar */
.sb-docs { padding: 0 0.5rem; }
.doc-row-sb {
    background: #1e1e1e;
    border: 1px solid #2a2a2a;
    border-radius: 8px;
    padding: 7px 10px;
    margin-bottom: 5px;
    font-size: 0.76rem;
}
.doc-row-name { color: #d4d4d4 !important; font-weight: 500; word-break: break-all; }
.doc-row-meta { color: #555 !important; font-size: 0.67rem; margin-top: 1px; }

/* ══════════════════ MAIN CONTENT ══════════════════ */
.main-content {
    background: #212121;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
}

/* ── Hero (empty state) ── */
.hero {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    min-height: 75vh;
    text-align: center;
    padding: 2rem 1rem;
}
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 0.6rem;
    line-height: 1.15;
}
.hero-sub {
    font-size: 0.97rem;
    color: #8e8ea0;
    margin: 0 0 2.5rem;
    max-width: 400px;
    line-height: 1.55;
}
.hint-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.65rem;
    max-width: 520px;
    width: 100%;
}
.hint-card {
    background: #2a2a2a;
    border: 1px solid #3a3a3a;
    border-radius: 12px;
    padding: 0.85rem 1rem;
    text-align: left;
    cursor: default;
    transition: border-color 0.15s;
}
.hint-card:hover { border-color: #555; }
.hint-card-title { font-size: 0.82rem; color: #ececec; font-weight: 500; margin-bottom: 2px; }
.hint-card-sub   { font-size: 0.74rem; color: #666; }

/* ── Compact top bar (chat mode) ── */
.chat-topbar {
    background: #212121;
    border-bottom: 1px solid #2a2a2a;
    padding: 0.7rem 1.5rem;
    font-size: 0.88rem;
    font-weight: 600;
    color: #ececec;
    display: flex;
    align-items: center;
    gap: 8px;
    position: sticky;
    top: 0;
    z-index: 50;
}
.topbar-dot { width:8px; height:8px; background:#10b981; border-radius:50%; display:inline-block; }

/* ── Chat messages ── */
[data-testid="stChatMessageContent"] {
    background: #2a2a2a !important;
    border: 1px solid #3a3a3a !important;
    border-radius: 12px !important;
    color: #ececec !important;
    font-size: 0.90rem !important;
    line-height: 1.70 !important;
    padding: 0.85rem 1.1rem !important;
    box-shadow: none !important;
}

/* ── Badges ── */
.gb-high   { display:inline-block; background:#052e16; color:#86efac; border:1px solid #166534; border-radius:9px; padding:2px 10px; font-size:0.69rem; font-weight:700; }
.gb-mid    { display:inline-block; background:#1c1400; color:#fde68a; border:1px solid #92400e; border-radius:9px; padding:2px 10px; font-size:0.69rem; font-weight:700; }
.gb-low    { display:inline-block; background:#1c0000; color:#fca5a5; border:1px solid #991b1b; border-radius:9px; padding:2px 10px; font-size:0.69rem; font-weight:700; }
.web-badge    { display:inline-block; background:#052e1e; color:#34d399; border:1px solid #065f46; border-radius:9px; padding:2px 10px; font-size:0.69rem; font-weight:700; }

/* Safety refusal card */
.safety-card {
    background: #1a0a0a;
    border: 1px solid #7f1d1d;
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
}
.safety-title { font-size:0.92rem; font-weight:700; color:#f87171; margin-bottom:0.5rem; }
.safety-body  { font-size:0.83rem; color:#d1d5db; line-height:1.65; }
.safety-rules { margin:0.6rem 0 0 0; padding-left:1.1rem; color:#9ca3af; font-size:0.80rem; line-height:1.8; }
.safety-footer{ font-size:0.73rem; color:#4b5563; margin-top:0.7rem; }

.src-pill { display:inline-block; background:#1e293b; color:#7dd3fc; border:1px solid #1e3a5f; border-radius:7px; padding:2px 8px; font-size:0.68rem; font-weight:500; margin:2px 3px 2px 0; }
.meta-row { margin-top:8px; display:flex; flex-wrap:wrap; gap:5px; align-items:center; }

/* ── Source cards ── */
.src-card { background:#1e1e1e; border-left:3px solid #6366f1; border-radius:0 8px 8px 0; padding:9px 13px; margin-bottom:8px; }
.src-card-head { font-size:0.74rem; font-weight:700; color:#818cf8; margin-bottom:4px; }
.src-card-text { font-size:0.78rem; color:#9ca3af; line-height:1.55; }
.rel-sc { font-size:0.67rem; color:#4b5563; font-weight:400; margin-left:5px; }

/* ── Expander ── */
[data-testid="stExpander"] { border:1px solid #3a3a3a !important; border-radius:10px !important; background:#1e1e1e !important; }
[data-testid="stExpander"] summary { font-size:0.78rem !important; color:#818cf8 !important; font-weight:600 !important; }

/* ── Status widget ── */
[data-testid="stStatusWidget"] { border:1px solid #3a3a3a !important; border-radius:10px !important; background:#1e1e1e !important; }

/* ── Chat input ── */
[data-testid="stChatInput"] { background:#2a2a2a !important; border:1px solid #3a3a3a !important; border-radius:14px !important; }
[data-testid="stChatInput"] textarea { color:#ececec !important; font-size:0.91rem !important; background:#2a2a2a !important; }
[data-testid="stChatInput"]:focus-within { border-color:#6366f1 !important; }

/* ── Step labels ── */
.step-ok {
    display: block;
    color: #86efac;
    font-size: 0.79rem;
    line-height: 1.8;
    padding: 1px 0;
}

/* ── Status widget inner content spacing ── */
[data-testid="stStatusWidget"] > div { padding: 4px 0 2px !important; }
[data-testid="stStatusWidget"] [data-testid="stMarkdownContainer"] p {
    margin: 0 !important;
    padding: 1px 0 !important;
}

/* Gap between status block and streamed answer */
[data-testid="stChatMessageContent"] > div > div:has([data-testid="stStatusWidget"]) {
    margin-bottom: 10px !important;
}

::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:#1a1a1a; }
::-webkit-scrollbar-thumb { background:#3a3a3a; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

USER_AVATAR = "🧑"
BOT_AVATAR  = "🤖"


# ── Time grouping helper ──────────────────────────────────────────────────────

def _time_group(ts_str: str) -> str:
    try:
        dt  = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        diff = (now - dt).days
        if diff == 0:   return "Today"
        if diff == 1:   return "Yesterday"
        if diff <= 7:   return "Previous 7 Days"
        if diff <= 30:  return "Previous 30 Days"
        return dt.strftime("%B %Y")
    except Exception:
        return "Earlier"


# ── Session state init ────────────────────────────────────────────────────────

def _init():
    if "pipeline" not in st.session_state:
        from pipeline.rag_pipeline import RAGPipeline
        with st.spinner("Loading models..."):
            st.session_state.pipeline = RAGPipeline()
    if "db" not in st.session_state:
        from db.sqlite_manager import SQLiteManager
        from config import load_config
        cfg = load_config()
        st.session_state.db = SQLiteManager(
            cfg.get("storage", {}).get("sqlite_path", "./storage/history.db")
        )
    if "user_id" not in st.session_state:
        st.session_state.user_id = "user1"
    if "session_id" not in st.session_state:
        st.session_state.session_id = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "show_docs" not in st.session_state:
        st.session_state.show_docs = False


def _load_session(session_id: str):
    db = st.session_state.db
    st.session_state.session_id = session_id
    st.session_state.messages   = db.get_session_messages(session_id, limit=50)


def _new_chat():
    st.session_state.session_id = None
    st.session_state.messages   = []


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _sidebar():
    db      = st.session_state.db
    users   = db.get_users()
    umap    = {u["name"]: u["id"] for u in users}

    # Top: logo + user selector
    st.markdown('<div class="sb-top">', unsafe_allow_html=True)
    st.markdown('<div class="sb-logo">🤖 RAG Assistant</div>', unsafe_allow_html=True)

    cur_name = next(u["name"] for u in users if u["id"] == st.session_state.user_id)
    chosen   = st.selectbox("user", list(umap.keys()),
                             index=list(umap.keys()).index(cur_name),
                             label_visibility="collapsed")
    new_uid  = umap[chosen]
    if new_uid != st.session_state.user_id:
        st.session_state.user_id   = new_uid
        st.session_state.session_id = None
        st.session_state.messages  = []
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # New Chat button
    st.markdown('<div style="padding:0.6rem 0.5rem 0;">', unsafe_allow_html=True)
    if st.button("✏️  New Chat", use_container_width=True):
        _new_chat()
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    # Conversation history
    sessions = db.get_sessions(st.session_state.user_id, limit=30)
    if sessions:
        groups: dict = {}
        for s in sessions:
            grp = _time_group(s["created_at"])
            groups.setdefault(grp, []).append(s)

        for grp_name, items in groups.items():
            st.markdown(f'<span class="sb-section-label">{grp_name}</span>', unsafe_allow_html=True)
            for item in items:
                is_active = item["id"] == st.session_state.session_id
                css = "session-item active" if is_active else "session-item"
                label = item["title"] or "New Chat"
                # Use a button styled like the GPT session item
                if st.button(f"💬  {label}", key=f"sess_{item['id']}", use_container_width=True):
                    _load_session(item["id"])
                    st.rerun()

    # Documents toggle
    st.markdown("---")
    if st.button("📁  Documents", use_container_width=True):
        st.session_state.show_docs = not st.session_state.show_docs
        st.rerun()

    if st.session_state.show_docs:
        _docs_panel()


def _docs_panel():
    db = st.session_state.db
    st.markdown('<div class="sb-docs">', unsafe_allow_html=True)

    files = st.file_uploader("Upload PDF / TXT", type=["pdf","txt"],
                             accept_multiple_files=True, label_visibility="collapsed")
    if files:
        _ingest_uploads(files)

    docs = db.get_documents()
    if not docs:
        st.caption("No documents indexed yet.")
    else:
        for doc in docs:
            col_info, col_del = st.columns([5, 1])
            with col_info:
                st.markdown(
                    f'<div class="doc-row-sb">'
                    f'<div class="doc-row-name">📄 {doc["filename"]}</div>'
                    f'<div class="doc-row-meta">{doc["chunks_count"]} chunks</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_del:
                st.write("")
                if st.button("✕", key=f"d_{doc['filename']}"):
                    _delete_doc(doc["filename"])
                    st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def _ingest_uploads(uploaded_files):
    import tempfile
    db       = st.session_state.db
    pipeline = st.session_state.pipeline
    docs_dir = Path("./documents")
    docs_dir.mkdir(exist_ok=True)
    new_files = [f for f in uploaded_files if not db.document_exists(f.name)]
    if not new_files:
        st.info("Already indexed.")
        return
    for f in new_files:
        (docs_dir / f.name).write_bytes(f.getvalue())
    with st.status(f"Indexing {len(new_files)} file(s)...", expanded=True) as s:
        from config import load_config
        from ingest.doc_loader import DocumentLoader
        from ingest.text_splitter import DocumentSplitter
        from ingest.embedder import Embedder
        from providers import get_embeddings
        cfg = load_config()
        st.write("Reading...")
        # Temp dir with only the new files so existing docs are not re-embedded
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            for f in new_files:
                (tmp_path / f.name).write_bytes(f.getvalue())
            docs = DocumentLoader(tmp, cfg["ingest"]["supported_formats"]).load()
        st.write("Chunking...")
        chunks = DocumentSplitter(cfg["ingest"]["chunk_size"], cfg["ingest"]["chunk_overlap"]).split(docs)
        st.write("Embedding...")
        Embedder(get_embeddings(cfg),
                 cfg["vector_store"]["persist_path"],
                 cfg["vector_store"]["collection_name"]).embed_and_store(chunks)
        pipeline.reset_store()
        for fname in [f.name for f in new_files]:
            n = sum(1 for c in chunks if c.metadata.get("source") == fname)
            db.add_document(fname, n, st.session_state.user_id)
        s.update(label=f"Done — {len(chunks)} chunks.", state="complete")
    st.success(f"{len(new_files)} file(s) indexed.")


def _delete_doc(filename: str):
    deleted = st.session_state.pipeline.delete_from_store(filename)
    st.session_state.db.delete_document(filename)
    p = Path("./documents") / filename
    if p.exists():
        p.unlink()
    st.success(f"Removed ({deleted} chunks).")


# ── Badges & pills ────────────────────────────────────────────────────────────

def _badge(score: float) -> str:
    if score >= 0.70: return f'<span class="gb-high">✓ {score:.2f}</span>'
    if score >= 0.50: return f'<span class="gb-mid">⚠ {score:.2f}</span>'
    return f'<span class="gb-low">✗ {score:.2f}</span>'

def _pills(sources: list) -> str:
    return "".join(f'<span class="src-pill">{s}</span>' for s in sources)

def _render_meta(msg: dict):
    parts = []
    if msg.get("sources"):
        parts.append(_pills(msg["sources"]))
    sc = msg.get("grounding_score", 0.0)
    if sc > 0:
        parts.append(_badge(sc))
    if parts:
        st.markdown(f'<div class="meta-row">{"".join(parts)}</div>', unsafe_allow_html=True)


# ── Chat rendering ────────────────────────────────────────────────────────────

def _render_history():
    for msg in st.session_state.messages:
        avatar = USER_AVATAR if msg["role"] == "user" else BOT_AVATAR
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                _render_meta(msg)


_CATEGORY_LABELS = {
    "illegal_activity":  "Illegal Activity",
    "violence":          "Violence / Physical Harm",
    "hate_speech":       "Hate Speech / Discrimination",
    "self_harm":         "Self-Harm / Suicide",
    "explicit_content":  "Explicit / CSAM Content",
    "privacy_violation": "Privacy Violation / Doxxing",
    "policy_violation":  "Policy Violation",
}

_REFUSAL_GUIDELINES = """
<ul class="safety-rules">
  <li>Ask questions related to the uploaded documents or general knowledge.</li>
  <li>Avoid requests for illegal instructions, harmful how-tos, or content that targets individuals.</li>
  <li>Medical, legal, and security topics are welcome when asked in an educational context.</li>
  <li>Rephrase your question if you believe it was flagged incorrectly.</li>
</ul>
"""


def _render_safety_block(category: str):
    label = _CATEGORY_LABELS.get(category, "Policy Violation")
    st.markdown(f"""
    <div class="safety-card">
        <div class="safety-title">🚫 Request Not Permitted — {label}</div>
        <div class="safety-body">
            This assistant is designed to help with document Q&amp;A and general knowledge queries.
            Your message was flagged because it appears to request content that could be harmful,
            illegal, or in violation of our usage policy.
            {_REFUSAL_GUIDELINES}
        </div>
        <div class="safety-footer">
            If you think this is a mistake, try rephrasing your question with more context.
            Repeated violations may restrict your access.
        </div>
    </div>""", unsafe_allow_html=True)


def _handle_query(user_input: str):
    db       = st.session_state.db
    pipeline = st.session_state.pipeline
    user_id  = st.session_state.user_id

    # Create session on first message
    if not st.session_state.session_id:
        sid = db.create_session(user_id)
        st.session_state.session_id = sid
        db.update_session_title(sid, user_input)

    session_id = st.session_state.session_id

    db.save_message(user_id, "user", user_input, session_id=session_id)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(user_input)

    from schemas.state import AgentState
    state = AgentState(
        query=user_input,
        session_id=session_id,
        history=db.context_pairs(session_id, limit=20),
    )

    # ── Safety check before anything else ──
    state = pipeline.run_safety_check(state)
    if state.is_blocked:
        with st.chat_message("assistant", avatar=BOT_AVATAR):
            _render_safety_block(state.block_category)
        refusal = f"[Blocked — {_CATEGORY_LABELS.get(state.block_category, 'Policy Violation')}]"
        db.save_message(user_id, "assistant", refusal, session_id=session_id)
        st.session_state.messages.append({"role": "assistant", "content": refusal,
                                          "sources": [], "grounding_score": 0.0})
        return

    with st.chat_message("assistant", avatar=BOT_AVATAR):
        with st.status("Working on it...", expanded=True) as agent_status:
            agent_status.update(label="Understanding context...", state="running")
            state = pipeline.run_context(state)
            st.write("✅ Context ready")

            agent_status.update(label="Planning search...", state="running")
            state = pipeline.run_planner(state)
            st.write(f"✅ {len(state.plan)} steps planned")

            agent_status.update(label="Searching...", state="running")
            state = pipeline.run_search(state)
            if state.is_web_answer:
                n_web = len(state.web_sources)
                is_temporal = pipeline._is_temporal_query(state.query)
                reason_label = "Real-time query" if is_temporal else "Not in documents"
                st.write(f"🌐 {reason_label} → web search ({n_web} results)")
                agent_status.update(label="Summarising web results...", state="running")
            else:
                n = len(state.retrieved_chunks)
                st.write(f"✅ {n} chunk{'s' if n != 1 else ''} from documents (hybrid)")
                agent_status.update(label="Writing answer...", state="running")

        st.write("")
        full_text = st.write_stream(pipeline.stream_answer(state))
        state.answer = full_text
        state = pipeline.run_guard(state)

        if not state.passed_guard:
            st.warning(state.answer)

        # ── Meta row ──
        meta_html = ""
        if state.is_web_answer:
            meta_html += '<span class="web-badge">🌐 Web Search</span>&nbsp;'
        if state.sources:
            meta_html += _pills(state.sources)
        if not state.is_web_answer and state.grounding_score > 0:
            meta_html += "&nbsp;" + _badge(state.grounding_score)
        if meta_html:
            st.markdown(f'<div class="meta-row">{meta_html}</div>', unsafe_allow_html=True)

        # ── Source cards ──
        if state.is_web_answer and state.web_sources:
            with st.expander(f"🌐 Web sources ({len(state.web_sources)})"):
                for i, r in enumerate(state.web_sources, 1):
                    st.markdown(f"""
                    <div class="src-card" style="border-left-color:#10b981;">
                        <div class="src-card-head" style="color:#34d399;">
                            🌐 [{i}] {r.title}
                        </div>
                        <div class="src-card-text">{r.body[:380]}{"..." if len(r.body)>380 else ""}</div>
                        <div style="margin-top:5px;font-size:0.67rem;">
                            <a href="{r.url}" target="_blank" style="color:#6366f1;">{r.url[:70]}</a>
                        </div>
                    </div>""", unsafe_allow_html=True)
        elif state.retrieved_chunks:
            with st.expander(f"📄 Source excerpts ({len(state.retrieved_chunks)} chunks)"):
                for chunk in state.retrieved_chunks:
                    preview = chunk.content[:420] + ("..." if len(chunk.content) > 420 else "")
                    st.markdown(f"""
                    <div class="src-card">
                        <div class="src-card-head">
                            📄 {chunk.source} — Page {chunk.page}
                            <span class="rel-sc">relevance {chunk.score:.2f}</span>
                        </div>
                        <div class="src-card-text">{preview}</div>
                    </div>""", unsafe_allow_html=True)

    db.save_message(user_id, "assistant", state.answer,
                    session_id=session_id,
                    sources=state.sources,
                    grounding_score=state.grounding_score)
    st.session_state.messages.append({
        "role": "assistant", "content": state.answer,
        "sources": state.sources, "grounding_score": state.grounding_score,
    })


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    _init()

    with st.sidebar:
        _sidebar()

    has_msgs = bool(st.session_state.messages)

    if has_msgs:
        st.markdown("""
        <div class="chat-topbar">
            🤖 MultiAgent RAG Assistant
            <span class="topbar-dot"></span>
        </div>
        """, unsafe_allow_html=True)
        st.write("")
        _render_history()
    else:
        st.markdown("""
        <div class="hero">
            <h1 class="hero-title">MultiAgent RAG<br>Assistant</h1>
            <p class="hero-sub">Ask anything about your uploaded documents.<br>
               Powered by multi-agent AI with citations.</p>
            <div class="hint-grid">
                <div class="hint-card">
                    <div class="hint-card-title">📋 Summarise</div>
                    <div class="hint-card-sub">Key points from a document</div>
                </div>
                <div class="hint-card">
                    <div class="hint-card-title">🔍 Find details</div>
                    <div class="hint-card-sub">Locate specific information</div>
                </div>
                <div class="hint-card">
                    <div class="hint-card-title">⚖️ Compare</div>
                    <div class="hint-card-sub">Sections or concepts</div>
                </div>
                <div class="hint-card">
                    <div class="hint-card-title">📊 List out</div>
                    <div class="hint-card-sub">Requirements or steps</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    user_input = st.chat_input("Message MultiAgent RAG Assistant...")
    if user_input and user_input.strip():
        _handle_query(user_input.strip())


if __name__ == "__main__":
    main()
