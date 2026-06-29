"""
DocMind — RAG Document Q&A
Streamlit interface for the full RAG pipeline.
"""

import os, sys, tempfile
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from document_processor import build_vectorstore, load_and_split
from qa_chain import build_qa_chain, ask
from utils import suggested_questions, export_chat, clean_text_preview
from config import SUPPORTED_MODELS, DEFAULT_MODEL, APP_TITLE, APP_ICON, APP_TAGLINE, MAX_FILE_SIZE_MB

# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="DocMind", page_icon=APP_ICON,
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
  .user-bubble {
    background:#1e3a5f; border-radius:12px 12px 4px 12px;
    padding:12px 16px; margin:8px 0; max-width:85%;
    margin-left:auto; color:#e6edf3;
  }
  .bot-bubble {
    background:#1c2128; border:1px solid #30363d;
    border-radius:12px 12px 12px 4px;
    padding:12px 16px; margin:8px 0; max-width:90%; color:#e6edf3;
  }
  .source-pill {
    display:inline-block; background:#21262d; border:1px solid #30363d;
    border-radius:20px; padding:2px 10px; font-size:12px;
    margin:2px; color:#58a6ff;
  }
  .stat-box { background:#1c2128; border:1px solid #30363d;
    border-radius:8px; padding:12px 16px; text-align:center; }
  .stat-num { font-size:22px; font-weight:700; color:#58a6ff; }
  .stat-lbl { font-size:11px; color:#8b949e; }
</style>
""", unsafe_allow_html=True)

# ── Session state ────────────────────────────────────────────
defaults = {
    "messages":    [],   # [{role, content, sources}]
    "history":     [],   # [(human, ai)] tuples for chain memory
    "chain":       None,
    "retriever":   None,
    "vectorstore": None,
    "doc_meta":    None,
    "doc_preview": None,
    "suggestions": [],
    "api_key":     "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"## {APP_ICON} DocMind")
    st.caption(APP_TAGLINE)
    st.divider()

    api_key = st.text_input("OpenAI API Key", type="password",
                             placeholder="sk-...", value=st.session_state.api_key,
                             help="Used only in this session — never stored.")
    if api_key:
        st.session_state.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

    st.divider()
    model = st.selectbox("Model", options=list(SUPPORTED_MODELS.keys()),
                          format_func=lambda m: SUPPORTED_MODELS[m])
    top_k = st.slider("Chunks retrieved per query", 3, 10, 5)
    st.divider()

    uploaded = st.file_uploader("Upload Document", type=["pdf", "txt"],
                                  help=f"Max {MAX_FILE_SIZE_MB} MB")

    if uploaded:
        file_mb = uploaded.size / (1024 * 1024)
        if file_mb > MAX_FILE_SIZE_MB:
            st.error(f"File too large ({file_mb:.1f} MB).")
        elif not st.session_state.api_key:
            st.warning("Enter your OpenAI API key first.")
        else:
            if st.button("Process Document", type="primary", use_container_width=True):
                with st.spinner("Extracting and embedding document…"):
                    try:
                        suffix = os.path.splitext(uploaded.name)[1]
                        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name

                        vs, meta = build_vectorstore(tmp_path, st.session_state.api_key)
                        chain, retriever = build_qa_chain(
                            vs, st.session_state.api_key, model=model, k=top_k)

                        chunks = load_and_split(tmp_path)
                        preview = " ".join(c.page_content for c in chunks[:3])

                        st.session_state.vectorstore = vs
                        st.session_state.chain       = chain
                        st.session_state.retriever   = retriever
                        st.session_state.doc_meta    = {**meta, "name": uploaded.name,
                                                         "size_mb": round(file_mb, 2)}
                        st.session_state.doc_preview = clean_text_preview(preview)
                        st.session_state.messages    = []
                        st.session_state.history     = []
                        st.session_state.suggestions = suggested_questions(preview)

                        os.unlink(tmp_path)
                        label = f"Ready! {meta.get('chunks','?')} chunks indexed."
                        if meta.get("cached"):
                            label = "⚡ Loaded from vector cache."
                        st.success(label)
                    except Exception as e:
                        st.error(f"Processing error: {e}")

    if st.session_state.doc_meta:
        st.divider()
        m = st.session_state.doc_meta
        st.markdown(f"**📄 `{m['name']}`**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"<div class='stat-box'><div class='stat-num'>{m.get('chunks','—')}</div><div class='stat-lbl'>Chunks</div></div>", unsafe_allow_html=True)
        with c2:
            st.markdown(f"<div class='stat-box'><div class='stat-num'>{m['size_mb']}MB</div><div class='stat-lbl'>Size</div></div>", unsafe_allow_html=True)

    st.divider()
    if st.session_state.messages:
        doc_name = st.session_state.doc_meta["name"] if st.session_state.doc_meta else "document"
        md = export_chat(st.session_state.messages, doc_name)
        st.download_button("⬇ Export Chat", data=md,
                            file_name="docmind_chat.md", mime="text/markdown",
                            use_container_width=True)
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.history  = []
            st.rerun()

# ── Main ─────────────────────────────────────────────────────
st.markdown(f"# {APP_ICON} {APP_TITLE}")
st.caption(APP_TAGLINE)

if not st.session_state.chain:
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("### 📤 Upload\nDrop any PDF or TXT in the sidebar.")
    with c2:
        st.markdown("### 🔑 Connect\nEnter your OpenAI API key.")
    with c3:
        st.markdown("### 💬 Ask\nAsk anything. Get cited answers.")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.info('📋 "Summarise the key findings"')
        st.info('📊 "What financial figures are mentioned?"')
        st.info('⚖️ "What are the termination clauses?"')
    with col2:
        st.info('🔬 "What methodology was used?"')
        st.info('📌 "List the main recommendations"')
        st.info('❓ "What does section 3 say about X?"')
    st.stop()

if st.session_state.doc_preview:
    with st.expander("📄 Document preview", expanded=False):
        st.caption(st.session_state.doc_preview)

# Suggested questions
if st.session_state.suggestions and not st.session_state.messages:
    st.markdown("#### Suggested questions")
    cols = st.columns(3)
    for i, q in enumerate(st.session_state.suggestions):
        with cols[i % 3]:
            if st.button(q, key=f"sug_{i}", use_container_width=True):
                st.session_state["_pending"] = q

st.markdown("---")

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-bubble'>🧑 {msg['content']}</div>",
                    unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🧠 {msg['content']}</div>",
                    unsafe_allow_html=True)
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                for s in msg["sources"]:
                    st.markdown(f"<span class='source-pill'>p.{s['page']} · {s['source']}</span>",
                                unsafe_allow_html=True)
                    st.caption(f"> {s['excerpt']}")

# Input
pending    = st.session_state.pop("_pending", None)
user_input = st.chat_input("Ask a question about your document…") or pending

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.markdown(f"<div class='user-bubble'>🧑 {user_input}</div>", unsafe_allow_html=True)

    with st.spinner("Searching document…"):
        try:
            result = ask(st.session_state.chain, st.session_state.retriever,
                         user_input, st.session_state.history)
            answer, sources = result["answer"], result["sources"]
        except Exception as e:
            answer, sources = f"Error: {e}", []

    st.session_state.history.append((user_input, answer))
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

    st.markdown(f"<div class='bot-bubble'>🧠 {answer}</div>", unsafe_allow_html=True)
    if sources:
        with st.expander(f"📎 {len(sources)} source(s)", expanded=True):
            for s in sources:
                st.markdown(f"<span class='source-pill'>p.{s['page']} · {s['source']}</span>",
                            unsafe_allow_html=True)
                st.caption(f"> {s['excerpt']}")
