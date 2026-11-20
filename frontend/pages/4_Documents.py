import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError
from services.sse_client import iter_sse

CATS = ["(auto-detect)", "DSA", "DBMS", "Operating Systems", "Computer Networks", "Java", "Python", "Machine Learning",
        "Deep Learning", "NLP", "RAG", "LangChain", "LangGraph", "FastAPI", "System Design", "General"]

setup_page("Documents", "📚")
api = require_login()
st.title("📚 Study materials")
up, ask = st.tabs(["Manage documents", "Ask your documents (RAG)"])

with up:
    with st.form("doc_form"):
        files = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
        cat = st.selectbox("Skill category", CATS)
        if st.form_submit_button("Upload & index", type="primary") and files:
            for f in files:
                try:
                    with st.spinner(f"Indexing {f.name}…"):
                        api.upload("/documents/upload", f.name, f.getvalue(), "application/octet-stream",
                                   {"skill_category": cat} if cat != "(auto-detect)" else None)
                    st.success(f"{f.name} indexed.")
                except APIError as exc:
                    show_error(exc)
    for d in api.get("/documents", params={"doc_type": "study"}):
        c1, c2 = st.columns([5, 1])
        c1.write(f"**{d['name']}** · {d['skill_category']} · {d['chunk_count']} chunks · {d['uploaded_at'][:10]}")
        if c2.button("Delete", key=d["id"]):
            api.delete(f"/documents/{d['id']}")
            st.rerun()

with ask:
    q = st.text_input("Question about your materials")
    cat_f = st.selectbox("Limit to category", ["(all)"] + CATS[1:], key="ask_cat")
    if st.button("Ask", type="primary") and q:
        body = {"question": q, "skill_category": None if cat_f == "(all)" else cat_f}
        box, text, sources, grounded = st.empty(), "", [], True
        try:
            for ev, data in iter_sse(api, "/chat/stream", "POST", body):
                if ev == "sources":
                    sources, grounded = data["sources"], data["grounded"]
                elif ev == "token":
                    text += data["text"]
                    box.markdown(text + "▌")
                elif ev == "error":
                    st.error(data["message"])
            box.markdown(text)
        except APIError as exc:
            show_error(exc)
        if not grounded:
            st.caption("⚠️ No relevant passages found in your documents; this is a general-knowledge answer.")
        for s in sources:
            st.caption(f"[{s['index']}] {s['document_name']} · p.{s['page']} · score {s['score']}")
