import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError

setup_page("Job Description", "💼")
api = require_login()
st.title("💼 Job descriptions")
t1, t2 = st.tabs(["Paste text", "Upload file"])
with t1:
    with st.form("jd_text"):
        title = st.text_input("Title", placeholder="Backend Engineer @ Acme")
        text = st.text_area("Job description", height=220)
        if st.form_submit_button("Save", type="primary"):
            try:
                api.post("/documents/job-description", json={"title": title, "text": text})
                st.success("Saved.")
            except APIError as exc:
                show_error(exc)
with t2:
    f = st.file_uploader("PDF / DOCX / TXT", type=["pdf", "docx", "txt"], key="jd_file")
    if f and st.button("Upload job description"):
        try:
            api.upload("/documents/upload", f.name, f.getvalue(), "application/octet-stream", {"doc_type": "job_description"})
            st.success("Uploaded.")
        except APIError as exc:
            show_error(exc)

st.subheader("Saved job descriptions")
for d in api.get("/documents", params={"doc_type": "job_description"}):
    c1, c2 = st.columns([5, 1])
    c1.write(f"**{d['name']}** · {d['uploaded_at'][:10]}")
    if c2.button("Delete", key=d["id"]):
        api.delete(f"/documents/{d['id']}")
        st.rerun()
st.page_link("pages/5_Interview_Setup.py", label="Start a job-description interview", icon="🎤")
