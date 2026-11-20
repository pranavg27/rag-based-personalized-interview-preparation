import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError

ROLES = ["AI/ML Engineer", "Machine Learning Engineer", "Python Developer", "Java Developer", "Backend Developer",
         "Software Development Engineer", "Data Scientist", "Generative AI Engineer", "Agentic AI Engineer"]

setup_page("Resume", "📄")
api = require_login()
st.title("📄 Resume")
with st.form("resume_form"):
    f = st.file_uploader("Resume (PDF)", type=["pdf"])
    role = st.selectbox("Target job role", ["(none)"] + ROLES)
    if st.form_submit_button("Upload & analyse", type="primary") and f:
        try:
            with st.spinner("Parsing resume and extracting skills…"):
                api.upload("/resume/upload", f.name, f.getvalue(), "application/pdf",
                           {"target_role": role} if role != "(none)" else None)
            st.success("Resume analysed.")
        except APIError as exc:
            show_error(exc)

try:
    resumes = api.get("/resume")
except APIError as exc:
    show_error(exc)
    st.stop()
if not resumes:
    st.info("No resumes uploaded yet.")
for r in resumes:
    p = r["profile"]
    with st.expander(f"{r['filename']} · {r['uploaded_at'][:10]} · parsed by {r['extraction']}"):
        st.write(p.get("summary", ""))
        st.markdown("**Skills:** " + (", ".join(p.get("skills", [])) or "–"))
        if p.get("projects"):
            st.markdown("**Projects**")
            for pr in p["projects"]:
                st.write(f"• **{pr.get('name', '')}** — {pr.get('description', '')}")
        if p.get("experience"):
            st.markdown("**Experience**")
            for e in p["experience"]:
                st.write(f"• {e.get('role', '')} {e.get('company', '')} {e.get('description', '')}")
        if p.get("education"):
            st.markdown("**Education**")
            for e in p["education"]:
                st.write(f"• {e.get('degree', '')} {e.get('institution', '')}")
        st.caption("You can correct extracted skills during the interview (Human-in-the-loop step).")
        if st.button("Delete", key=f"del_{r['id']}"):
            api.delete(f"/resume/{r['id']}")
            st.rerun()
