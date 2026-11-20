import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError

ROLES = ["AI/ML Engineer", "Machine Learning Engineer", "Python Developer", "Java Developer", "Backend Developer",
         "Software Development Engineer", "Data Scientist", "Generative AI Engineer", "Agentic AI Engineer"]

setup_page("Settings", "⚙️")
api = require_login()
st.title("⚙️ Settings")
p = api.get("/preferences")
with st.form("prefs"):
    role = st.selectbox("Preferred role", ROLES, index=ROLES.index(p["preferred_role"]) if p.get("preferred_role") in ROLES else 0)
    diffs = ["easy", "medium", "hard"]
    diff = st.selectbox("Preferred starting difficulty", diffs, index=diffs.index(p.get("preferred_difficulty", "medium")))
    n = st.slider("Default number of questions", 1, 20, int(p.get("default_questions") or 5))
    learn = st.text_area("Learning preferences", p.get("learning_pref", ""), placeholder="e.g. short answers, code examples")
    if st.form_submit_button("Save", type="primary"):
        try:
            api.put("/preferences", json={"preferred_role": role, "preferred_difficulty": diff,
                                          "default_questions": n, "learning_preferences": learn or None})
            st.success("Saved.")
        except APIError as exc:
            show_error(exc)
try:
    h = api.get("/health")
    st.subheader("Backend status")
    st.json(h)
except APIError as exc:
    show_error(exc)
