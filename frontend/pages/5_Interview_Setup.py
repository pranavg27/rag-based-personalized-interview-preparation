import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError

ROLES = ["AI/ML Engineer", "Machine Learning Engineer", "Python Developer", "Java Developer", "Backend Developer",
         "Software Development Engineer", "Data Scientist", "Generative AI Engineer", "Agentic AI Engineer"]
SKILLS = ["DSA", "DBMS", "Operating Systems", "Computer Networks", "Java", "Python", "Machine Learning",
          "Deep Learning", "NLP", "RAG", "LangChain", "LangGraph", "FastAPI", "System Design"]
MODES = {"resume_based": "Resume-based", "skill_based": "Skill-based", "jd_based": "Job description-based",
         "mock_interview": "Mock interview (easy → hard, follow-up, behavioral)", "weak_topics": "Practice weak topics"}

setup_page("Interview Setup", "🎤")
api = require_login()
st.title("🎤 Configure interview")
prefs = api.get("/preferences")
keys = list(MODES)
mode = st.selectbox("Mode", keys, format_func=MODES.get, index=keys.index(st.session_state.get("setup_mode", "skill_based")))
role = st.selectbox("Target role", ROLES, index=ROLES.index(prefs["preferred_role"]) if prefs.get("preferred_role") in ROLES else 4)

resume_id, jd_id, skills = None, None, []
if mode in ("resume_based", "mock_interview"):
    resumes = api.get("/resume")
    opts = {r["id"]: r["filename"] for r in resumes}
    if opts:
        resume_id = st.selectbox("Resume", list(opts), format_func=opts.get, key="resume_sel")
    elif mode == "resume_based":
        st.warning("Upload a resume first.")
        st.page_link("pages/2_Resume_Upload.py", label="Upload resume", icon="📄")
if mode in ("skill_based", "mock_interview"):
    skills = st.multiselect("Skills", SKILLS, default=["Python"] if mode == "skill_based" else [])
if mode == "jd_based":
    jds = api.get("/documents", params={"doc_type": "job_description"})
    opts = {d["id"]: d["name"] for d in jds}
    if opts:
        jd_id = st.selectbox("Job description", list(opts), format_func=opts.get)
    else:
        st.warning("Add a job description first.")
        st.page_link("pages/3_Job_Description.py", label="Add job description", icon="💼")
    resumes = api.get("/resume")
    ropts = {"": "(none)", **{r["id"]: r["filename"] for r in resumes}}
    resume_id = st.selectbox("Compare against resume (optional)", list(ropts), format_func=ropts.get) or None
if mode == "weak_topics":
    st.info("Questions are prioritised from your long-term memory of weak topics.")

c1, c2 = st.columns(2)
n = c1.slider("Number of questions", 1, 20, int(prefs.get("default_questions") or 5))
diff = c2.selectbox("Starting difficulty (adapts automatically)", ["(auto)", "easy", "medium", "hard"])
st.markdown("**Human-in-the-loop checkpoints**")
h1, h2, h3 = st.columns(3)
hitl = {"skills": h1.checkbox("Review extracted skills", True), "plan": h2.checkbox("Approve interview plan", True),
        "feedback": h3.checkbox("Review each feedback", True)}

if st.button("Start interview", type="primary"):
    cfg = {"mode": mode, "role": role, "skills": skills, "max_questions": n, "hitl": hitl,
           "resume_id": resume_id, "jd_document_id": jd_id}
    if diff != "(auto)":
        cfg["difficulty"] = diff
    try:
        out = api.start_interview(**cfg)
        st.session_state.update(interview_id=out["interview"]["id"], run_stream=True)
        st.switch_page("pages/6_Live_Interview.py")
    except APIError as exc:
        show_error(exc)
