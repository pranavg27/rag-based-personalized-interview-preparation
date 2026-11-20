import pandas as pd
import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError

setup_page("Dashboard", "📊")
api = require_login()
st.title("📊 Dashboard")
try:
    d = api.get("/analytics/dashboard")
except APIError as exc:
    show_error(exc)
    st.stop()

c1, c2, c3 = st.columns(3)
c1.metric("Total interviews", d["total_interviews"])
c2.metric("Completed", d["completed_interviews"])
c3.metric("Average score", f"{d['average_score']}/10" if d["average_score"] is not None else "–")

left, right = st.columns(2)
with left:
    st.subheader("💪 Strongest skills")
    for s in d["strongest_skills"] or []:
        st.write(f"**{s['topic']}** — {s['avg_score']}/10 ({s['attempts']} answers)")
    if not d["strongest_skills"]:
        st.caption("Answer a few questions per topic to see your strengths.")
with right:
    st.subheader("🎯 Weakest skills")
    for s in d["weakest_skills"] or []:
        st.write(f"**{s['topic']}** — {s['avg_score']}/10 ({s['attempts']} answers)")
    if not d["weakest_skills"]:
        st.caption("No weak topics detected yet.")

if d["score_trend"]:
    st.subheader("Score trend")
    st.line_chart(pd.DataFrame(d["score_trend"]).set_index("date")["avg_score"])

st.subheader("Recommended practice topics")
st.write(", ".join(d["recommended_topics"]) or "Run a few interviews to get recommendations.")
if d["recommended_topics"] and st.button("Practise weak topics now", type="primary"):
    st.session_state["setup_mode"] = "weak_topics"
    st.switch_page("pages/5_Interview_Setup.py")

st.subheader("Recent interviews")
if d["recent_history"]:
    df = pd.DataFrame(d["recent_history"])[["created_at", "mode", "role", "status", "avg_score", "answered"]]
    st.dataframe(df, use_container_width=True, hide_index=True)
else:
    st.info("No interviews yet.")
