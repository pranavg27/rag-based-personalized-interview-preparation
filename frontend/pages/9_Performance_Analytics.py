import pandas as pd
import streamlit as st

from components.auth import require_login
from components.ui import setup_page, show_error
from services.api_client import APIError

setup_page("Analytics", "📈")
api = require_login()
st.title("📈 Performance analytics")
perf = api.get("/analytics/performance")
if not perf["answers_scored"]:
    st.info("Answer some questions to unlock analytics.")
    st.stop()
st.metric("Overall average", f"{perf['overall_avg']}/10", help=f"{perf['answers_scored']} scored answers")
c1, c2 = st.columns(2)
with c1:
    st.subheader("By topic")
    st.bar_chart(pd.DataFrame({k: v["avg"] for k, v in perf["by_topic"].items()}, index=["avg"]).T)
with c2:
    st.subheader("By difficulty")
    st.bar_chart(pd.DataFrame({k: v["avg"] for k, v in perf["by_difficulty"].items()}, index=["avg"]).T)

st.subheader("Learning recommendations")
for r in api.get("/analytics/recommendations"):
    st.markdown(f"- **{r['topic']}** — {r['reason']}. _{r['action']}_")

st.subheader("Practice questions on your weak topics")
if st.button("Generate practice questions"):
    try:
        with st.spinner("Asking the question MCP server…"):
            res = api.get("/analytics/practice-questions")
        for block in res["practice"]:
            st.markdown(f"**{block['topic']}**")
            for q in block["questions"]:
                st.write(f"• {q['question']}")
    except APIError as exc:
        show_error(exc)

st.subheader("AI progress report (via analytics MCP server)")
if st.button("Generate report"):
    try:
        with st.spinner("Generating…"):
            st.write(api.get("/analytics/report")["report"])
    except APIError as exc:
        show_error(exc)
