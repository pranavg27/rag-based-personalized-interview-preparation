import pandas as pd
import streamlit as st

from components.auth import require_login
from components.ui import setup_page

setup_page("History", "🕘")
api = require_login()
st.title("🕘 Interview history")
rows = api.get("/analytics/history", params={"limit": 100})
if not rows:
    st.info("No interviews yet.")
    st.stop()
st.dataframe(pd.DataFrame(rows)[["created_at", "mode", "role", "status", "difficulty", "answered", "avg_score"]],
             use_container_width=True, hide_index=True)
resumable = [r for r in rows if r["status"] in ("awaiting_input", "paused", "needs_retry")]
if resumable:
    st.subheader("Resume an interrupted interview")
    pick = st.selectbox("Interview", [r["id"] for r in resumable],
                        format_func=lambda x: next(f"{r['mode']} · {r['created_at'][:16]} · {r['status']}" for r in resumable if r["id"] == x))
    if st.button("Resume from last question", type="primary"):
        st.session_state["interview_id"] = pick
        st.switch_page("pages/6_Live_Interview.py")
