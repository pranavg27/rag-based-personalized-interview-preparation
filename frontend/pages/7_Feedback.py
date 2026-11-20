import streamlit as st

from components.auth import require_login
from components.ui import score_pill, setup_page, show_error
from services.api_client import APIError

setup_page("Feedback", "🧾")
api = require_login()
st.title("🧾 Interview feedback")
rows = api.get("/interviews")
if not rows:
    st.info("No interviews yet.")
    st.stop()
default = st.session_state.get("interview_id") or rows[0]["id"]
ids = [r["id"] for r in rows]
iid = st.selectbox("Interview", ids, index=ids.index(default) if default in ids else 0,
                   format_func=lambda x: next(f"{r['mode']} · {r['created_at'][:16]} · {r['status']} · avg {r['avg_score']}" for r in rows if r["id"] == x))
try:
    answers = api.get(f"/interviews/{iid}/answers")
except APIError as exc:
    show_error(exc)
    st.stop()
if not answers:
    st.info("No answers recorded for this interview.")
for a in answers:
    ev = a.get("evaluation") or {}
    label = "⏭️ skipped" if a.get("skipped") else f"{a.get('score')}/10"
    with st.expander(f"Q{a['index'] + 1} · {a['topic']} · {a['difficulty']} · {label}"):
        st.markdown(f"**Question:** {a['question']}")
        if a.get("skipped"):
            continue
        st.markdown(f"**Your answer:** {a['answer']}")
        st.markdown(score_pill(ev.get("score"), ev.get("max_score", 10)), unsafe_allow_html=True)
        st.write(a.get("feedback_text", ""))
        c1, c2 = st.columns(2)
        c1.markdown("**Missing points**\n" + "\n".join(f"- {x}" for x in ev.get("missing_points", [])))
        c2.markdown("**Improvements**\n" + "\n".join(f"- {x}" for x in ev.get("improvements", [])))
        st.markdown(f"**Ideal answer:** {ev.get('ideal_answer', '')}")
        if a.get("feedback_decision") == "reject":
            st.caption("You rejected this feedback; it was excluded from your statistics.")
