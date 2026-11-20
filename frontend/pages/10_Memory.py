import streamlit as st

from components.auth import require_login
from components.ui import setup_page

setup_page("Memory", "🧠")
api = require_login()
st.title("🧠 Long-term memory")
st.caption("What the system remembers about you across interviews. Delete anything you don't want used for personalisation.")
summary = api.get("/memory/summary")
st.info(summary["summary"])
kind = st.selectbox("Filter", ["(all)", "topic_stat", "mistake", "skill", "progress", "preferred_role",
                               "preferred_difficulty", "learning_pref", "progress_summary"])
rows = api.get("/memory", params={} if kind == "(all)" else {"kind": kind})
if not rows:
    st.write("Nothing stored yet.")
for r in rows:
    c1, c2 = st.columns([6, 1])
    c1.markdown(f"**{r['kind']}** · `{r['key'][:30]}` — {r['value']}")
    if c2.button("🗑️", key=r["id"]):
        api.delete(f"/memory/{r['id']}")
        st.rerun()
