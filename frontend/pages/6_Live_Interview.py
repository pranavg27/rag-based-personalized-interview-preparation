import streamlit as st

from components.auth import require_login
from components.ui import score_pill, setup_page, show_error
from services.api_client import APIError
from services.sse_client import iter_sse

setup_page("Live Interview", "🎙️")
api = require_login()
st.title("🎙️ Live interview")

NODE_LABEL = {"generate_question": "Question", "feedback": "Feedback", "summary": "Summary", "chat": "Answer"}


def run_stream(iid: str) -> None:
    """Consume the SSE stream for the queued command, rendering tokens incrementally."""
    box = st.container(border=True)
    status = box.status("Working…", expanded=True)
    out, buf, node = status.empty(), "", None
    try:
        for ev, data in iter_sse(api, f"/interviews/{iid}/stream"):
            if ev == "node_start":
                status.update(label=data.get("message", "Working…"))
                buf, node = "", data.get("node")      # reset buffer: a retried node must not duplicate text
            elif ev == "status":
                status.update(label=data.get("message", ""))
            elif ev == "token":
                buf += data["text"]
                out.markdown(f"**{NODE_LABEL.get(data.get('node'), '')}**\n\n{buf}▌")
            elif ev == "error":
                status.update(label="Something went wrong", state="error")
                st.error(data["message"])
                return
        status.update(label="Done", state="complete", expanded=False)
    except APIError as exc:
        status.update(label="Stream failed", state="error")
        st.error(f"{exc.detail}. Reload the page; use *Retry* if the interview shows 'needs retry'.")
        return
    st.rerun()


def act(fn, *args, **kw) -> None:
    try:
        out = fn(*args, **kw)
        st.session_state["run_stream"] = bool(out and out.get("queued"))
        st.rerun()
    except APIError as exc:
        show_error(exc)


# ---- pick an interview ---------------------------------------------------------------
iid = st.session_state.get("interview_id")
if not iid:
    active = [i for i in api.get("/interviews") if i["status"] in ("awaiting_input", "paused", "needs_retry", "created")]
    if not active:
        st.info("No active interview.")
        st.page_link("pages/5_Interview_Setup.py", label="Start one", icon="🎤")
        st.stop()
    pick = st.selectbox("Resume an interview", [i["id"] for i in active],
                        format_func=lambda x: next(f"{i['mode']} · {i['created_at'][:16]} · {i['status']}" for i in active if i["id"] == x))
    if st.button("Continue", type="primary"):
        st.session_state["interview_id"] = pick
        st.rerun()
    st.stop()

if st.session_state.pop("run_stream", False):
    run_stream(iid)

try:
    v = api.interview(iid)
except APIError as exc:
    show_error(exc)
    st.stop()

state, wf, status = v["state"], v.get("waiting_for"), v["status"]
total = state.get("total_questions") or 0
top = st.columns([3, 1, 1])
top[0].caption(f"Mode: **{v['mode']}** · Role: **{v.get('role') or '–'}** · Status: **{status}** · Difficulty: **{state.get('difficulty') or '–'}**")
if total:
    st.progress(min(state.get("question_index", 0) / total, 1.0), text=f"Question {min(state.get('question_index', 0) + 1, total)} of {total}")

with st.sidebar:
    st.markdown("### Interview controls")
    if status == "paused":
        if st.button("▶️ Resume", use_container_width=True):
            act(lambda: api.post(f"/interviews/{iid}/resume"))
    elif status in ("awaiting_input", "needs_retry"):
        if st.button("⏸️ Pause", use_container_width=True):
            act(lambda: api.post(f"/interviews/{iid}/pause"))
    if status not in ("completed", "ended") and st.button("⏹️ End interview", use_container_width=True):
        act(api.end, iid)
    if st.button("Leave (keep for later)", use_container_width=True):
        st.session_state.pop("interview_id", None)
        st.switch_page("pages/1_Dashboard.py")

# ---- transcript ---------------------------------------------------------------------------
for msg in state.get("messages", []):
    role = "assistant" if msg["role"] == "assistant" else "user"
    with st.chat_message(role):
        tag = {"question": "❓ ", "feedback": "💬 ", "summary": "📝 "}.get(msg.get("kind"), "")
        st.markdown(tag + msg["content"])

# ---- state machine ---------------------------------------------------------------------------
if status == "paused":
    st.info("Interview paused. Press **Resume** in the sidebar to continue exactly where you left off.")
elif status == "needs_retry":
    st.warning("A step failed (for example the LLM was unavailable). Your progress is saved.")
    if st.button("🔁 Retry", type="primary"):
        act(api.decide, iid, "retry")
elif status in ("completed", "ended"):
    s = state.get("summary") or {}
    st.success("Interview finished.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Average score", f"{s.get('avg_score')}/10" if s.get("avg_score") is not None else "–")
    c2.metric("Answered", s.get("answered", 0))
    c3.metric("Skipped", s.get("skipped", 0))
    if s.get("topic_scores"):
        st.bar_chart(s["topic_scores"])
    st.write(s.get("narrative", ""))
    c1, c2 = st.columns(2)
    c1.page_link("pages/7_Feedback.py", label="Review detailed feedback", icon="🧾")
    if c2.button("Start a new interview"):
        st.session_state.pop("interview_id", None)
        st.switch_page("pages/5_Interview_Setup.py")
elif wf:
    kind = wf["kind"]
    if kind == "review_skills":
        st.subheader("✏️ Review extracted skills")
        st.caption("Remove wrong skills or add missing ones. Questions are generated from this list.")
        text = st.text_area("Skills (comma separated)", ", ".join(wf.get("skills", [])))
        if wf.get("jd_skills"):
            st.write("**Job description requires:** " + ", ".join(wf["jd_skills"]))
            cmp_ = (wf.get("jd_analysis") or {}).get("comparison") or {}
            if cmp_.get("match_score") is not None:
                st.write(f"Resume match: **{cmp_['match_score']}%** · Missing: {', '.join(cmp_.get('missing', [])) or '–'}")
        if st.button("Approve skills", type="primary"):
            act(api.decide, iid, "approve", skills=[s.strip() for s in text.split(",") if s.strip()])
    elif kind == "approve_plan":
        st.subheader("🗺️ Approve interview plan")
        if wf.get("weak_topics"):
            st.info("Prioritising weak topics from your history: " + ", ".join(wf["weak_topics"]))
        edited = st.data_editor(
            [{"topic": p["topic"], "type": p["type"], "difficulty": p.get("difficulty") or "adaptive"} for p in wf["plan"]],
            num_rows="dynamic", use_container_width=True, key="plan_editor",
            column_config={"type": st.column_config.SelectboxColumn(options=["technical", "project", "followup", "behavioral"]),
                           "difficulty": st.column_config.SelectboxColumn(options=["adaptive", "easy", "medium", "hard"])})
        note = st.text_input("Not happy? Tell the planner what to change (used when you click Regenerate)")
        c1, c2 = st.columns(2)
        if c1.button("✅ Approve & start", type="primary"):
            plan = [{"topic": r["topic"], "type": r["type"], "difficulty": None if r["difficulty"] == "adaptive" else r["difficulty"]}
                    for r in edited if r.get("topic")]
            act(api.decide, iid, "approve", plan=plan)
        if c2.button("🔄 Regenerate plan"):
            act(api.decide, iid, "reject", note=note or None)
    elif kind == "await_answer":
        q = wf["question"]
        st.markdown(f"<div class='q-card'><span class='muted'>{q['topic']} · {q['difficulty']} · {q['type']}</span><br><b>{q['text']}</b></div>",
                    unsafe_allow_html=True)
        ans = st.text_area("Your answer", height=180, key=f"ans_{q['id']}")
        extra = st.text_input("Extra context for the evaluator (optional)", key=f"ctx_{q['id']}")
        c1, c2 = st.columns([1, 1])
        if c1.button("Submit answer", type="primary", disabled=not ans.strip()):
            act(api.answer, iid, ans, extra or None)
        if c2.button("⏭️ Skip question"):
            act(api.decide, iid, "skip")
    elif kind == "review_feedback":
        ev = wf["evaluation"]
        st.subheader("🧾 Feedback")
        st.markdown(score_pill(ev["score"], ev["max_score"]), unsafe_allow_html=True)
        if ev.get("degraded"):
            st.warning("The evaluator was unavailable; this is a rough automatic score.")
        st.write(wf.get("feedback_text", ""))
        c1, c2, c3 = st.columns(3)
        c1.markdown("**Strengths**\n" + "\n".join(f"- {s}" for s in ev["strengths"]))
        c2.markdown("**Missing points**\n" + "\n".join(f"- {s}" for s in ev["missing_points"]))
        c3.markdown("**Improvements**\n" + "\n".join(f"- {s}" for s in ev["improvements"]))
        with st.expander("Ideal answer"):
            st.write(ev["ideal_answer"])
        if ev.get("follow_up_question"):
            st.info("Possible follow-up: " + ev["follow_up_question"])
        note = st.text_input("Add context / explain your answer to request re-evaluation")
        a, b, c = st.columns(3)
        if a.button("✅ Accept & continue", type="primary"):
            act(api.decide, iid, "accept")
        if b.button("🔁 Re-evaluate", disabled=not note.strip()):
            act(api.decide, iid, "reevaluate", note=note)
        if c.button("🚫 Reject (don't count)"):
            act(api.decide, iid, "reject")
else:
    st.info("Waiting…")
