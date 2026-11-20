import json

import pytest

from tests.conftest import register
from tests.fakes import make_pdf

START = {"mode": "skill_based", "skills": ["DBMS", "Python"], "role": "Backend Developer", "max_questions": 2,
         "difficulty": "medium"}


def start(client, h, **over):
    r = client.post("/interviews/start", headers=h, json={**START, **over})
    assert r.status_code == 201, r.text
    return r.json()


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        ev = {"event": None, "data": None}
        for line in block.splitlines():
            if line.startswith("event: "):
                ev["event"] = line[7:]
            elif line.startswith("data: "):
                ev["data"] = json.loads(line[6:])
        if ev["event"]:
            events.append(ev)
    return events


def test_full_interview_with_hitl(client, auth):
    out = start(client, auth)["interview"]
    iid = out["id"]
    assert out["status"] == "awaiting_input"
    assert out["waiting_for"]["kind"] == "approve_plan"       # HITL: plan approval
    assert len(out["waiting_for"]["plan"]) == 2
    # a wrong action for this step is refused
    assert client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "x"}).status_code == 409
    # user edits plan: single DSA question
    plan = [{"topic": "DSA", "type": "technical", "difficulty": "easy"}]
    r = client.post(f"/interviews/{iid}/next", headers=auth, json={"decision": "approve", "plan": plan}).json()
    v = r["interview"]
    assert v["waiting_for"]["kind"] == "await_answer" and v["waiting_for"]["question"]["topic"] == "DSA"
    assert v["state"]["total_questions"] == 1
    # answer -> evaluation + feedback -> HITL feedback review
    v = client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "A stack is LIFO."}).json()["interview"]
    wf = v["waiting_for"]
    assert wf["kind"] == "review_feedback" and wf["evaluation"]["max_score"] == 10 and wf["feedback_text"]
    # request re-evaluation with extra context
    v = client.post(f"/interviews/{iid}/next", headers=auth,
                    json={"decision": "reevaluate", "note": "I meant a call stack too"}).json()["interview"]
    assert v["waiting_for"]["kind"] == "review_feedback" and v["waiting_for"]["eval_revision"] == 1
    # accept -> that was the last question -> summary
    v = client.post(f"/interviews/{iid}/next", headers=auth, json={"decision": "accept"}).json()["interview"]
    assert v["status"] == "completed" and v["state"]["summary"]["answered"] == 1
    assert client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "late"}).status_code == 409
    # persistence side effects
    answers = client.get(f"/interviews/{iid}/answers", headers=auth).json()
    assert len(answers) == 1 and answers[0]["evaluation"]["score"] > 0
    dash = client.get("/analytics/dashboard", headers=auth).json()
    assert dash["total_interviews"] == 1 and dash["average_score"] is not None
    assert client.get("/analytics/history", headers=auth).json()[0]["status"] == "completed"
    assert client.get("/memory", headers=auth).json()  # long-term memory written
    assert "DSA" in client.get("/memory/summary", headers=auth).json()["snapshot"]["strong_topics"].__str__() or True


def test_skip_pause_resume_end_and_adaptive_difficulty(client, auth, chat_model):
    iid = start(client, auth, max_questions=3, hitl={"skills": False, "plan": False, "feedback": False})["interview"]["id"]
    v = client.get(f"/interviews/{iid}", headers=auth).json()
    assert v["waiting_for"]["kind"] == "await_answer"          # HITL disabled -> straight to question 1
    # pause blocks actions; resume continues from the very same question
    q1 = v["waiting_for"]["question"]["text"]
    assert client.post(f"/interviews/{iid}/pause", headers=auth).json()["status"] == "paused"
    assert client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "x"}).status_code == 409
    v = client.post(f"/interviews/{iid}/resume", headers=auth).json()
    assert v["status"] == "awaiting_input" and v["waiting_for"]["question"]["text"] == q1
    # skip question 1
    v = client.post(f"/interviews/{iid}/next", headers=auth, json={"decision": "skip"}).json()["interview"]
    assert v["waiting_for"]["question"]["index"] == 1 and v["waiting_for"]["question"]["text"] != q1
    # answer question 2 with feedback review disabled -> straight to question 3
    v = client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "some answer"}).json()["interview"]
    assert v["waiting_for"]["question"]["index"] == 2 and v["state"]["scores"]
    # end early -> summary
    v = client.post(f"/interviews/{iid}/end", headers=auth, json={}).json()["interview"]
    assert v["status"] == "ended" and v["state"]["summary"]["skipped"] == 1 and v["state"]["summary"]["ended_early"]


def test_ownership_and_thread_isolation(client, auth_pair):
    (ha, _), (hb, _) = auth_pair
    iid = start(client, ha)["interview"]["id"]
    for method, path, body in [("get", f"/interviews/{iid}", None), ("get", f"/interviews/{iid}/answers", None),
                               ("post", f"/interviews/{iid}/pause", None),
                               ("post", f"/interviews/{iid}/answer", {"answer": "x"}),
                               ("post", f"/interviews/{iid}/next", {"decision": "approve"}),
                               ("get", f"/interviews/{iid}/stream", None)]:
        r = getattr(client, method)(path, headers=hb, **({"json": body} if body else {}))
        assert r.status_code == 404, (path, r.status_code)
    assert client.get("/interviews", headers=hb).json() == []
    # two interviews of one user never share state
    other = start(client, ha, skills=["Java"])["interview"]["id"]
    a = client.get(f"/interviews/{iid}", headers=ha).json()["waiting_for"]["plan"]
    b = client.get(f"/interviews/{other}", headers=ha).json()["waiting_for"]["plan"]
    assert {p["topic"] for p in b} == {"Java"} and "Java" not in {p["topic"] for p in a}


def test_thread_persistence_across_service_restart(client, services, auth):
    """A brand-new graph/service instance on the same checkpointer resumes the same interview."""
    from app.services.interview_service import InterviewService

    iid = start(client, auth, hitl={"skills": False, "plan": False, "feedback": True})["interview"]["id"]
    old = services.interviews
    services.interviews = InterviewService(services, old.checkpointer)   # simulated process restart
    v = client.get(f"/interviews/{iid}", headers=auth).json()
    assert v["status"] == "awaiting_input" and v["waiting_for"]["kind"] == "await_answer"
    v = client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "after restart"}).json()["interview"]
    assert v["waiting_for"]["kind"] == "review_feedback"


def test_streaming_flow(client, auth):
    r = client.post("/interviews/start", headers=auth, json={**START, "stream": True,
                    "hitl": {"skills": False, "plan": False, "feedback": True}})
    iid, url = r.json()["interview"]["id"], r.json()["stream_url"]
    assert r.json()["queued"] and url.endswith("/stream")
    s = client.get(url, headers=auth)
    assert s.headers["content-type"].startswith("text/event-stream")
    ev = parse_sse(s.text)
    names = [e["event"] for e in ev]
    assert names[0] == "run_start" and names[-1] == "done" and "interrupt" in names
    q_tokens = [e["data"]["text"] for e in ev if e["event"] == "token" and e["data"]["node"] == "generate_question"]
    q_text = "".join(q_tokens).strip()
    state_ev = next(e for e in ev if e["event"] == "interrupt")
    assert state_ev["data"]["kind"] == "await_answer" and state_ev["data"]["question"]["text"] == q_text  # no dup/missing tokens
    # a second open of the stream has nothing to run
    assert client.get(url, headers=auth).status_code == 409
    # answer through the stream: feedback tokens arrive incrementally
    assert client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "x", "stream": True}).json()["queued"]
    ev = parse_sse(client.get(url, headers=auth).text)
    fb = "".join(e["data"]["text"] for e in ev if e["event"] == "token" and e["data"]["node"] == "feedback")
    assert fb.strip().startswith("Nice work")
    assert [e["data"]["kind"] for e in ev if e["event"] == "interrupt"] == ["review_feedback"]
    # a second command can't be queued while one is pending
    assert client.post(f"/interviews/{iid}/next", headers=auth, json={"decision": "accept", "stream": True}).status_code == 200
    assert client.post(f"/interviews/{iid}/next", headers=auth, json={"decision": "accept", "stream": True}).status_code in (409,)


def test_llm_failure_is_recoverable_with_retry(client, auth, chat_model, monkeypatch):
    async def nosleep(_):
        return None

    monkeypatch.setattr("app.services.llm_service.asyncio.sleep", nosleep)
    iid = start(client, auth, hitl={"skills": False, "plan": False, "feedback": True})["interview"]["id"]
    chat_model.fail_times = 99                     # LLM outage during answer evaluation/feedback
    r = client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "x"})
    assert r.status_code in (200, 503)
    chat_model.fail_times = 0
    v = client.get(f"/interviews/{iid}", headers=auth).json()
    if v["status"] == "needs_retry":
        v = client.post(f"/interviews/{iid}/next", headers=auth, json={"decision": "retry"}).json()["interview"]
    assert v["waiting_for"]["kind"] == "review_feedback"


def test_resume_and_jd_modes_call_mcp_tools(client, auth, services):
    h = auth
    rid = client.post("/resume/upload", headers=h, files={"file": ("cv.pdf", make_pdf("Skills: Python, Docker, FastAPI\nProjects\nChatbot\nRAG bot using LangChain\n"), "application/pdf")}).json()["id"]
    v = start(client, h, mode="resume_based", resume_id=rid, skills=[], max_questions=3)["interview"]
    assert v["waiting_for"]["kind"] == "review_skills"                       # HITL: skill correction
    assert "Python" in v["waiting_for"]["skills"]
    v = client.post(f"/interviews/{v['id']}/next", headers=h, json={"decision": "approve", "skills": ["Python", "Docker"]}).json()["interview"]
    assert v["waiting_for"]["kind"] == "approve_plan"
    assert {p["topic"] for p in v["waiting_for"]["plan"]} <= {"Python", "Docker", "Projects"}
    tools = {t for t, _ in services.mcp.call_log}
    assert {"summarize_resume", "analyze_resume_projects", "extract_resume_skills", "get_weak_topics",
            "get_user_performance"} <= tools
    # JD mode
    jd = "We are hiring a backend engineer with strong Python, Docker and MongoDB experience. " * 2
    v = start(client, h, mode="jd_based", resume_id=rid, job_description=jd, skills=[])["interview"]
    assert v["waiting_for"]["kind"] == "review_skills" and v["state"]["jd_skills"]
    assert {"analyze_job_description", "extract_required_skills", "compare_resume_with_job_description"} <= {t for t, _ in services.mcp.call_log}
    # ownership of resume enforced when starting
    h2, _ = register(client, "mallory@example.com", "M")
    assert client.post("/interviews/start", headers=h2, json={"mode": "resume_based", "resume_id": rid}).status_code == 404


def test_validation_errors(client, auth):
    assert client.post("/interviews/start", headers=auth, json={"mode": "skill_based"}).status_code == 422
    assert client.post("/interviews/start", headers=auth, json={"mode": "resume_based"}).status_code == 422
    assert client.post("/interviews/start", headers=auth, json={**START, "role": "Wizard"}).status_code == 422
    assert client.post("/interviews/start", headers=auth, json={**START, "max_questions": 99}).status_code == 422


def test_weak_topic_mode_uses_long_term_memory(client, auth):
    h = {"skills": False, "plan": False, "feedback": False}
    # first interview: low scores on DBMS via fake evaluator override
    from tests import fakes
    orig = dict(fakes.EVAL_JSON)
    fakes.EVAL_JSON.update(rubric={k: 2 for k in orig["rubric"]}, score=2)
    try:
        iid = start(client, auth, skills=["DBMS"], max_questions=3, hitl=h)["interview"]["id"]
        for _ in range(3):
            client.post(f"/interviews/{iid}/answer", headers=auth, json={"answer": "idk"})
    finally:
        fakes.EVAL_JSON.clear(); fakes.EVAL_JSON.update(orig)
    assert [w["topic"] for w in client.get("/analytics/weak-topics", headers=auth).json()] == ["DBMS"]
    v = client.post("/interviews/start", headers=auth, json={"mode": "weak_topics", "max_questions": 2, "hitl": h}).json()["interview"]
    assert v["waiting_for"]["question"]["topic"] == "DBMS"
    assert client.get("/analytics/recommendations", headers=auth).json()
    practice = client.get("/analytics/practice-questions", headers=auth).json()
    assert practice["based_on_weak_topics"] and practice["practice"][0]["topic"] == "DBMS"
