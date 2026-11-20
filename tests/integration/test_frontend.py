"""Drives the real Streamlit pages (AppTest) against a live backend over HTTP + SSE."""
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import pytest
import uvicorn

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"


@pytest.fixture
def backend(services):
    from app.main import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    t.join(timeout=10)


@pytest.fixture
def frontend_env(backend, monkeypatch):
    monkeypatch.setenv("BACKEND_URL", backend)
    monkeypatch.syspath_prepend(str(FRONTEND))
    for m in [m for m in sys.modules if m == "services" or m.startswith(("services.", "components"))]:
        del sys.modules[m]
    r = httpx.post(f"{backend}/auth/register", json={"email": "f@e.com", "full_name": "Fran", "password": "Passw0rd1"})
    assert r.status_code == 201
    tok = httpx.post(f"{backend}/auth/login", json={"email": "f@e.com", "password": "Passw0rd1"}).json()
    return backend, tok


def test_live_interview_page_streams_and_walks_hitl(frontend_env):
    from streamlit.testing.v1 import AppTest

    backend, tok = frontend_env
    h = {"Authorization": f"Bearer {tok['access_token']}"}
    start = httpx.post(f"{backend}/interviews/start", headers=h, json={
        "mode": "skill_based", "skills": ["DBMS"], "role": "Backend Developer", "max_questions": 1,
        "hitl": {"skills": False, "plan": True, "feedback": True}, "stream": True}, timeout=30).json()
    iid = start["interview"]["id"]
    at = AppTest.from_file(str(FRONTEND / "pages" / "6_Live_Interview.py"), default_timeout=60)
    at.session_state["token"] = tok["access_token"]
    at.session_state["user"] = tok["user"]
    at.session_state["interview_id"] = iid
    at.session_state["run_stream"] = True          # the page consumes the SSE stream first
    at.run()
    assert not at.exception, at.exception
    assert any("Approve interview plan" in s.value for s in at.subheader)
    next(b for b in at.button if "Approve" in b.label).click()
    at.run()
    assert not at.exception, at.exception
    # approving queued a stream; the page streamed the question and now waits for an answer
    assert any("Your answer" in t.label for t in at.text_area)
    at.text_area[0].input("A normal form removes redundancy.").run()   # blur -> rerun enables Submit
    next(b for b in at.button if b.label == "Submit answer").click()
    at.run()
    assert not at.exception, at.exception
    assert any("Feedback" in s.value for s in at.subheader)
    note_ok = [b for b in at.button if "Accept" in b.label]
    assert note_ok
    note_ok[0].click()
    at.run()
    # AppTest cannot resolve st.page_link outside the multipage runner; that is the only tolerated exception.
    assert all("Could not find page" in e.message for e in at.exception), at.exception
    assert any("Interview finished" in s.value for s in at.success)
