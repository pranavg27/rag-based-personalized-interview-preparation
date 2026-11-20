import asyncio
import socket
import threading
import time

import pytest
import uvicorn
from fastmcp import Client
from fastmcp.exceptions import ToolError

from app.mcp.client import InMemoryMCPToolbox, MCPToolbox
from app.mcp.logic import TOOL_SERVER
from app.mcp.servers import SERVERS
from app.errors import MCPToolError
from tests.conftest import register
from tests.fakes import make_pdf


def test_every_tool_is_registered_with_schema_and_description():
    exposed = {}
    async def go():
        for name, srv in SERVERS.items():
            async with Client(srv) as c:
                for t in await c.list_tools():
                    exposed[t.name] = (name, t)
    asyncio.run(go())
    assert set(exposed) == set(TOOL_SERVER)
    for tool, (server, t) in exposed.items():
        assert TOOL_SERVER[tool] == server and t.description and t.inputSchema["properties"]


async def test_tools_execute_over_mcp_protocol(client, services, auth):
    uid = client.get("/auth/me", headers=auth).json()["id"]
    rid = client.post("/resume/upload", headers=auth, files={"file": ("cv.pdf", make_pdf("Skills: Python, Docker\nProjects\nBot\nBuilt a RAG bot with LangChain\n"), "application/pdf")}).json()["id"]
    box = InMemoryMCPToolbox(SERVERS)
    assert "Python" in (await box.call("extract_resume_skills", {"user_id": uid, "resume_id": rid}))["skills"]
    assert (await box.call("analyze_resume_projects", {"user_id": uid, "resume_id": rid}))["projects"]
    assert (await box.call("summarize_resume", {"user_id": uid, "resume_id": rid}))["skill_count"] >= 2
    jd = "Looking for an engineer with Python, Docker and Kubernetes experience for backend work."
    cmp_ = await box.call("compare_resume_with_job_description", {"user_id": uid, "resume_id": rid, "job_description": jd})
    assert "Kubernetes" in cmp_["missing"] and "Python" in cmp_["matched"]
    assert (await box.call("analyze_job_description", {"user_id": uid, "job_description": jd}))["required_skills"]
    assert "Docker" in (await box.call("extract_required_skills", {"job_description": jd}))["skills"]
    qs = await box.call("generate_interview_questions", {"user_id": uid, "topic": "DBMS", "count": 1})
    assert qs["questions"][0]["question"]
    assert (await box.call("get_weak_topics", {"user_id": uid}))["weak_topics"] == []
    assert "interviews" in await box.call("get_interview_history", {"user_id": uid})
    assert "answers_scored" in await box.call("get_user_performance", {"user_id": uid})
    assert (await box.call("generate_progress_report", {"user_id": uid}))["report"]
    assert (await box.call("search_technical_knowledge", {"user_id": uid, "query": "x"}))["found"] is False
    assert "questions" in await box.call("retrieve_interview_questions", {"user_id": uid, "topic": "DBMS"})


async def test_mcp_input_validation_and_authorization(client, services, auth_pair):
    (ha, ua), (hb, ub) = auth_pair
    rid = client.post("/resume/upload", headers=ha, files={"file": ("cv.pdf", make_pdf("Skills: Python and Docker for backend development work"), "application/pdf")}).json()["id"]
    box = InMemoryMCPToolbox(SERVERS)
    with pytest.raises(MCPToolError):   # bad id format rejected by the tool schema
        await box.call("extract_resume_skills", {"user_id": "nope", "resume_id": rid})
    with pytest.raises(MCPToolError, match="not found"):   # another user's resume
        await box.call("extract_resume_skills", {"user_id": ub, "resume_id": rid})
    with pytest.raises(MCPToolError):   # enum validation
        await box.call("generate_interview_questions", {"user_id": ua, "topic": "x", "difficulty": "impossible"})
    assert await box.call("extract_resume_skills", {"user_id": ub, "resume_id": rid}, default={"skills": []}) == {"skills": []}


async def test_toolbox_local_fallback_when_server_down(client, services, auth):
    uid = client.get("/auth/me", headers=auth).json()["id"]
    box = MCPToolbox({n: "http://127.0.0.1:9/mcp" for n in ("resume", "questions", "analytics", "jd")}, "t",
                     mode="external", fallback_local=True, timeout=3)
    out = await box.call("get_weak_topics", {"user_id": uid})
    assert out == {"weak_topics": []} and box.call_log[-1] == ("get_weak_topics", "local")
    strict = MCPToolbox(box.urls, "t", mode="external", fallback_local=False, timeout=3)
    with pytest.raises(MCPToolError):
        await strict.call("get_weak_topics", {"user_id": uid})


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live_server(services, monkeypatch):
    """Real uvicorn server with the embedded MCP servers mounted and bearer-protected."""
    from app.config import get_settings
    from app.main import create_app

    port = _free_port()
    s = get_settings()
    monkeypatch.setattr(s, "mcp_mode", "embedded")
    monkeypatch.setattr(s, "backend_url", f"http://127.0.0.1:{port}")
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.1)
    yield port, s
    server.should_exit = True
    t.join(timeout=10)


def test_embedded_mcp_over_http_with_auth(live_server, services, db):
    import httpx

    port, s = live_server
    base = f"http://127.0.0.1:{port}"
    assert httpx.post(f"{base}/mcp-servers/jd/mcp", json={}).status_code == 401          # no token
    assert httpx.post(f"{base}/mcp-servers/jd/mcp", json={}, headers={"Authorization": "Bearer wrong"}).status_code == 401
    box = MCPToolbox.from_settings(s)
    box.fallback_local = False
    jd = "Looking for an engineer with Python, Docker and Kubernetes experience for backend work."

    async def go():
        return await box.call("extract_required_skills", {"job_description": jd})

    out = asyncio.run(go())
    assert "Docker" in out["skills"] and box.call_log[-1] == ("extract_required_skills", "mcp")
