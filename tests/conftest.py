from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_TMP = tempfile.mkdtemp(prefix="ip_tests_")
os.environ.update({
    "APP_ENV": "test", "RATE_LIMIT_ENABLED": "false", "CHECKPOINTER": "memory", "MCP_MODE": "disabled",
    "RERANKER_ENABLED": "false", "UPLOAD_DIR": f"{_TMP}/uploads", "CHROMA_PERSIST_DIR": f"{_TMP}/chroma",
    "FAISS_INDEX_DIR": f"{_TMP}/faiss", "LLM_MAX_RETRIES": "1", "HF_TOKEN": "", "LANGSMITH_TRACING": "false",
    "SIMILARITY_THRESHOLD": "0.1", "JWT_SECRET_KEY": "test-secret-key-for-unit-tests-only-1234567890",
})
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.database import mongo  # noqa: E402
from app.services.container import Services, set_services  # noqa: E402
from app.services.llm_service import LLMService  # noqa: E402
from tests.fakes import FakeChatModel, FakeDB, HashEmbeddings  # noqa: E402


@pytest.fixture
def db():
    d = FakeDB()
    mongo.set_db(d)
    yield d
    mongo.set_db(None)


@pytest.fixture
def chat_model():
    return FakeChatModel(calls=[])


@pytest.fixture
def services(db, chat_model, tmp_path):
    s = get_settings().model_copy(update={
        "chroma_persist_dir": str(tmp_path / "chroma"), "faiss_index_dir": str(tmp_path / "faiss"),
        "upload_dir": str(tmp_path / "uploads")})
    svc = Services(s)
    svc.llm = LLMService(s, model=chat_model)
    svc.embeddings = HashEmbeddings()
    set_services(svc)
    yield svc
    set_services(None)


@pytest.fixture
def client(services):
    from starlette.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def register(client, email="a@example.com", name="Alice"):
    r = client.post("/auth/register", json={"email": email, "full_name": name, "password": "Passw0rdX"})
    assert r.status_code == 201, r.text
    t = client.post("/auth/login", json={"email": email, "password": "Passw0rdX"}).json()
    return {"Authorization": f"Bearer {t['access_token']}"}, t["user"]["id"]


@pytest.fixture
def auth(client):
    return register(client)[0]


@pytest.fixture
def auth_pair(client):
    return register(client, "a@example.com", "Alice"), register(client, "b@example.com", "Bob")
