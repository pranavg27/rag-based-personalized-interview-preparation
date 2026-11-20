from tests.conftest import register
from tests.fakes import make_pdf

RESUME = "Jane Doe\nSkills: Python, FastAPI, MongoDB, Docker\nProjects\nChatbot\nBuilt a RAG chatbot with LangChain\n"


def test_auth_flow(client):
    assert client.get("/health").json()["mongodb"] == "ok"
    assert client.get("/auth/me").status_code == 401
    assert client.post("/auth/register", json={"email": "bad", "full_name": "x", "password": "Passw0rd1"}).status_code == 422
    assert client.post("/auth/register", json={"email": "a@b.co", "full_name": "x", "password": "short"}).status_code == 422
    h, _ = register(client, "z@z.com")
    assert client.get("/auth/me", headers=h).json()["email"] == "z@z.com"
    assert client.post("/auth/register", json={"email": "z@z.com", "full_name": "x", "password": "Passw0rd1"}).status_code == 409
    assert client.post("/auth/login", json={"email": "z@z.com", "password": "nope12345"}).status_code == 401
    assert client.get("/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401


def test_resume_upload_and_security(client, auth):
    r = client.post("/resume/upload", headers=auth, files={"file": ("cv.pdf", make_pdf(RESUME), "application/pdf")},
                    data={"target_role": "Backend Developer"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert "Python" in body["profile"]["skills"] and body["extraction"] == "llm"
    assert client.get("/resume", headers=auth).json()[0]["id"] == body["id"]
    # validation
    assert client.post("/resume/upload", headers=auth, files={"file": ("cv.txt", b"hello", "text/plain")}).status_code == 422
    assert client.post("/resume/upload", headers=auth, files={"file": ("cv.pdf", b"not a pdf at all", "application/pdf")}).status_code == 422
    assert client.post("/resume/upload", headers=auth, files={"file": ("cv.pdf", make_pdf(RESUME), "application/pdf")},
                       data={"target_role": "Astronaut"}).status_code == 422
    assert client.delete(f"/resume/{body['id']}", headers=auth).status_code == 204
    assert client.get("/resume", headers=auth).json() == []


def test_resume_isolation(client, auth_pair):
    (ha, _), (hb, _) = auth_pair
    rid = client.post("/resume/upload", headers=ha, files={"file": ("cv.pdf", make_pdf(RESUME), "application/pdf")}).json()["id"]
    assert client.get("/resume", headers=hb).json() == []
    assert client.delete(f"/resume/{rid}", headers=hb).status_code == 404


def test_upload_size_limit(client, auth, services):
    services.settings.max_upload_mb = 0  # any non-empty file is too large
    r = client.post("/documents/upload", headers=auth, files={"file": ("a.txt", b"x" * 100, "text/plain")})
    assert r.status_code == 400


def test_documents_lifecycle_and_chat(client, auth_pair):
    (ha, _), (hb, _) = auth_pair
    text = ("Normalization reduces redundancy in relational databases and avoids update anomalies. " * 10).encode()
    r = client.post("/documents/upload", headers=ha, files={"file": ("dbms.txt", text, "text/plain")})
    assert r.status_code == 201 and r.json()["skill_category"] == "DBMS" and r.json()["chunk_count"] >= 1
    doc_id = r.json()["id"]
    ans = client.post("/chat/ask", headers=ha, json={"question": "What is normalization?"}).json()
    assert ans["grounded"] and ans["sources"][0]["document_name"] == "dbms.txt"
    other = client.post("/chat/ask", headers=hb, json={"question": "What is normalization?"}).json()
    assert not other["grounded"] and other["sources"] == []
    s = client.post("/chat/stream", headers=ha, json={"question": "What is normalization?"})
    assert s.headers["content-type"].startswith("text/event-stream")
    assert "event: sources" in s.text and "event: token" in s.text and s.text.rstrip().endswith('data: {"type": "done"}')
    assert client.delete(f"/documents/{doc_id}", headers=hb).status_code == 404
    assert client.delete(f"/documents/{doc_id}", headers=ha).status_code == 204
    assert not client.post("/chat/ask", headers=ha, json={"question": "What is normalization?"}).json()["grounded"]


def test_job_description_upload(client, auth):
    r = client.post("/documents/job-description", headers=auth,
                    json={"title": "Backend role", "text": "We need Python, Docker and MongoDB experience. " * 3})
    assert r.status_code == 201 and r.json()["doc_type"] == "job_description"
