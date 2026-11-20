import docx
import pytest

from app.errors import ValidationFailed
from app.rag.ingestion import build_chunks, detect_category, ingest_file
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore, FaissVectorStore
from app.rag.reranker import Reranker
from tests.fakes import HashEmbeddings, make_pdf

DBMS = "Normalization reduces redundancy in a relational database. First normal form requires atomic values. " * 8
OS = "A deadlock occurs when processes wait for each other. Semaphores and mutexes coordinate threads. " * 8


def _ingest(settings, store, tmp_path, user, text, doc_id, name="notes.txt", category=None):
    p = tmp_path / name
    p.write_text(text)
    res = ingest_file(p, "txt", settings=settings, user_id=user, document_id=doc_id, document_name=name,
                      skill_category=category)
    store.add_chunks(res.chunks)
    return res


@pytest.fixture(params=["chroma", "faiss"])
def store(request, services):
    cls = ChromaVectorStore if request.param == "chroma" else FaissVectorStore
    return cls(services.settings, HashEmbeddings())


def test_chunk_metadata_and_category(services, tmp_path):
    p = tmp_path / "dbms.txt"
    p.write_text(DBMS)
    res = ingest_file(p, "txt", settings=services.settings, user_id="u1", document_id="d1",
                      document_name="dbms.txt", skill_category=None)
    assert res.skill_category == "DBMS" and detect_category(OS) == "Operating Systems"
    m = res.chunks[0].metadata
    assert {"document_id", "user_id", "document_name", "skill_category", "page", "chunk_id", "uploaded_at"} <= set(m)
    assert m["chunk_id"] == "d1_0" and m["page"] == 1


def test_pdf_and_docx_loading(services, tmp_path):
    pdf = tmp_path / "a.pdf"
    pdf.write_bytes(make_pdf("Transformers use attention mechanisms for sequence modelling in deep learning."))
    r = ingest_file(pdf, "pdf", settings=services.settings, user_id="u", document_id="d", document_name="a.pdf",
                    skill_category=None)
    assert "attention" in r.chunks[0].page_content
    d = docx.Document()
    d.add_paragraph("Docker containers package applications with their dependencies for deployment.")
    path = tmp_path / "a.docx"
    d.save(path)
    r2 = ingest_file(path, "docx", settings=services.settings, user_id="u", document_id="d2",
                     document_name="a.docx", skill_category=None)
    assert "Docker" in r2.chunks[0].page_content


def test_empty_document_rejected(services):
    from langchain_core.documents import Document

    with pytest.raises(ValidationFailed):
        build_chunks([Document(page_content="   ", metadata={})], settings=services.settings, user_id="u",
                     document_id="d", document_name="x", skill_category=None)


async def test_retrieval_isolation_and_citations(services, store, tmp_path):
    s = services.settings
    _ingest(s, store, tmp_path, "alice", DBMS, "da", "dbms.txt")
    _ingest(s, store, tmp_path, "bob", OS, "db", "os.txt")
    r = Retriever(store, Reranker(s), s)
    res = await r.retrieve("alice", "what is normalization in database", "technical")
    assert res.found and "[1]" in res.context and "dbms.txt" in res.context
    assert all(c.metadata["user_id"] == "alice" for c in res.chunks)
    # bob's content is invisible to alice, even on an exact-match query
    leak = await r.retrieve("alice", "deadlock semaphores mutexes threads", "technical")
    assert all(c.metadata["user_id"] == "alice" for c in leak.chunks)
    assert not any("deadlock" in c.text for c in leak.chunks)
    with pytest.raises(ValueError):
        await r.retrieve("", "x")


async def test_category_filter_threshold_and_delete(services, store, tmp_path):
    s = services.settings
    _ingest(s, store, tmp_path, "alice", DBMS, "da", "dbms.txt", category="DBMS")
    r = Retriever(store, Reranker(s), s)
    assert (await r.retrieve("alice", "normalization", "technical", skill_category="DBMS")).found
    assert not (await r.retrieve("alice", "normalization", "technical", skill_category="Java")).found
    strict = Retriever(store, Reranker(s), s.model_copy(update={"similarity_threshold": 0.99}))
    assert not (await strict.retrieve("alice", "completely unrelated zebra", "technical")).found
    store.delete_document("alice", "da")
    assert store.count("alice") == 0


def test_reranker_disabled_truncates(services):
    from app.rag.vector_store import ScoredChunk

    chunks = [ScoredChunk(f"t{i}", {}, 1 - i / 10) for i in range(5)]
    assert len(Reranker(services.settings).rerank("q", chunks, 2)) == 2
