"""Load -> clean -> split -> annotate. Returns LangChain Documents ready for the vector store."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import Settings
from app.errors import ValidationFailed
from app.utils.ids import utcnow

CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "DSA": ["algorithm", "data structure", "linked list", "binary tree", "graph", "dynamic programming", "sorting"],
    "DBMS": ["sql", "normalization", "transaction", "acid", "database", "index", "join"],
    "Operating Systems": ["process", "thread", "deadlock", "scheduling", "semaphore", "virtual memory", "paging"],
    "Computer Networks": ["tcp", "udp", "http", "dns", "osi", "routing", "ip address"],
    "Java": ["jvm", "java", "spring", "garbage collection", "servlet"],
    "Python": ["python", "pip", "decorator", "generator", "asyncio", "pandas"],
    "Machine Learning": ["regression", "classification", "overfitting", "gradient", "random forest", "feature"],
    "Deep Learning": ["neural network", "cnn", "backpropagation", "transformer", "lstm", "activation"],
    "NLP": ["tokenization", "embedding", "named entity", "language model", "bert"],
    "RAG": ["retrieval", "rag", "vector database", "chunking", "reranking"],
    "LangChain": ["langchain", "runnable", "chain", "prompt template"],
    "LangGraph": ["langgraph", "stategraph", "checkpointer", "interrupt"],
    "FastAPI": ["fastapi", "pydantic", "uvicorn", "dependency injection", "endpoint"],
    "System Design": ["scalability", "load balancer", "caching", "sharding", "microservice", "cap theorem"],
}


@dataclass
class IngestResult:
    chunks: list[Document]
    skill_category: str
    text_length: int


def detect_category(text: str, filename: str = "") -> str:
    hay = f"{filename} {text[:6000]}".lower()
    best, best_hits = "General", 0
    for cat, kws in CATEGORY_KEYWORDS.items():
        hits = sum(hay.count(k) for k in kws)
        if hits > best_hits:
            best, best_hits = cat, hits
    return best


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("\r\n", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # de-hyphenate line wraps
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(path: Path, ext: str) -> list[Document]:
    ext = ext.lower().lstrip(".")
    if ext == "pdf":
        return PyPDFLoader(str(path)).load()
    if ext == "docx":
        return Docx2txtLoader(str(path)).load()
    if ext == "txt":
        last: Exception | None = None
        for enc in ("utf-8", "latin-1"):
            try:
                return TextLoader(str(path), encoding=enc).load()
            except Exception as exc:  # decode problems surface as RuntimeError
                last = exc
        raise ValidationFailed(f"Could not read text file: {last}")
    raise ValidationFailed(f"Unsupported file type: {ext}")


def build_chunks(
    docs: list[Document], *, settings: Settings, user_id: str, document_id: str,
    document_name: str, skill_category: str | None, doc_type: str = "study",
) -> IngestResult:
    for d in docs:
        d.page_content = clean_text(d.page_content)
    docs = [d for d in docs if d.page_content]
    if not docs:
        raise ValidationFailed("No extractable text found (scanned or empty document?).")
    full_text = "\n".join(d.page_content for d in docs)
    category = skill_category or detect_category(full_text, document_name)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    uploaded_at = utcnow().isoformat()
    for i, c in enumerate(chunks):
        page = c.metadata.get("page")
        c.metadata = {
            "document_id": document_id, "user_id": user_id, "document_name": document_name,
            "skill_category": category, "doc_type": doc_type,
            "page": int(page) + 1 if isinstance(page, int) else 1,
            "chunk_id": f"{document_id}_{i}", "chunk_index": i, "uploaded_at": uploaded_at,
        }
    return IngestResult(chunks=chunks, skill_category=category, text_length=len(full_text))


def ingest_file(
    path: Path, ext: str, *, settings: Settings, user_id: str, document_id: str,
    document_name: str, skill_category: str | None, doc_type: str = "study",
) -> IngestResult:
    docs = load_documents(path, ext)
    return build_chunks(
        docs, settings=settings, user_id=user_id, document_id=document_id,
        document_name=document_name, skill_category=skill_category, doc_type=doc_type,
    )
