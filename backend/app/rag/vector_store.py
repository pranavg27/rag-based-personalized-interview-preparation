"""Vector store abstraction: ChromaDB (default) or FAISS. All searches are filter-scoped."""
from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import Settings

log = logging.getLogger(__name__)


@dataclass
class ScoredChunk:
    text: str
    metadata: dict[str, Any]
    score: float  # cosine similarity in [-1, 1]; higher is better
    rerank_score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def _clean_meta(meta: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in meta.items() if isinstance(v, (str, int, float, bool))}


class BaseVectorStore(ABC):
    @abstractmethod
    def add_chunks(self, chunks: list[Document]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int, filters: dict[str, Any]) -> list[ScoredChunk]: ...

    @abstractmethod
    def delete_document(self, user_id: str, document_id: str) -> None: ...

    @abstractmethod
    def count(self, user_id: str | None = None) -> int: ...


class ChromaVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings, embeddings: Embeddings):
        import chromadb

        self.embeddings = embeddings
        if settings.chroma_mode == "http":
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        else:
            settings.chroma_path.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(settings.chroma_path))
        self.collection = client.get_or_create_collection(
            settings.chroma_collection, metadata={"hnsw:space": "cosine"}
        )

    @staticmethod
    def _where(filters: dict[str, Any]) -> dict[str, Any] | None:
        items = [{k: {"$eq": v}} for k, v in filters.items() if v is not None]
        if not items:
            return None
        return items[0] if len(items) == 1 else {"$and": items}

    def add_chunks(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        texts = [c.page_content for c in chunks]
        self.collection.upsert(
            ids=[c.metadata["chunk_id"] for c in chunks],
            documents=texts,
            metadatas=[_clean_meta(c.metadata) for c in chunks],
            embeddings=self.embeddings.embed_documents(texts),
        )

    def search(self, query: str, k: int, filters: dict[str, Any]) -> list[ScoredChunk]:
        total = self.collection.count()
        if total == 0:
            return []
        res = self.collection.query(
            query_embeddings=[self.embeddings.embed_query(query)],
            n_results=min(k, total),
            where=self._where(filters),
            include=["documents", "metadatas", "distances"],
        )
        docs, metas, dists = res["documents"][0], res["metadatas"][0], res["distances"][0]
        return [ScoredChunk(d, m or {}, 1.0 - float(dist)) for d, m, dist in zip(docs, metas, dists)]

    def delete_document(self, user_id: str, document_id: str) -> None:
        self.collection.delete(where=self._where({"user_id": user_id, "document_id": document_id}))

    def count(self, user_id: str | None = None) -> int:
        if user_id is None:
            return self.collection.count()
        return len(self.collection.get(where={"user_id": user_id}, include=[])["ids"])


class FaissVectorStore(BaseVectorStore):
    def __init__(self, settings: Settings, embeddings: Embeddings):
        self.embeddings = embeddings
        self.path = settings.faiss_path
        self._lock = threading.Lock()
        self._store = None
        if (self.path / "index.faiss").exists():
            from langchain_community.vectorstores import FAISS

            self._store = FAISS.load_local(
                str(self.path), embeddings, allow_dangerous_deserialization=True
            )

    def _save(self) -> None:
        self.path.mkdir(parents=True, exist_ok=True)
        self._store.save_local(str(self.path))

    def add_chunks(self, chunks: list[Document]) -> None:
        if not chunks:
            return
        from langchain_community.vectorstores import FAISS

        chunks = [Document(page_content=c.page_content, metadata=_clean_meta(c.metadata)) for c in chunks]
        ids = [c.metadata["chunk_id"] for c in chunks]
        with self._lock:
            if self._store is None:
                self._store = FAISS.from_documents(chunks, self.embeddings, ids=ids)
            else:
                self._store.add_documents(chunks, ids=ids)
            self._save()

    def search(self, query: str, k: int, filters: dict[str, Any]) -> list[ScoredChunk]:
        if self._store is None:
            return []
        flt = {k_: v for k_, v in filters.items() if v is not None}
        hits = self._store.similarity_search_with_score(
            query, k=k, filter=flt or None, fetch_k=max(k * 20, 100)
        )
        # FAISS returns squared-L2 on unit vectors: cos = 1 - d/2
        return [ScoredChunk(d.page_content, d.metadata, 1.0 - float(dist) / 2.0) for d, dist in hits]

    def delete_document(self, user_id: str, document_id: str) -> None:
        if self._store is None:
            return
        with self._lock:
            ids = [
                i for i, d in self._store.docstore._dict.items()
                if d.metadata.get("user_id") == user_id and d.metadata.get("document_id") == document_id
            ]
            if ids:
                self._store.delete(ids)
                self._save()

    def count(self, user_id: str | None = None) -> int:
        if self._store is None:
            return 0
        docs = self._store.docstore._dict.values()
        return sum(1 for d in docs if user_id is None or d.metadata.get("user_id") == user_id)


def get_vector_store(settings: Settings, embeddings: Embeddings) -> BaseVectorStore:
    if settings.vector_store.lower() == "faiss":
        return FaissVectorStore(settings, embeddings)
    return ChromaVectorStore(settings, embeddings)
