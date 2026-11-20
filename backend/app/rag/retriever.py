"""Strategy-based retrieval with metadata filtering, score threshold, optional rerank and citations."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from langsmith import traceable

from app.config import Settings
from app.rag.reranker import Reranker
from app.rag.vector_store import BaseVectorStore, ScoredChunk

MAX_CONTEXT_CHARS = 6000


@dataclass
class RetrievalResult:
    chunks: list[ScoredChunk] = field(default_factory=list)
    context: str = ""
    sources: list[dict[str, Any]] = field(default_factory=list)

    @property
    def found(self) -> bool:
        return bool(self.chunks)

    def as_dict(self) -> dict[str, Any]:
        return {"context": self.context, "sources": self.sources, "found": self.found}


@dataclass(frozen=True)
class Strategy:
    doc_types: tuple[str, ...] | None  # None = any
    top_n: int | None = None           # None = settings.rerank_top_n
    threshold_delta: float = 0.0
    k_mult: float = 1.0


STRATEGIES: dict[str, Strategy] = {
    "technical": Strategy(("study",)),
    "resume": Strategy(("study",), threshold_delta=-0.05),
    "jd": Strategy(("job_description", "study")),
    "evaluation": Strategy(("study",), top_n=3, threshold_delta=-0.05, k_mult=1.5),
}


class Retriever:
    def __init__(self, store: BaseVectorStore, reranker: Reranker, settings: Settings):
        self.store, self.reranker, self.settings = store, reranker, settings

    @staticmethod
    def build_query(strategy: str, topic: str = "", question: str = "", answer: str = "",
                    skills: list[str] | None = None) -> str:
        if strategy == "evaluation":
            return f"{topic} {question} {answer[:400]}".strip()
        if strategy == "resume":
            return f"{topic} {' '.join((skills or [])[:6])} concepts interview".strip()
        if strategy == "jd":
            return f"{topic} {' '.join((skills or [])[:6])} requirements".strip()
        return f"{topic} {question}".strip()

    async def retrieve(
        self, user_id: str, query: str, strategy: str = "technical",
        skill_category: str | None = None, top_k: int | None = None,
    ) -> RetrievalResult:
        if not user_id:
            raise ValueError("user_id is required for retrieval")
        if not query.strip():
            return RetrievalResult()
        return await self._retrieve(user_id, query, strategy, skill_category, top_k)

    @traceable(name="rag_retrieve", run_type="retriever")
    async def _retrieve(
        self, user_id: str, query: str, strategy: str, skill_category: str | None, top_k: int | None
    ) -> RetrievalResult:
        cfg = STRATEGIES.get(strategy, STRATEGIES["technical"])
        s = self.settings
        k = int((top_k or s.retrieval_top_k) * cfg.k_mult)
        # user_id is ALWAYS part of the filter -> cross-user retrieval is impossible.
        filters: dict[str, Any] = {"user_id": user_id}
        if skill_category:
            filters["skill_category"] = skill_category
        raw = await asyncio.to_thread(self.store.search, query, k * 2, filters)
        threshold = s.similarity_threshold + cfg.threshold_delta
        chunks = [
            c for c in raw
            if c.score >= threshold and c.metadata.get("user_id") == user_id
            and (cfg.doc_types is None or c.metadata.get("doc_type", "study") in cfg.doc_types)
        ][:k]
        top_n = cfg.top_n or s.rerank_top_n
        if chunks:
            chunks = await asyncio.to_thread(self.reranker.rerank, query, chunks, top_n)
        return self._assemble(chunks)

    @staticmethod
    def _assemble(chunks: list[ScoredChunk]) -> RetrievalResult:
        parts, sources, used = [], [], 0
        for i, c in enumerate(chunks, 1):
            m = c.metadata
            block = f"[{i}] (source: {m.get('document_name', 'unknown')}, p.{m.get('page', '?')})\n{c.text}"
            if used + len(block) > MAX_CONTEXT_CHARS and parts:
                break
            parts.append(block)
            used += len(block)
            sources.append({
                "index": i, "document_id": m.get("document_id"), "document_name": m.get("document_name"),
                "page": m.get("page"), "chunk_id": m.get("chunk_id"), "score": round(c.score, 3),
                "snippet": c.text[:200],
            })
        return RetrievalResult(chunks=chunks[: len(parts)], context="\n\n".join(parts), sources=sources)
