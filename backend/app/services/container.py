"""Service container: one place that wires every dependency (lazy; override-able in tests)."""
from __future__ import annotations

from functools import cached_property
from typing import Any

from app.config import Settings, get_settings
from app.rag.embeddings import get_embeddings
from app.rag.reranker import Reranker
from app.rag.retriever import Retriever
from app.rag.vector_store import get_vector_store
from app.services.analytics_service import AnalyticsService
from app.services.evaluation_service import EvaluationService
from app.services.llm_service import LLMService
from app.services.memory_service import MemoryService
from app.services.personalization_service import PersonalizationService
from app.services.rag_service import RAGService
from app.services.resume_service import ResumeService


class Services:
    interviews: Any = None  # InterviewService, attached during app startup

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    @cached_property
    def embeddings(self):
        return get_embeddings(self.settings)

    @cached_property
    def vector_store(self):
        return get_vector_store(self.settings, self.embeddings)

    @cached_property
    def reranker(self):
        return Reranker(self.settings)

    @cached_property
    def retriever(self):
        return Retriever(self.vector_store, self.reranker, self.settings)

    @cached_property
    def llm(self):
        return LLMService(self.settings)

    @cached_property
    def rag(self):
        return RAGService(self.llm, self.retriever, self.settings)

    @cached_property
    def memory(self):
        return MemoryService(self.settings)

    @cached_property
    def personalization(self):
        return PersonalizationService(self.settings)

    @cached_property
    def evaluation(self):
        return EvaluationService(self.llm, self.settings)

    @cached_property
    def resumes(self):
        return ResumeService(self.llm)

    @cached_property
    def analytics(self):
        return AnalyticsService(self.memory)

    @cached_property
    def mcp(self):
        from app.mcp.client import MCPToolbox

        return MCPToolbox.from_settings(self.settings)


_services: Services | None = None


def get_services() -> Services:
    global _services
    if _services is None:
        _services = Services()
    return _services


def set_services(s: Services | None) -> None:
    global _services
    _services = s
