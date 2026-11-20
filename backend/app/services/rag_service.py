"""Grounded answer generation over the user's own documents."""
from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from app.config import Settings
from app.rag.retriever import Retriever
from app.services import prompts
from app.services.llm_service import LLMService


class RAGService:
    def __init__(self, llm: LLMService, retriever: Retriever, settings: Settings):
        self.llm, self.retriever, self.s = llm, retriever, settings

    async def answer(self, user_id: str, question: str, skill_category: str | None = None) -> dict[str, Any]:
        res = await self.retriever.retrieve(user_id, question, "technical", skill_category)
        if not res.found and not self.s.rag_allow_general_fallback:
            return {"answer": "I could not find relevant information in your uploaded materials.",
                    "sources": [], "grounded": False}
        text = await self.llm.ainvoke_text(prompts.rag_messages(question, res.context, res.found))
        return {"answer": text, "sources": res.sources, "grounded": res.found}

    async def answer_stream(self, user_id: str, question: str,
                            skill_category: str | None = None) -> AsyncIterator[dict[str, Any]]:
        yield {"type": "node_start", "node": "chat", "message": "Searching your documents…"}
        res = await self.retriever.retrieve(user_id, question, "technical", skill_category)
        yield {"type": "sources", "sources": res.sources, "grounded": res.found}
        if not res.found and not self.s.rag_allow_general_fallback:
            yield {"type": "token", "node": "chat", "text": "I could not find relevant information in your uploaded materials."}
            return
        async for piece in self.llm.astream_text(prompts.rag_messages(question, res.context, res.found)):
            yield {"type": "token", "node": "chat", "text": piece}
