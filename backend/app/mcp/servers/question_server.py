from __future__ import annotations

from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from pydantic import Field

from app.mcp import logic
from app.mcp.servers._common import Topic, UserId, guard

mcp = FastMCP("interview-questions", instructions="Generate and retrieve interview questions and study knowledge.")


@mcp.tool(description="Generate a batch of interview questions for a topic and difficulty using the LLM.")
async def generate_interview_questions(
    user_id: UserId, topic: Topic, difficulty: Literal["easy", "medium", "hard"] = "medium",
    count: Annotated[int, Field(ge=1, le=10)] = 3, role: Annotated[str | None, Field(max_length=60)] = None,
) -> dict[str, Any]:
    return await guard(logic.generate_interview_questions(user_id, topic, difficulty, count, role))


@mcp.tool(description="Retrieve questions the user has previously been asked on a topic (question bank / dedupe).")
async def retrieve_interview_questions(
    user_id: UserId, topic: Topic, limit: Annotated[int, Field(ge=1, le=50)] = 10
) -> dict[str, Any]:
    return await guard(logic.retrieve_interview_questions(user_id, topic, limit))


@mcp.tool(description="Semantic search over the user's own study materials (RAG). Returns context with citations.")
async def search_technical_knowledge(
    user_id: UserId, query: Annotated[str, Field(min_length=1, max_length=1000)],
    strategy: Literal["technical", "resume", "jd", "evaluation"] = "technical",
    skill_category: Annotated[str | None, Field(max_length=60)] = None,
    top_k: Annotated[int | None, Field(ge=1, le=30)] = None,
) -> dict[str, Any]:
    return await guard(logic.search_technical_knowledge(user_id, query, strategy, skill_category, top_k))
