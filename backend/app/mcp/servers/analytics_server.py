from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from app.mcp import logic
from app.mcp.servers._common import UserId, guard

mcp = FastMCP("performance-analytics", instructions="Read-only analytics over a user's interview performance.")


@mcp.tool(description="Aggregate performance by topic and difficulty.")
async def get_user_performance(user_id: UserId) -> dict[str, Any]:
    return await guard(logic.get_user_performance(user_id))


@mcp.tool(description="Topics where the user's recent average score is below the weak-topic threshold.")
async def get_weak_topics(user_id: UserId, limit: Annotated[int, Field(ge=1, le=20)] = 5) -> dict[str, Any]:
    return await guard(logic.get_weak_topics(user_id, limit))


@mcp.tool(description="Most recent interviews with status and average score.")
async def get_interview_history(user_id: UserId, limit: Annotated[int, Field(ge=1, le=50)] = 10) -> dict[str, Any]:
    return await guard(logic.get_interview_history(user_id, limit))


@mcp.tool(description="Generate a narrative progress report with statistics and recommendations.")
async def generate_progress_report(user_id: UserId) -> dict[str, Any]:
    return await guard(logic.generate_progress_report(user_id))
