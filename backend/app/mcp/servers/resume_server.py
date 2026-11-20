from __future__ import annotations

from typing import Any

from fastmcp import FastMCP

from app.mcp import logic
from app.mcp.servers._common import ResumeId, UserId, guard

mcp = FastMCP("resume-analysis", instructions="Analyse a user's stored resume: skills, projects, summary.")


@mcp.tool(description="Return the list of technical skills extracted from the user's resume.")
async def extract_resume_skills(user_id: UserId, resume_id: ResumeId) -> dict[str, Any]:
    return await guard(logic.extract_resume_skills(user_id, resume_id))


@mcp.tool(description="Analyse resume projects: technologies used and topics worth probing in an interview.")
async def analyze_resume_projects(user_id: UserId, resume_id: ResumeId) -> dict[str, Any]:
    return await guard(logic.analyze_resume_projects(user_id, resume_id))


@mcp.tool(description="Summarise the resume: profile summary, experience and education.")
async def summarize_resume(user_id: UserId, resume_id: ResumeId) -> dict[str, Any]:
    return await guard(logic.summarize_resume(user_id, resume_id))
