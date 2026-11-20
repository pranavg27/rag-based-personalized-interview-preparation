from __future__ import annotations

from typing import Annotated, Any

from fastmcp import FastMCP
from pydantic import Field

from app.mcp import logic
from app.mcp.servers._common import JDText, ResumeId, UserId, guard

mcp = FastMCP("job-description", instructions="Analyse job descriptions and compare them with a resume.")


@mcp.tool(description="Analyse a job description: role, seniority, required/preferred skills, responsibilities.")
async def analyze_job_description(user_id: UserId, job_description: JDText,
                                  role: Annotated[str | None, Field(max_length=60)] = None) -> dict[str, Any]:
    return await guard(logic.analyze_job_description(user_id, job_description, role))


@mcp.tool(description="Extract the list of required skills from a job description (deterministic).")
async def extract_required_skills(job_description: JDText) -> dict[str, Any]:
    return await guard(logic.extract_required_skills(job_description))


@mcp.tool(description="Compare the user's resume skills with a job description: matched, missing, match score.")
async def compare_resume_with_job_description(user_id: UserId, resume_id: ResumeId,
                                              job_description: JDText) -> dict[str, Any]:
    return await guard(logic.compare_resume_with_job_description(user_id, resume_id, job_description))
