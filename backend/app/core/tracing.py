"""LangSmith wiring. The app runs identically when tracing is disabled."""
from __future__ import annotations

import logging
import os
from typing import Any

from app.config import Settings

log = logging.getLogger(__name__)


def configure_tracing(settings: Settings) -> bool:
    enabled = bool(settings.langsmith_tracing and settings.langsmith_api_key)
    if enabled:
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
        log.info("LangSmith tracing enabled (project=%s)", settings.langsmith_project)
    else:
        os.environ["LANGSMITH_TRACING"] = "false"
        if settings.langsmith_tracing:
            log.warning("LANGSMITH_TRACING=true but LANGSMITH_API_KEY is empty; tracing disabled.")
    return enabled


def run_config(name: str, user_id: str, interview_id: str, **extra: Any) -> dict[str, Any]:
    """Metadata attached to every graph run so traces can be filtered per user/interview."""
    return {
        "run_name": name,
        "tags": ["interview-prep", name],
        "metadata": {"user_id": user_id, "interview_id": interview_id, **extra},
    }
