"""Application exceptions with safe, user-facing messages."""
from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "internal_error"

    def __init__(self, message: str = "Internal error"):
        super().__init__(message)
        self.message = message


class BadRequestError(AppError):
    status_code, code = 400, "bad_request"


class UnauthorizedError(AppError):
    status_code, code = 401, "unauthorized"


class ForbiddenError(AppError):
    status_code, code = 403, "forbidden"


class NotFoundError(AppError):
    status_code, code = 404, "not_found"


class ConflictError(AppError):
    status_code, code = 409, "conflict"


class ValidationFailed(AppError):
    status_code, code = 422, "validation_error"


class LLMError(AppError):
    status_code, code = 502, "llm_error"


class LLMConfigError(LLMError):
    code = "llm_config_error"


class LLMUnavailableError(LLMError):
    status_code, code = 503, "llm_unavailable"


class LLMRateLimitError(LLMError):
    status_code, code = 429, "llm_rate_limited"


class LLMOutputError(LLMError):
    code = "llm_bad_output"


class MCPToolError(AppError):
    status_code, code = 502, "mcp_tool_error"


def public_message(exc: BaseException) -> str:
    if isinstance(exc, AppError):
        return exc.message
    return "An internal error occurred. Please try again."
