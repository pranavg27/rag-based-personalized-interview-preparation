from __future__ import annotations

import hmac
import logging
from typing import Annotated, Any

from fastmcp.exceptions import ToolError
from pydantic import Field
from starlette.middleware import Middleware

from app.errors import AppError

log = logging.getLogger(__name__)

UserId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$", description="Authenticated user id (32 hex chars)")]
ResumeId = Annotated[str, Field(pattern=r"^[0-9a-f]{32}$", description="Resume id (32 hex chars)")]
Topic = Annotated[str, Field(min_length=1, max_length=80, description="Technical topic, e.g. 'DBMS'")]
JDText = Annotated[str, Field(min_length=30, max_length=30000, description="Job description text")]


async def guard(coro) -> dict[str, Any]:
    """Turn application errors into clean MCP tool errors (no stack traces leak)."""
    try:
        return await coro
    except AppError as exc:
        raise ToolError(exc.message) from exc
    except ToolError:
        raise
    except Exception as exc:
        log.exception("MCP tool crashed")
        raise ToolError("Tool failed unexpectedly.") from exc


class BearerAuthMiddleware:
    """Pure-ASGI shared-secret check for MCP HTTP endpoints."""

    def __init__(self, app, token: str):
        self.app, self.token = app, token

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            supplied = headers.get(b"authorization", b"").decode()
            expected = f"Bearer {self.token}"
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                body = b'{"error":"unauthorized"}'
                await send({"type": "http.response.start", "status": 401,
                            "headers": [(b"content-type", b"application/json"),
                                        (b"content-length", str(len(body)).encode())]})
                await send({"type": "http.response.body", "body": body})
                return
        await self.app(scope, receive, send)


def http_app(server, token: str):
    return server.http_app(path="/mcp", middleware=[Middleware(BearerAuthMiddleware, token=token)])
