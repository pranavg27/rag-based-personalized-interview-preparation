from __future__ import annotations

import contextlib
import logging
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.api import analytics, auth, chat, documents, health, interviews, memory, resume
from app.config import get_settings
from app.core.logging import setup_logging
from app.core.rate_limit import limiter
from app.core.tracing import configure_tracing
from app.database import mongo
from app.database.indexes import ensure_indexes
from app.errors import AppError
from app.graph.checkpointer import create_checkpointer
from app.services.container import get_services
from app.services.interview_service import InterviewService

log = logging.getLogger("app")


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level)
    mcp_apps: dict = {}
    if settings.mcp_mode == "embedded":
        from app.mcp.servers import SERVERS
        from app.mcp.servers._common import http_app

        mcp_apps = {name: http_app(srv, settings.mcp_internal_token) for name, srv in SERVERS.items()}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        settings.validate_for_runtime()
        configure_tracing(settings)
        await ensure_indexes(mongo.get_db())
        services = get_services()
        services.interviews = InterviewService(services, create_checkpointer(settings))
        async with AsyncExitStack() as stack:
            for sub in mcp_apps.values():
                await stack.enter_async_context(sub.lifespan(sub))
            log.info("Interview Prep API ready (mcp_mode=%s, model=%s)", settings.mcp_mode, settings.llm_model_id)
            yield
        await mongo.close()

    app = FastAPI(title="RAG-Based Personalized Interview Preparation System", version="1.0.0", lifespan=lifespan)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origin_list, allow_credentials=True,
                       allow_methods=["*"], allow_headers=["*"])

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid, start = uuid.uuid4().hex[:8], time.perf_counter()
        response = await call_next(request)
        log.info("%s %s -> %s (%.0f ms) rid=%s", request.method, request.url.path, response.status_code,
                 (time.perf_counter() - start) * 1000, rid)
        response.headers["X-Request-ID"] = rid
        return response

    @app.exception_handler(AppError)
    async def app_error(_: Request, exc: AppError):
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.message, "code": exc.code})

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(status_code=500, content={"detail": "Internal server error.", "code": "internal_error"})

    for r in (health.router, auth.router, resume.router, documents.router, interviews.router, chat.router,
              memory.router, analytics.router):
        app.include_router(r)
    for name, sub in mcp_apps.items():
        app.mount(f"/mcp-servers/{name}", sub)
    return app


app = create_app()
