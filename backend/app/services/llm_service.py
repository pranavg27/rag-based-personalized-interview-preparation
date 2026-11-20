"""Modular LLM service around Meta Llama (Hugging Face Inference Providers).

* Model id / provider come from settings -> swap models by config only.
* Retries with backoff on 429/5xx/timeouts; maps auth/model errors to LLMConfigError.
* `ajson` gives schema-validated structured output with one repair retry.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import threading
from collections.abc import AsyncIterator
from typing import Any, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.config import Settings, get_settings
from app.errors import LLMConfigError, LLMError, LLMOutputError, LLMRateLimitError, LLMUnavailableError
from app.utils.jsonparse import extract_json

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}
_STATUS_RE = re.compile(r"\b([45]\d\d)\s+(?:Client|Server)\s+Error|status(?:\s+code)?[:=\s]+([45]\d\d)", re.I)


def msgs(system: str, user: str) -> list[BaseMessage]:
    return [SystemMessage(content=system), HumanMessage(content=user)]


def _status(exc: BaseException) -> int | None:
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 400 <= v < 600:
            return v
    resp = getattr(exc, "response", None)
    v = getattr(resp, "status_code", None)
    if isinstance(v, int):
        return v
    m = _STATUS_RE.search(str(exc)[:400])
    return int(m.group(1) or m.group(2)) if m else None


def _is_retryable(exc: BaseException, code: int | None) -> bool:
    if code is not None:
        return code in RETRYABLE
    name = type(exc).__name__.lower()
    return any(t in name for t in ("timeout", "connect", "readerror", "remoteprotocol")) or isinstance(
        exc, (asyncio.TimeoutError, ConnectionError)
    )


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _to_error(exc: BaseException, code: int | None) -> LLMError:
    if code in (401, 403):
        return LLMConfigError(
            "Hugging Face rejected the request (401/403). Check HF_TOKEN permissions "
            "('Make calls to Inference Providers') and that you accepted the Llama license."
        )
    if code == 404 or (code == 400 and "not supported" in str(exc).lower()):
        return LLMConfigError(
            "The model is not available from the selected inference provider. "
            "Change HF_PROVIDER or LLM_MODEL_ID."
        )
    if code == 429:
        return LLMRateLimitError("The LLM provider is rate limiting requests. Please retry shortly.")
    return LLMUnavailableError("The LLM is temporarily unavailable. Please retry.")


class LLMService:
    def __init__(self, settings: Settings | None = None, model: Any = None):
        self.settings = settings or get_settings()
        self._model = model
        self._lock = threading.Lock()

    @property
    def model(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._build()
        return self._model

    def _build(self) -> Any:
        s = self.settings
        if not s.hf_token:
            raise LLMConfigError("HF_TOKEN is not configured. See README -> Hugging Face setup.")
        from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

        kwargs: dict[str, Any] = dict(
            repo_id=s.llm_model_id, task="conversational", huggingfacehub_api_token=s.hf_token,
            max_new_tokens=s.llm_max_new_tokens, temperature=max(s.llm_temperature, 0.01),
            timeout=s.llm_timeout_seconds,
        )
        if s.hf_provider:
            kwargs["provider"] = s.hf_provider
        return ChatHuggingFace(llm=HuggingFaceEndpoint(**kwargs))

    # ------------------------------------------------------------------
    async def ainvoke_text(self, messages: list[BaseMessage]) -> str:
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            try:
                return _text((await self.model.ainvoke(messages)).content)
            except LLMError:
                raise
            except Exception as exc:
                code = _status(exc)
                if attempt + 1 >= attempts or not _is_retryable(exc, code):
                    log.warning("LLM call failed (%s): %s", code, exc)
                    raise _to_error(exc, code) from exc
                await asyncio.sleep(min(2**attempt, 8) + random.random() * 0.3)
        raise LLMUnavailableError()  # pragma: no cover

    async def astream_text(self, messages: list[BaseMessage]) -> AsyncIterator[str]:
        attempts = self.settings.llm_max_retries + 1
        for attempt in range(attempts):
            started = False
            try:
                async for chunk in self.model.astream(messages):
                    piece = _text(chunk.content)
                    if piece:
                        started = True
                        yield piece
                return
            except LLMError:
                raise
            except Exception as exc:
                code = _status(exc)
                # never retry once tokens were emitted: that would duplicate output
                if started or attempt + 1 >= attempts or not _is_retryable(exc, code):
                    log.warning("LLM stream failed (%s): %s", code, exc)
                    raise _to_error(exc, code) from exc
                await asyncio.sleep(min(2**attempt, 8) + random.random() * 0.3)

    async def ajson(self, system: str, user: str, schema: type[T]) -> T:
        instr = (
            "\n\nRespond with ONLY one JSON object (no prose, no markdown fences) matching this JSON schema:\n"
            + json.dumps(schema.model_json_schema())
        )
        messages = msgs(system + instr, user)
        last_err = ""
        for attempt in range(2):
            raw = await self.ainvoke_text(messages)
            try:
                return schema.model_validate(extract_json(raw))
            except (ValueError, ValidationError) as exc:
                last_err = str(exc)[:300]
                messages = messages + [
                    AIMessage(content=raw[:3000]),
                    HumanMessage(content=f"That was not valid for the schema ({last_err}). "
                                         "Reply again with ONLY the corrected JSON object."),
                ]
        raise LLMOutputError(f"Model returned invalid structured output: {last_err}")
