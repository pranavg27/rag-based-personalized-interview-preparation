"""Server-Sent Events helpers: formatting, keep-alives, disconnect + cancellation handling."""
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from app.errors import public_message

log = logging.getLogger(__name__)

SSE_HEADERS = {"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"}


def format_sse(event: str, data: dict[str, Any], event_id: int | None = None) -> str:
    lines = []
    if event_id is not None:
        lines.append(f"id: {event_id}")
    lines.append(f"event: {event}")
    payload = json.dumps(data, ensure_ascii=False, default=str)
    lines.extend(f"data: {part}" for part in payload.split("\n"))
    return "\n".join(lines) + "\n\n"


_SENTINEL = object()


async def sse_stream(
    request: Any, events: AsyncIterator[dict[str, Any]], keepalive: float = 15.0
) -> AsyncIterator[str]:
    """Pump `events` through a queue so we can emit keep-alives, notice client disconnects,
    cancel the producer, and *always* finish with a `done` event (unless the client left)."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=512)

    async def pump() -> None:
        try:
            async for ev in events:
                await queue.put(ev)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # surfaced to the client as an `error` event
            log.exception("stream producer failed")
            await queue.put({"type": "error", "message": public_message(exc)})
        finally:
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(_SENTINEL)

    task = asyncio.create_task(pump())
    event_id = 0
    client_gone = False
    try:
        while True:
            try:
                item = await asyncio.wait_for(queue.get(), timeout=keepalive)
            except asyncio.TimeoutError:
                if request is not None and await request.is_disconnected():
                    client_gone = True
                    break
                yield ": keepalive\n\n"
                continue
            if item is _SENTINEL:
                break
            event_id += 1
            yield format_sse(item.get("type", "message"), item, event_id)
            if request is not None and await request.is_disconnected():
                client_gone = True
                break
        if not client_gone:
            yield format_sse("done", {"type": "done"}, event_id + 1)
    finally:
        if not task.done():
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await task
