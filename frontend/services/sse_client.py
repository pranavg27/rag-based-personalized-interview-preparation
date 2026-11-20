"""Minimal Server-Sent-Events consumer (httpx streaming)."""
from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from services.api_client import APIClient, APIError


def iter_sse(client: APIClient, path: str, method: str = "GET", json_body: dict | None = None) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield (event, data) pairs. Handles multi-line data, keep-alive comments and early termination."""
    timeout = httpx.Timeout(15.0, read=300.0)
    try:
        with httpx.stream(method, client.base + path, headers=client.headers, json=json_body, timeout=timeout) as r:
            if r.status_code >= 400:
                r.read()
                try:
                    detail = r.json().get("detail", r.text)
                except Exception:
                    detail = r.text
                raise APIError(r.status_code, str(detail))
            event, data_lines = "message", []
            for line in r.iter_lines():
                if line == "":
                    if data_lines:
                        try:
                            yield event, json.loads("\n".join(data_lines))
                        except json.JSONDecodeError:
                            pass
                        if event == "done":
                            return
                    event, data_lines = "message", []
                elif line.startswith(":"):
                    continue  # keep-alive
                elif line.startswith("event:"):
                    event = line[6:].strip()
                elif line.startswith("data:"):
                    data_lines.append(line[5:].lstrip())
    except httpx.HTTPError as exc:
        raise APIError(0, f"Stream interrupted: {exc}") from exc
