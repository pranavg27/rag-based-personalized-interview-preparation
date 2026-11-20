"""MCP client used by the LangGraph agent.

Tools are called over MCP (streamable HTTP via langchain-mcp-adapters). If a server is unreachable
and MCP_FALLBACK_LOCAL=true, the identical in-process implementation from app.mcp.logic is used.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from app.config import Settings
from app.errors import MCPToolError
from app.mcp.logic import LOCAL_TOOLS, TOOL_SERVER

log = logging.getLogger(__name__)
_MISSING = object()
COOLDOWN_S = 30.0


def parse_tool_output(raw: Any) -> Any:
    if isinstance(raw, tuple):
        raw = raw[0]
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", str(b)) for b in raw)
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"text": raw}
    return raw


class ToolRejected(Exception):
    """The MCP server was reached but the tool itself reported an error (no local fallback)."""


class MCPToolbox:
    def __init__(self, urls: dict[str, str], token: str, mode: str = "embedded",
                 fallback_local: bool = True, timeout: float = 90.0):
        self.urls, self.token, self.mode = urls, token, mode
        self.fallback_local, self.timeout = fallback_local, timeout
        self._tools: dict[str, Any] = {}
        self._failed_until: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self.call_log: list[tuple[str, str]] = []  # (tool, transport) – handy for tests/diagnostics

    @classmethod
    def from_settings(cls, s: Settings) -> "MCPToolbox":
        return cls(s.mcp_urls, s.mcp_internal_token, s.mcp_mode, s.mcp_fallback_local, s.mcp_timeout_seconds)

    # -- transports --------------------------------------------------------------------
    async def _load_server(self, server: str) -> None:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(
            {server: {"transport": "streamable_http", "url": self.urls[server],
                      "headers": {"Authorization": f"Bearer {self.token}"}}},
            handle_tool_errors=False,
        )
        for t in await asyncio.wait_for(client.get_tools(server_name=server), timeout=15):
            self._tools[t.name] = t

    async def _remote(self, name: str, args: dict[str, Any]) -> Any:
        server = TOOL_SERVER[name]
        if name not in self._tools:
            if time.monotonic() < self._failed_until.get(server, 0):
                raise ConnectionError(f"MCP server '{server}' in cooldown")
            async with self._lock:
                if name not in self._tools:
                    try:
                        await self._load_server(server)
                    except Exception:
                        self._failed_until[server] = time.monotonic() + COOLDOWN_S
                        raise
        tool = self._tools.get(name)
        if tool is None:
            raise ConnectionError(f"tool {name} not exposed by server {server}")
        try:
            return parse_tool_output(await asyncio.wait_for(tool.ainvoke(args), timeout=self.timeout))
        except Exception as exc:
            if type(exc).__name__ in ("ToolException", "ToolError"):
                raise ToolRejected(str(exc)) from exc
            self._tools = {k: v for k, v in self._tools.items() if TOOL_SERVER.get(k) != server}
            self._failed_until[server] = time.monotonic() + COOLDOWN_S
            raise

    # -- public ----------------------------------------------------------------------------
    async def call(self, name: str, args: dict[str, Any], default: Any = _MISSING) -> Any:
        if name not in TOOL_SERVER:
            raise MCPToolError(f"Unknown MCP tool '{name}'")
        try:
            if self.mode == "disabled":
                raise ConnectionError("MCP disabled")
            out = await self._remote(name, args)
            self.call_log.append((name, "mcp"))
            return out
        except ToolRejected as exc:
            if default is not _MISSING:
                log.warning("MCP tool %s rejected: %s", name, exc)
                return default
            raise MCPToolError(str(exc)) from exc
        except Exception as exc:
            if self.fallback_local:
                if self.mode != "disabled":
                    log.warning("MCP '%s' unreachable (%s); using local implementation", name, exc)
                try:
                    out = await LOCAL_TOOLS[name](**args)
                    self.call_log.append((name, "local"))
                    return out
                except Exception as inner:
                    if default is not _MISSING:
                        log.warning("local tool %s failed: %s", name, inner)
                        return default
                    raise
            if default is not _MISSING:
                return default
            raise MCPToolError(f"MCP tool '{name}' failed: {exc}") from exc


class InMemoryMCPToolbox(MCPToolbox):
    """Talks real MCP protocol to FastMCP servers in-process (used by tests / single-process dev)."""

    def __init__(self, servers: dict[str, Any], **kw: Any):
        super().__init__({}, "", "memory", fallback_local=False, **kw)
        self.servers = servers

    async def _remote(self, name: str, args: dict[str, Any]) -> Any:
        from fastmcp import Client
        from fastmcp.exceptions import ToolError

        try:
            async with Client(self.servers[TOOL_SERVER[name]]) as c:
                res = await c.call_tool(name, args)
        except ToolError as exc:
            raise ToolRejected(str(exc)) from exc
        if res.structured_content is not None:
            sc = res.structured_content
            return sc.get("result", sc) if set(sc) == {"result"} else sc
        return parse_tool_output(res.content[0].text)
