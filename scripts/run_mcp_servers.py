"""Run the four MCP servers as standalone processes-in-one (MCP_MODE=external).

    python scripts/run_mcp_servers.py

Ports come from MCP_*_URL in .env (defaults 8101-8104). Use CHROMA_MODE=http so the question server and the
API share one Chroma instance (PersistentClient is not multi-process safe). Requests need
`Authorization: Bearer $MCP_INTERNAL_TOKEN`.
"""
import asyncio
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
import uvicorn  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.mcp.servers import SERVERS  # noqa: E402
from app.mcp.servers._common import http_app  # noqa: E402


async def main() -> None:
    s = get_settings()
    s.mcp_mode = "external"
    servers = []
    for name, srv in SERVERS.items():
        port = urlparse(s.mcp_urls[name]).port
        cfg = uvicorn.Config(http_app(srv, s.mcp_internal_token), host="0.0.0.0", port=port, log_level="info")
        servers.append(uvicorn.Server(cfg))
        print(f"MCP '{name}' -> {s.mcp_urls[name]}")
    await asyncio.gather(*(srv.serve() for srv in servers))


if __name__ == "__main__":
    asyncio.run(main())
