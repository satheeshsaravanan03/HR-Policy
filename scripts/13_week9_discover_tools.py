"""Connect to the local MCP server over stdio and display discovered tools.

Run from the project root with:
    python scripts/13_week9_discover_tools.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = PROJECT_ROOT / "mcp_server.py"


async def discover_tools() -> int:
    """Start the server as a subprocess, initialize MCP, and list its tools."""
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    try:
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                initialized = await session.initialize()
                response = await session.list_tools()

                print(f"Connected to MCP server: {initialized.serverInfo.name}")
                print("Discovered tools:")
                for tool in response.tools:
                    print(json.dumps(tool.model_dump(mode="json", by_alias=True), indent=2))

                if not response.tools:
                    print("No tools were advertised by the server.")
                    return 1
        # Exiting both context managers closes the MCP session and child process.
        return 0
    except Exception as exc:
        print(f"MCP discovery failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(discover_tools()))
