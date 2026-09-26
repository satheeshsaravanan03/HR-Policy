"""Local-only Streamable HTTP entry point with a demo bearer-token gate.

This is a learning example, not production authentication or OAuth. Keep the
stdio entry point in mcp_server.py unchanged and use synthetic data only.

PowerShell:
    $env:MCP_DEMO_TOKEN = [guid]::NewGuid().ToString("N")
    py mcp_http_server.py
"""

from __future__ import annotations

import hmac
import json
import os

import uvicorn

from mcp_server import mcp


class DemoBearerTokenGate:
    """Require one environment-provided bearer token on the MCP HTTP route."""

    def __init__(self, app, expected_token: str) -> None:
        self.app = app
        self.expected_token = expected_token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http" or scope.get("path") != "/mcp":
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        authorization = headers.get(b"authorization", b"").decode("latin-1")
        scheme, separator, supplied_token = authorization.partition(" ")
        if (
            not separator
            or scheme.lower() != "bearer"
            or not hmac.compare_digest(supplied_token, self.expected_token)
        ):
            body = json.dumps({"error": "authentication_required"}).encode("utf-8")
            await send({
                "type": "http.response.start",
                "status": 401,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"www-authenticate", b"Bearer"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return

        await self.app(scope, receive, send)


def main() -> None:
    token = os.environ.get("MCP_DEMO_TOKEN", "")
    if len(token) < 24:
        raise SystemExit(
            "Set MCP_DEMO_TOKEN to a random value of at least 24 characters "
            "before starting this local demo server."
        )

    app = DemoBearerTokenGate(mcp.streamable_http_app(), token)
    print("Demo-authenticated MCP server: http://127.0.0.1:8000/mcp")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")


if __name__ == "__main__":
    main()
