"""Minimal local MCP server for Week 9, Step 2.

Run with: python mcp_server.py

The server uses stdio: stdout belongs to MCP protocol messages, so do not add
print-based diagnostics here. The demo tool is intentionally harmless and
does not read application data or call external services.
"""

import asyncio
import json
import re
import sys
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from rag.employee_data import lookup_record


mcp = FastMCP("HR Policy Learning Demo")


@mcp.tool()
def explain_mcp_demo() -> str:
    """Return a short explanation from the Week 9 learning server."""
    return (
        "This response came from the local Week 9 MCP demo server. "
        "The host asks through an MCP client, and this server runs the tool."
    )


@mcp.tool()
async def search_hr_policy(
    query: str,
    policy_id: str | None = None,
    region: str | None = None,
    strategy: str = "structure",
    method: str = "semantic",
    top_k: int = 5,
) -> dict[str, Any]:
    """Search indexed HR policy documents and return evidence with citations.

    Use an optional policy_id or region to limit the search. Results include
    source file, policy ID, section, chunk ID, score, and matching text.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if policy_id is not None and (not isinstance(policy_id, str) or not policy_id.strip()):
        raise ValueError("policy_id must be omitted or a non-empty string")
    if region is not None and (not isinstance(region, str) or not region.strip()):
        raise ValueError("region must be omitted or a non-empty string")
    if strategy not in {"recursive", "structure"}:
        raise ValueError("strategy must be 'recursive' or 'structure'")
    if method not in {"semantic", "bm25", "hybrid"}:
        raise ValueError("method must be 'semantic', 'bm25', or 'hybrid'")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 5:
        raise ValueError("top_k must be an integer from 1 to 5")

    # Isolate native/vector-store work from the MCP server's stdio protocol.
    worker_path = Path(__file__).resolve().parent / "scripts" / "15_week9_policy_search_worker.py"
    request = {
        "query": query.strip(),
        "policy_id": policy_id.strip() if policy_id else None,
        "region": region.strip() if region else None,
        "strategy": strategy,
        "method": method,
        "top_k": top_k,
    }
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(worker_path),
        cwd=str(worker_path.parent.parent),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await process.communicate(json.dumps(request).encode("utf-8"))
    if process.returncode != 0:
        detail = stderr.decode("utf-8", errors="replace")[-1500:]
        raise RuntimeError(f"Policy search worker failed: {detail or process.returncode}")
    try:
        return json.loads(stdout.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        preview = stdout[:120].decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")
        raise RuntimeError(f"Policy search worker returned invalid JSON; stdout began {preview!r}") from exc


@mcp.tool()
def get_employee_record(employee_id: str) -> dict[str, Any]:
    """Read a limited synthetic employee record by one explicit employee ID.

    This learning demo validates and scopes the lookup to one employee. It does
    not authenticate the caller; use only the synthetic records in this repo.
    """
    if not isinstance(employee_id, str) or not employee_id.strip():
        raise ValueError("employee_id must be a non-empty string")
    normalized_id = employee_id.strip().upper()
    if not re.fullmatch(r"EMP-\d{3}", normalized_id):
        raise ValueError("employee_id must use the format EMP-001")

    record = lookup_record(normalized_id)
    if record is None:
        raise ValueError("No employee record found for that ID")

    # Only return fields needed for the demo. Exclude name, email, and other
    # identifiers even though they exist in the local synthetic data.
    allowed_fields = (
        "employee_id", "region", "policy_id", "experience_years",
        "current_leave_balance", "compensatory_leave_balance",
        "annual_leave_entitlement", "carry_forward_cap", "compensatory_leave_cap",
    )
    return {field: record[field] for field in allowed_fields}


if __name__ == "__main__":
    mcp.run(transport="stdio")
