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
from rag.employee_data import lookup_leave_transaction, lookup_record, search_leave_transactions
from rag.workflows import employee_policy_case


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
def get_employee_record(identifier: str) -> dict[str, Any]:
    """Read a limited synthetic employee record by ID or email.

    This learning demo validates and scopes the lookup to one employee. It does
    not authenticate the caller; use only the synthetic records in this repo.
    """
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be a non-empty string")
    raw_identifier = identifier.strip()
    normalized_id = raw_identifier.upper()
    if not (re.fullmatch(r"EMP-\d{3}", normalized_id) or re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", raw_identifier)):
        raise ValueError("identifier must be an employee ID such as EMP-001 or an email address")

    record = lookup_record(normalized_id if normalized_id.startswith("EMP-") else raw_identifier.lower())
    if record is None:
        raise ValueError("No employee record found for that ID")

    # Only return identity and current-state fields. Policy limits and mutable
    # leave transactions are exposed by separate tools.
    allowed_fields = (
        "employee_id", "email", "name", "region", "policy_id",
        "experience_years", "current_leave_balance",
    )
    return {field: record[field] for field in allowed_fields}


@mcp.tool()
def get_employee_leave_transactions(employee_id: str) -> dict[str, Any]:
    """Read mutable leave-transaction data for one synthetic employee."""
    normalized_id = employee_id.strip().upper() if isinstance(employee_id, str) else ""
    if not re.fullmatch(r"EMP-\d{3}", normalized_id):
        raise ValueError("employee_id must use the format EMP-001")
    if lookup_record(normalized_id) is None:
        raise ValueError("No employee record found for that ID")
    transaction = lookup_leave_transaction(normalized_id)
    if transaction is None:
        raise ValueError("No leave transaction record found for that ID")
    return transaction


@mcp.tool()
def get_employee_leave_summary(identifier: str) -> dict[str, Any]:
    """Retrieve policy evidence and calculate a leave summary by ID or email."""
    if not isinstance(identifier, str) or not identifier.strip():
        raise ValueError("identifier must be an employee ID or email")
    result = employee_policy_case(
        f"For {identifier.strip()}, show current leave balance, annual entitlement, carry-forward, and compensatory leave.",
        strategy="structure", top_k=5,
    )
    if result.get("status") != "success":
        record = result.get("record") or {}
        return {
            "status": result.get("status", "blocked"),
            "reason": result.get("reason") or "Unable to produce a policy-backed leave summary",
            "answer": result.get("answer", ""),
            "missing_information": result.get("missing_information", []),
            "employee_id": record.get("employee_id"),
            "policy_id": record.get("policy_id"),
        }
    calculation = result.get("calculation") or {}
    record = result.get("record") or {}
    return {
        "status": "success",
        "employee_id": record.get("employee_id"),
        "policy_id": record.get("policy_id"),
        "values": calculation.get("values", {}),
        "citations": [
            {"chunk_id": hit.chunk_id, "policy_id": hit.policy_id, "section": hit.section}
            for hit in result.get("hits", [])
        ],
    }


@mcp.tool()
def compare_employee_leave_transactions(first_identifier: str, second_identifier: str) -> dict[str, Any]:
    """Compare leave transactions for two synthetic employees by ID or email."""
    identifiers = (first_identifier, second_identifier)
    if any(not isinstance(value, str) or not value.strip() for value in identifiers):
        raise ValueError("Both employee identifiers are required")
    records = []
    for identifier in identifiers:
        record = lookup_record(identifier.strip().lower())
        if record is None:
            raise ValueError(f"No employee record found for {identifier}")
        transaction = lookup_leave_transaction(record["employee_id"])
        if transaction is None:
            raise ValueError(f"No leave transaction record found for {record['employee_id']}")
        records.append({
            "employee_id": record["employee_id"],
            "email": record["email"],
            "policy_id": record["policy_id"],
            "transaction": transaction,
        })
    if records[0]["employee_id"] == records[1]["employee_id"]:
        raise ValueError("The two identifiers must refer to different employees")

    fields = ("compensatory_leave_balance", "leave_filed_days", "leave_used_days", "pending_leave_days")
    differences = {}
    left, right = records[0]["transaction"], records[1]["transaction"]
    for field in fields:
        left_value, right_value = left.get(field), right.get(field)
        if left_value is not None and right_value is not None:
            differences[field] = right_value - left_value
        else:
            differences[field] = None
    return {
        "employees": records,
        "differences_second_minus_first": differences,
    }


@mcp.tool()
def find_employee_ids_by_leave_transactions(
    field: str,
    operator: str = "gt",
    threshold: float = 0,
    second_field: str | None = None,
    second_operator: str | None = None,
    second_threshold: float | None = None,
    region: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Find matching employee IDs using one or two leave-data conditions.

    Conditions are combined with AND. This tool returns IDs only; call
    get_employee_details separately to fetch matching employee information.
    """
    if not isinstance(field, str) or not field.strip():
        raise ValueError("field is required")
    matches = search_leave_transactions(
        field=field.strip(),
        operator=operator.strip().lower() if isinstance(operator, str) else operator,
        threshold=threshold,
        second_field=second_field.strip() if isinstance(second_field, str) and second_field.strip() else None,
        second_operator=second_operator.strip().lower() if isinstance(second_operator, str) else second_operator,
        second_threshold=second_threshold,
        region=region.strip() if isinstance(region, str) and region.strip() else None,
        policy_id=policy_id.strip() if isinstance(policy_id, str) and policy_id.strip() else None,
    )
    return {
        "status": "success",
        "criteria": {
            "field": field.strip(),
            "operator": operator,
            "threshold": float(threshold),
            "second_field": second_field,
            "second_operator": second_operator,
            "second_threshold": second_threshold,
            "region": region,
            "policy_id": policy_id,
        },
        "count": len(matches),
        "employee_ids": [item["employee_id"] for item in matches],
    }


@mcp.tool()
def find_employee_ids_by_employee_criteria(
    field: str,
    operator: str = "gt",
    threshold: float = 0,
    second_field: str | None = None,
    second_operator: str | None = None,
    second_threshold: float | None = None,
    region: str | None = None,
    policy_id: str | None = None,
) -> dict[str, Any]:
    """Find employee IDs by numeric employee-record criteria such as experience or balance.

    Conditions are combined with AND. This tool returns IDs only; call
    get_employee_details separately for the matching employee records.
    """
    return find_employee_ids_by_leave_transactions(
        field=field,
        operator=operator,
        threshold=threshold,
        second_field=second_field,
        second_operator=second_operator,
        second_threshold=second_threshold,
        region=region,
        policy_id=policy_id,
    )


@mcp.tool()
def get_employee_details(employee_ids: list[str]) -> dict[str, Any]:
    """Fetch employee identity, balance, and transaction details by IDs."""
    if not isinstance(employee_ids, list) or not employee_ids:
        raise ValueError("employee_ids must be a non-empty list")
    if len(employee_ids) > 100:
        raise ValueError("At most 100 employee IDs can be requested at once")
    employees = []
    seen: set[str] = set()
    for raw_id in employee_ids:
        employee_id = raw_id.strip().upper() if isinstance(raw_id, str) else ""
        if not re.fullmatch(r"EMP-\d{3}", employee_id):
            raise ValueError("Each employee ID must use the format EMP-001")
        if employee_id in seen:
            continue
        seen.add(employee_id)
        record = lookup_record(employee_id)
        if record is None:
            continue
        employees.append({
            "employee_id": record["employee_id"],
            "email": record["email"],
            "name": record["name"],
            "region": record["region"],
            "policy_id": record["policy_id"],
            "experience_years": record["experience_years"],
            "current_leave_balance": record["current_leave_balance"],
            "transaction": lookup_leave_transaction(employee_id),
        })
    return {"status": "success", "count": len(employees), "employees": employees}


if __name__ == "__main__":
    mcp.run(transport="stdio")
