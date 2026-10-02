"""Week 9 host-side flow that discovers and calls MCP tools."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from langchain_groq import ChatGroq
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .agent import _select_initial_workflow
from .generate import (
    Citation,
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    _citations,
    _context,
    refusal_check,
    response_text,
)
from .employee_data import (
    identifier_from_query,
    identifiers_from_query,
    lookup_leave_transaction,
    lookup_record,
)
from .retrieve import Hit, normalize_query
from .safety import unsafe_hits
from .tracing import redact
from .trajectory import write_trajectory
from .workflows import policy_audit


def _tool_payload(result: Any) -> dict[str, Any]:
    """Read a structured result, with text JSON as a compatibility fallback."""
    payload = getattr(result, "structuredContent", None)
    if payload is None:
        payload = getattr(result, "structured_content", None)
    if isinstance(payload, dict):
        return payload

    for block in getattr(result, "content", []):
        text = getattr(block, "text", None)
        if text:
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
    raise ValueError("MCP tool returned no structured policy-search result")


def _needs_input(message: str, rationale: str, stop_reason: str) -> dict[str, Any]:
    """Build a user-facing response when preflight says a tool call is premature."""
    return {
        "status": "needs_input",
        "answer": message,
        "citations": [],
        "hits": [],
        "stop_reason": stop_reason,
        "safety_findings": [],
        "route_rationale": rationale,
        "discovered_tool": "Not called",
        "tool_calls": 0,
    }


async def _run_mcp_policy_agent(query: str) -> dict[str, Any]:
    """Route policy and employee questions through discovered MCP tools."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please provide a non-empty policy question.")

    workflow, rationale = _select_initial_workflow(query)

    # The general router treats "carry-forward" as structured employee data.
    # Without a personal reference or employee identifier, it is a policy-rule
    # question and should search the policy corpus instead.
    lowered_query = query.casefold()
    personal_reference = any(
        marker in lowered_query
        for marker in ("my ", "for me", "myself", "i have", "i get", "i am")
    )
    if (
        workflow == "employee_case"
        and not identifier_from_query(query)
        and not personal_reference
        and any(term in lowered_query for term in ("carry", "forward", "leave balance", "entitlement"))
    ):
        workflow = "policy_lookup"
        rationale = "Detected a policy-rule question without an employee identifier"
    elif workflow == "policy_comparison":
        # The policy search tool accepts a combined query; answer generation
        # checks the returned evidence for both policies before comparing.
        workflow = "policy_lookup"
        rationale = "Searching MCP policy evidence for the requested comparison"

    # Preflight employee requests before starting the MCP server. This catches
    # missing identifiers and missing local employee/transaction records
    # without making a tool call that is already known to fail.
    if workflow in {"employee_case", "employee_policy_case"}:
        identifier = identifier_from_query(query)
        if not identifier:
            return _needs_input(
                "Please provide an employee ID (for example EMP-001) or the email on the employee record.",
                rationale,
                "employee_identifier_required",
            )
        employee_record = lookup_record(identifier)
        if employee_record is None:
            return _needs_input(
                f"I could not find an employee record for `{identifier}` in the available records, so I did not call an employee tool.",
                rationale,
                "employee_record_not_found",
            )
        if workflow == "employee_policy_case" and any(
            term in lowered_query for term in ("compensatory", "comp off", "comp-off")
        ):
            transaction = lookup_leave_transaction(employee_record["employee_id"])
            if transaction is None or transaction.get("compensatory_leave_balance") is None:
                return _needs_input(
                    f"I found `{employee_record['employee_id']}`, but its compensatory-leave transaction detail is not available, so I did not call the summary tool.",
                    rationale,
                    "employee_transaction_detail_missing",
                )

    if workflow == "employee_comparison":
        identifiers = identifiers_from_query(query)
        if len(identifiers) < 2:
            return _needs_input(
                "Please provide two employee IDs or emails before requesting an employee comparison.",
                rationale,
                "two_employee_identifiers_required",
            )
        if len(identifiers) != 2 or identifiers[0] == identifiers[1]:
            return _needs_input(
                "The comparison needs two different employee records. Please check the IDs or emails and try again.",
                rationale,
                "two_distinct_employee_identifiers_required",
            )
        missing_records = [key for key in identifiers if lookup_record(key) is None]
        if missing_records:
            missing_label = ", ".join("`" + key + "`" for key in missing_records)
            return _needs_input(
                f"I could not find employee record(s) for {missing_label}, so I did not call the comparison tool.",
                rationale,
                "employee_record_not_found",
            )
        records = [lookup_record(key) for key in identifiers]
        if records[0]["employee_id"] == records[1]["employee_id"]:
            return _needs_input(
                "Those two identifiers refer to the same employee. Please provide two different employee records.",
                rationale,
                "two_distinct_employee_records_required",
            )
        missing_transactions = [
            record["employee_id"]
            for record in records
            if record is not None and lookup_leave_transaction(record["employee_id"]) is None
        ]
        if missing_transactions:
            missing_label = ", ".join("`" + key + "`" for key in missing_transactions)
            return _needs_input(
                f"Leave-transaction records are missing for {missing_label}, so I did not call the comparison tool.",
                rationale,
                "employee_transaction_record_missing",
            )

    server_path = Path(__file__).resolve().parent.parent / "mcp_server.py"
    # The MCP SDK launches stdio servers with a restricted environment by
    # default. Forward only the retrieval settings/secrets this server needs.
    allowed_server_env = (
        "VECTOR_STORE", "EMBED_BACKEND", "QDRANT_URL", "QDRANT_API_KEY",
        "QDRANT_TIMEOUT", "QDRANT_COLLECTION", "QDRANT_DIMENSIONS",
        "GOOGLE_API_KEY", "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "no_proxy",
    )
    server_env = {key: os.environ[key] for key in allowed_server_env if os.environ.get(key)}
    server = StdioServerParameters(command=sys.executable, args=[str(server_path)], env=server_env)

    print("Starting local MCP server and connecting over stdio...", file=sys.stderr, flush=True)
    async with stdio_client(server) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            discovered = await session.list_tools()
            tools = {item.name: item for item in discovered.tools}

            if workflow == "employee_policy_case":
                tool_name = "get_employee_leave_summary"
                tool = tools.get(tool_name)
                identifier = identifier_from_query(query)
                if tool is None or not identifier:
                    raise RuntimeError(f"MCP server must advertise {tool_name} for employee questions")
                call_result = await session.call_tool(tool_name, arguments={"identifier": identifier})
                if getattr(call_result, "isError", False) or getattr(call_result, "is_error", False):
                    raise RuntimeError(f"{tool_name} returned an MCP tool error")
                payload = _tool_payload(call_result)
                if payload.get("status") != "success":
                    status = payload.get("status", "blocked")
                    reason = payload.get("reason", "required policy evidence was not available")
                    return {
                        "status": "needs_input" if status == "needs_input" else "blocked",
                        "answer": payload.get("answer") or f"I could not complete this employee request: {reason}.",
                        "citations": [],
                        "hits": [],
                        "stop_reason": "employee_policy_evidence_missing",
                        "safety_findings": [],
                        "route_rationale": rationale,
                        "discovered_tool": tool_name,
                        "tool_calls": 1,
                    }
                values = payload.get("values", {})
                requested_fields = []
                if any(term in lowered_query for term in ("carry", "forward")):
                    requested_fields.append(("carry_forward", "carry-forward limit"))
                if any(term in lowered_query for term in ("compensatory", "comp off", "comp-off")):
                    requested_fields.append(("compensatory_leave", "compensatory-leave detail"))
                if any(term in lowered_query for term in ("annual entitlement", "annual leave", "entitlement")):
                    requested_fields.append(("annual_leave_entitlement", "annual-leave entitlement"))
                missing_fields = [label for key, label in requested_fields if key not in values]
                if missing_fields:
                    return {
                        "status": "blocked",
                        "answer": f"I found `{payload.get('employee_id', identifier)}`, but the available policy or employee data does not include the requested {', '.join(missing_fields)}. I cannot provide that value.",
                        "citations": [],
                        "hits": [],
                        "stop_reason": "requested_employee_detail_missing",
                        "safety_findings": [],
                        "route_rationale": rationale,
                        "discovered_tool": tool_name,
                        "tool_calls": 1,
                    }
                lines = [f"Employee `{payload.get('employee_id', identifier)}` leave summary:"]
                labels = {
                    "current_leave_balance": "Current leave balance",
                    "annual_leave_entitlement": "Annual leave entitlement",
                    "carry_forward": "Carry-forward available",
                    "compensatory_leave": "Compensatory leave available",
                }
                for key, label in labels.items():
                    if key in values:
                        lines.append(f"- {label}: **{values[key]:g} days**")
                lines.append(f"\nPolicy: `{payload.get('policy_id', '')}`")
                citations = [Citation(chunk_id=c["chunk_id"], policy_id=c["policy_id"], section=c.get("section", ""), resolves=True) for c in payload.get("citations", [])]
                return {"status": "success", "answer": "\n".join(lines), "citations": citations, "hits": [], "stop_reason": "employee_summary_tool_complete", "safety_findings": [], "route_rationale": rationale, "discovered_tool": tool_name}

            if workflow == "employee_case":
                tool_name = "get_employee_record"
                tool = tools.get(tool_name)
                identifier = identifier_from_query(query)
                if tool is None or not identifier:
                    raise RuntimeError(f"MCP server must advertise {tool_name} for employee-record requests")
                call_result = await session.call_tool(tool_name, arguments={"identifier": identifier})
                if getattr(call_result, "isError", False) or getattr(call_result, "is_error", False):
                    raise RuntimeError(f"{tool_name} returned an MCP tool error")
                payload = _tool_payload(call_result)
                lines = [f"Employee record found for `{payload.get('employee_id', identifier)}`:"]
                for key, label in (("region", "Region"), ("policy_id", "Policy"), ("current_leave_balance", "Current leave balance")):
                    if payload.get(key) is not None:
                        suffix = " days" if key == "current_leave_balance" else ""
                        value = payload[key]
                        if key == "current_leave_balance" and isinstance(value, (int, float)):
                            value = f"{value:g}"
                        lines.append(f"- {label}: **{value}{suffix}**")
                return {"status": "success", "answer": "\n".join(lines), "citations": [], "hits": [], "stop_reason": "employee_record_tool_complete", "safety_findings": [], "route_rationale": rationale, "discovered_tool": tool_name, "tool_calls": 1}

            if workflow == "employee_comparison":
                tool_name = "compare_employee_leave_transactions"
                tool = tools.get(tool_name)
                identifiers = identifiers_from_query(query)
                if tool is None or len(identifiers) < 2:
                    raise RuntimeError(f"MCP server must advertise {tool_name} and the question must contain two employees")
                call_result = await session.call_tool(tool_name, arguments={"first_identifier": identifiers[0], "second_identifier": identifiers[1]})
                if getattr(call_result, "isError", False) or getattr(call_result, "is_error", False):
                    raise RuntimeError(f"{tool_name} returned an MCP tool error")
                payload = _tool_payload(call_result)
                lines = ["Employee leave-transaction comparison:", ""]
                for item in payload.get("employees", []):
                    tx = item.get("transaction", {})
                    values = {
                        key: tx.get(key) if tx.get(key) is not None else "not recorded"
                        for key in ("leave_filed_days", "leave_used_days", "pending_leave_days", "compensatory_leave_balance")
                    }
                    lines.append(f"- **{item['employee_id']}**: filed {values['leave_filed_days']}; used {values['leave_used_days']}; pending {values['pending_leave_days']}; compensatory {values['compensatory_leave_balance']} days.")
                lines.append(f"\nDifferences (second minus first): `{payload.get('differences_second_minus_first', {})}`")
                return {"status": "success", "answer": "\n".join(lines), "citations": [], "hits": [], "stop_reason": "employee_comparison_tool_complete", "safety_findings": [], "route_rationale": rationale, "discovered_tool": tool_name}

            if workflow != "policy_lookup":
                raise ValueError(f"MCP mode does not support workflow {workflow!r} yet")

            tool_name = "search_hr_policy"
            tool = tools.get(tool_name)
            if tool is None:
                raise RuntimeError("The MCP server did not advertise search_hr_policy")
            print("Connected; discovered search_hr_policy. Searching indexed policies...", file=sys.stderr, flush=True)

            # Tool selection comes from the host's existing workflow router;
            # validate this call against the live schema before sending it.
            arguments = {
                "query": normalize_query(query),
                "strategy": "structure",
                "method": "hybrid",
                "top_k": 5,
            }
            tool_description = tool.model_dump(mode="json", by_alias=True)
            input_schema = tool_description.get("inputSchema", tool_description.get("input_schema", {}))
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])
            if set(arguments) - set(properties):
                raise RuntimeError("The discovered search_hr_policy schema does not support this call")
            if set(required) - set(arguments):
                raise RuntimeError("The discovered search_hr_policy schema requires additional arguments")
            call_result = await session.call_tool(
                tool.name,
                arguments=arguments,
            )
            if getattr(call_result, "isError", False) or getattr(call_result, "is_error", False):
                details = "; ".join(
                    str(getattr(block, "text", ""))
                    for block in getattr(call_result, "content", [])
                    if getattr(block, "text", None)
                )
                raise RuntimeError(f"search_hr_policy reported an MCP tool error: {details}")
            payload = _tool_payload(call_result)

            if not payload.get("results"):
                return {
                    "status": "refused",
                    "answer": "I could not find matching policy evidence in the indexed documents, so I cannot verify this answer.",
                    "citations": [],
                    "hits": [],
                    "stop_reason": "no_policy_evidence_found",
                    "safety_findings": [],
                    "route_rationale": rationale,
                    "discovered_tool": tool.name,
                    "tool_calls": 1,
                }

    print(f"Policy search returned {len(payload.get('results', []))} result(s).", file=sys.stderr, flush=True)
    hits = [
        Hit(
            rank=int(item["rank"]),
            chunk_id=str(item["chunk_id"]),
            policy_id=str(item["policy_id"]),
            section=str(item.get("section", "")),
            region=str(item.get("region", "")),
            source_file=str(item.get("source_file", "")),
            score=float(item["score"]),
            content=str(item["text"]),
            semantic_score=(
                float(item["semantic_score"])
                if item.get("semantic_score") is not None
                else None
            ),
        )
        for item in payload.get("results", [])
    ]

    findings = unsafe_hits(hits)
    if findings:
        return {
            "status": "refused",
            "answer": "I cannot use the retrieved document because it contains an unsafe instruction-like passage.",
            "citations": [],
            "hits": hits,
            "stop_reason": "untrusted_document_instruction",
            "safety_findings": findings,
        }

    refuse, _gate, reason = refusal_check(normalize_query(query), hits)
    if refuse:
        return {
            "status": "refused",
            "answer": f"I cannot answer this from the indexed policy documents.\nReason: {reason}.",
            "citations": [],
            "hits": hits,
            "stop_reason": "refusal_gate_fired",
            "safety_findings": [],
        }

    print("Evidence passed safety checks; generating a cited answer with Groq...", file=sys.stderr, flush=True)
    llm = ChatGroq(model=GENERATION_MODEL, temperature=0.0, timeout=60, max_retries=0)
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context chunks:\n{_context(hits)}\n\nQuestion: {query}"),
        ]
    )
    answer = response_text(response.content)
    citations = _citations(answer, hits)
    audit = policy_audit(query, answer_text=answer, citations=citations, hits=hits)
    if audit["status"] != "success":
        return {
            "status": "refused",
            "answer": "I could not verify the generated answer against the retrieved policy evidence.",
            "citations": audit.get("citations", []),
            "hits": hits,
            "stop_reason": "policy_audit_failed",
            "safety_findings": [],
        }

    return {
        "status": "success",
        "answer": answer,
        "citations": citations,
        "hits": hits,
        "stop_reason": "answer_generated_and_audited",
        "safety_findings": [],
        "route_rationale": rationale,
        "discovered_tool": tool.name,
    }


async def run_mcp_policy_agent(query: str) -> dict[str, Any]:
    """Run the MCP policy agent and append a redacted trajectory record.

    The record includes validated tool arguments and source identifiers, but
    never the retrieved policy text or environment values/secrets.
    """
    safe_query = query if isinstance(query, str) else "<invalid query>"
    validated_input = {
        "query": redact(normalize_query(safe_query)) if safe_query != "<invalid query>" else safe_query,
        "strategy": "structure",
        "method": "hybrid",
        "top_k": 5,
    }
    try:
        result = await _run_mcp_policy_agent(query)
    except Exception as exc:
        write_trajectory(
            query=safe_query,
            steps=[{
                "step": 1,
                "workflow": "mcp_tool_call",
                "tool_name": "search_hr_policy",
                "validated_input": validated_input,
                "result_status": "error",
                "result_summary": "MCP request did not complete; retrieved content omitted",
                "stop_reason": type(exc).__name__,
            }],
            status="error",
            workflows_called=["mcp_tool_call"],
            answer="MCP request failed.",
            stop_reason=type(exc).__name__,
            metrics={"tool_calls_completed": 0, "result_count": 0},
        )
        raise

    hits = result.get("hits", [])
    status = result.get("status", "error")
    tool_name = result.get("discovered_tool", "search_hr_policy")
    stop_reason = result.get("stop_reason", "unknown")
    tool_calls = int(result.get("tool_calls", 1))
    if tool_calls == 0:
        write_trajectory(
            query=safe_query,
            steps=[{
                "step": 1,
                "workflow": "pre_tool_validation",
                "tool_name": "Not called",
                "validated_input": {"query": redact(normalize_query(safe_query))},
                "result_status": status,
                "result_summary": "Request stopped because required employee information or records were missing",
                "stop_reason": stop_reason,
            }],
            status=status,
            workflows_called=["pre_tool_validation"],
            answer=result.get("answer", ""),
            stop_reason=stop_reason,
            metrics={"tool_calls_completed": 0, "result_count": 0},
            safety_findings=[],
        )
        return result

    result_summary = {
        "result_count": len(hits),
        "sources": [
            {"policy_id": hit.policy_id, "chunk_id": hit.chunk_id}
            for hit in hits
        ],
    }
    steps = [{
        "step": 1,
        "workflow": "mcp_tool_call",
        "tool_name": tool_name,
        "validated_input": validated_input,
        "result_status": "success" if hits else "empty",
        "result_summary": result_summary,
        "stop_reason": "",
    }, {
        "step": 2,
        "workflow": "safety_and_policy_audit",
        "tool_name": tool_name,
        "result_status": status,
        "result_summary": "Retrieved text omitted; safety checks and citation audit applied",
        "stop_reason": stop_reason,
    }]
    workflows = ["mcp_tool_call", "safety_checks"]
    if stop_reason == "answer_generated_and_audited":
        workflows.append("policy_audit")
    write_trajectory(
        query=safe_query,
        steps=steps,
        status=status,
        workflows_called=workflows,
        answer=result.get("answer", ""),
        stop_reason=stop_reason,
        metrics={"tool_calls": tool_calls, "result_count": len(hits)},
        safety_findings=[],
    )
    return result
