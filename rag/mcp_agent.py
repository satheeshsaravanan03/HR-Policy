"""Week 9 host-side policy flow that discovers and calls MCP tools."""

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
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    _citations,
    _context,
    refusal_check,
    response_text,
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


async def _run_mcp_policy_agent(query: str) -> dict[str, Any]:
    """Run the existing policy route with retrieval supplied through MCP.

    This first MCP agent exercise covers single-policy questions. The existing
    rule-based workflow selector chooses the policy-search capability; the
    MCP client verifies that the server advertises it before calling it.
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("Please provide a non-empty policy question.")

    workflow, rationale = _select_initial_workflow(query)
    if workflow != "policy_lookup":
        raise ValueError(
            "This Week 9 MCP example currently handles single-policy questions only; "
            f"the existing router selected {workflow!r}."
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
            tool = next((item for item in discovered.tools if item.name == "search_hr_policy"), None)
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
        metrics={"tool_calls": 1, "result_count": len(hits)},
        safety_findings=[],
    )
    return result
