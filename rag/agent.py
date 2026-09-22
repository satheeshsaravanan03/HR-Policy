"""Week 7 Agent and Fixed Workflow implementations.

Provides two independent execution paths for policy questions:
1. run_fixed_workflow: Fixed, deterministic sequence of RAG steps.
2. run_agent: Small, visible agent loop with dynamic workflow selection,
   structured step inspection, and strict safety guardrails.

Both functions return structured results with identical metric shapes so
Compare mode can display them side-by-side.
"""

from __future__ import annotations

import re
import time
from typing import Any

from langchain_groq import ChatGroq

from .generate import (
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    Citation,
    _citations,
    _context,
    refusal_check,
    response_text,
)
from .retrieve import (
    HYBRID,
    RERANK_LOCAL,
    Hit,
    normalize_query,
    search,
    understand_query,
)
from .employee_data import identifier_from_query
from .employee_data import identifiers_from_query
from .safety import unsafe_hits
from .trajectory import write_trajectory
from .workflows import (
    employee_case,
    employee_comparison,
    employee_policy_case,
    evidence_validation,
    policy_applicability,
    policy_audit,
    policy_comparison,
    policy_lookup,
)


def _agent_step_record(
    step: int,
    decision: str,
    workflow: str,
    input_str: str,
    result_status: str,
    evidence_summary: str,
    next_decision: str,
    stop_reason: str = "",
) -> dict[str, Any]:
    return {
        "step": step,
        "decision": decision,
        "workflow": workflow,
        "input": input_str,
        "result_status": result_status,
        "evidence_summary": evidence_summary,
        "next_decision": next_decision,
        "stop_reason": stop_reason,
    }


# ----------------------------------------------------------------------
# 1. Fixed Workflow Path (for Compare mode)
# ----------------------------------------------------------------------
def run_fixed_workflow(
    query: str,
    *,
    strategy: str = "structure",
    top_k: int = 5,
    method: str = HYBRID,
    rerank: str = RERANK_LOCAL,
) -> dict[str, Any]:
    """Execute the fixed linear RAG sequence.

    Steps:
      1. Normalize query & infer metadata
      2. Retrieve & rerank candidates
      3. Apply refusal gate
      4. If approved, generate cited answer with LLM
    """
    start_time = time.perf_counter()
    steps: list[dict[str, Any]] = []
    tool_calls = 0
    llm_calls = 0

    # Step 1: Normalize & infer metadata
    norm_query = normalize_query(query)
    _inferred_policy, inferred_region = understand_query(norm_query)
    steps.append(
        _agent_step_record(
            step=1,
            decision="Execute deterministic normalization and metadata extraction",
            workflow="normalize_and_infer",
            input_str=query,
            result_status="success",
            evidence_summary=f"Normalized: '{norm_query}', region: {inferred_region or 'none'}",
            next_decision="Run hybrid retrieval and reranking",
        )
    )

    # Step 2: Retrieve & rerank
    tool_calls += 1
    hits = search(
        strategy,
        norm_query,
        top_k=top_k,
        region=inferred_region,
        method=method,
        rerank=rerank,
    )
    safety_findings = unsafe_hits(hits)
    steps.append(
        _agent_step_record(
            step=2,
            decision="Retrieve and rerank top policy chunks",
            workflow="hybrid_search_and_rerank",
            input_str=norm_query,
            result_status="success" if hits else "empty",
            evidence_summary=f"Retrieved {len(hits)} chunk(s)",
            next_decision="Evaluate refusal gate",
        )
    )

    # Step 3: Reject instruction-like text from retrieved documents before any
    # model sees it. Retrieved content is data, never an instruction source.
    if safety_findings:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        return {
            "mode": "fixed_workflow", "status": "refused", "steps": steps,
            "workflows_called": ["fixed_rag_pipeline"],
            "answer": "I cannot use the retrieved document because it contains an unsafe instruction-like passage.",
            "citations": [], "hits": hits, "stop_reason": "untrusted_document_instruction",
            "safety_findings": safety_findings,
            "metrics": {"step_count": len(steps), "tool_calls": tool_calls, "llm_calls": llm_calls, "elapsed_ms": elapsed_ms},
        }

    # Step 4: Refusal gate check
    refuse, gate, reason = refusal_check(norm_query, hits)
    if refuse:
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        answer_text = (
            "I cannot answer this from the indexed policy documents.\n"
            f"Reason: {reason}."
        )
        steps.append(
            _agent_step_record(
                step=3,
                decision="Refusal gate fired; short-circuit generation",
                workflow="refusal_gate",
                input_str=norm_query,
                result_status="refused",
                evidence_summary=f"Gate: {gate} — {reason}",
                next_decision="Stop and return refusal response",
                stop_reason="refusal_gate_fired",
            )
        )
        return {
            "mode": "fixed_workflow",
            "status": "refused",
            "steps": steps,
            "workflows_called": ["fixed_rag_pipeline"],
            "answer": answer_text,
            "citations": [],
            "hits": hits,
            "stop_reason": "refusal_gate_fired",
            "metrics": {
                "step_count": len(steps),
                "tool_calls": tool_calls,
                "llm_calls": llm_calls,
                "elapsed_ms": elapsed_ms,
            },
        }

    # Step 4: Generation via LLM
    llm_calls += 1
    llm = ChatGroq(model=GENERATION_MODEL, temperature=0.0, max_retries=2)
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context chunks:\n{_context(hits)}\n\nQuestion: {norm_query}"),
        ]
    )
    raw_text = response_text(response.content)
    citations = _citations(raw_text, hits)
    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)

    steps.append(
        _agent_step_record(
            step=4,
            decision="Generate cited answer from verified evidence context",
            workflow="llm_generation",
            input_str=f"{len(hits)} context chunks",
            result_status="success",
            evidence_summary=f"Generated answer with {len(citations)} citation(s)",
            next_decision="Complete fixed pipeline",
            stop_reason="generation_complete",
        )
    )

    return {
        "mode": "fixed_workflow",
        "status": "success",
        "steps": steps,
        "workflows_called": ["fixed_rag_pipeline"],
        "answer": raw_text,
        "citations": citations,
        "hits": hits,
        "stop_reason": "generation_complete",
        "metrics": {
            "step_count": len(steps),
            "tool_calls": tool_calls,
            "llm_calls": llm_calls,
            "elapsed_ms": elapsed_ms,
        },
    }


# ----------------------------------------------------------------------
# 2. Standalone Agent Loop
# ----------------------------------------------------------------------
def _select_initial_workflow(query: str) -> tuple[str, str]:
    """Inspect query to select the most appropriate initial workflow."""
    lowered = query.lower()

    if any(term in lowered for term in ("show all employee", "reveal employee", "dump employee", "employee records")):
        return "security_refusal", "Detected unauthorized employee-data request"

    if len(identifiers_from_query(query)) >= 2:
        return "employee_comparison", "Detected multiple employee identifiers"

    if any(term in lowered for term in ("what policy applies", "which policy applies", "applicable policy", "policy for emp")):
        return "employee_policy_case", "Detected employee-specific policy applicability request"

    if identifier_from_query(query) or any(
        term in lowered for term in ("employee record", "leave balance", "carry forward", "compensatory leave")
    ):
        return "employee_case", "Detected employee-specific structured-data request"

    # Multi-policy comparison signals
    if any(k in lowered for k in ("compare", "difference", "versus", "vs", "between", "how do", "similarities")):
        return "policy_comparison", "Detected multi-policy comparison intent"

    # Specific employee applicability / eligibility signals
    if any(
        k in lowered
        for k in (
            "am i eligible",
            "am i entitled",
            "can i work",
            "can i take",
            "do i qualify",
            "eligibility requirement",
            "my eligibility",
        )
    ):
        return "policy_applicability", "Detected employee eligibility/applicability intent"

    return "policy_lookup", "Standard single-policy lookup selected"


def run_agent(
    query: str,
    *,
    max_steps: int = 5,
    timeout_seconds: int = 60,
    strategy: str = "structure",
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute the dynamic agent loop with safety ceilings and step logging.

    Ceilings:
      - max_steps: <= 5 steps
      - tool_calls: <= 3 retrieval calls
      - timeout: <= 60 seconds
    """
    start_time = time.perf_counter()
    steps: list[dict[str, Any]] = []
    workflows_called: list[str] = []
    tool_calls = 0
    llm_calls = 0
    current_hits: list[Hit] = []
    final_answer = ""
    citations: list[Citation] = []
    stop_reason = ""
    agent_status = "success"
    safety_findings: list[dict[str, str]] = []

    # Step 1: Initial decision
    current_workflow, initial_rationale = _select_initial_workflow(query)

    for step_num in range(1, max_steps + 1):
        elapsed = time.perf_counter() - start_time
        if elapsed >= timeout_seconds:
            agent_status = "timeout"
            stop_reason = "timeout_exceeded"
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision=f"Abort execution: elapsed time {elapsed:.2f}s exceeded limit of {timeout_seconds}s",
                    workflow="safety_monitor",
                    input_str=query,
                    result_status="timeout",
                    evidence_summary="",
                    next_decision="Terminate",
                    stop_reason=stop_reason,
                )
            )
            break

        if tool_calls >= 3 and current_workflow in ("policy_lookup", "policy_comparison"):
            agent_status = "error"
            stop_reason = "tool_call_limit_reached"
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision="Safety limit triggered: maximum 3 tool calls reached",
                    workflow="safety_monitor",
                    input_str=query,
                    result_status="error",
                    evidence_summary="",
                    next_decision="Terminate",
                    stop_reason=stop_reason,
                )
            )
            break

        # Execute chosen workflow
        workflows_called.append(current_workflow)

        if current_workflow == "security_refusal":
            agent_status = "refused"
            stop_reason = "unauthorized_data_request"
            final_answer = "I cannot reveal employee records. Provide an authorized employee-specific question instead."
            steps.append(_agent_step_record(step=step_num, decision=initial_rationale, workflow="security_refusal", input_str=query, result_status="refused", evidence_summary="Employee data export is not an allowed action", next_decision="Terminate", stop_reason=stop_reason))
            break

        if current_workflow == "employee_comparison":
            tool_calls += 1
            res = employee_comparison(query, strategy=strategy, top_k=top_k)
            current_hits = res.get("hits", [])
            steps.append(_agent_step_record(step=step_num, decision=initial_rationale, workflow="employee_comparison", input_str=query, result_status=res["status"], evidence_summary=res.get("reason") or f"Compared {len(res.get('records', []))} records", next_decision="Return comparison" if res["status"] == "success" else "Request two records", stop_reason="employee_comparison_complete" if res["status"] == "success" else "employee_comparison_missing"))
            if res["status"] == "success":
                final_answer = res["answer"]
                stop_reason = "employee_comparison_complete"
                agent_status = "success"
            else:
                final_answer = "I need two valid employee IDs, customer IDs, or emails to compare."
                stop_reason = "employee_comparison_missing"
                agent_status = "needs_input"
            break

        if current_workflow == "employee_policy_case":
            tool_calls += 1
            res = employee_policy_case(query, strategy=strategy, top_k=top_k)
            current_hits = res.get("hits", [])
            steps.append(_agent_step_record(step=step_num, decision=initial_rationale, workflow="employee_policy_case", input_str=query, result_status=res["status"], evidence_summary=res.get("reason") or f"Loaded {res.get('record', {}).get('policy_id', 'policy')}", next_decision="Return combined result" if res["status"] == "success" else "Request employee identifier", stop_reason="employee_policy_complete" if res["status"] == "success" else "employee_data_missing"))
            if res["status"] == "success":
                final_answer = res["answer"]
                stop_reason = "employee_policy_complete"
                agent_status = "success"
            else:
                final_answer = "I need a valid employee identifier before checking the applicable policy."
                stop_reason = "employee_data_missing"
                agent_status = "needs_input"
            break

        if current_workflow == "employee_case":
            tool_calls += 1
            res = employee_case(query)
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision=initial_rationale if step_num == 1 else "Resolve employee record and calculate requested values",
                    workflow="employee_case",
                    input_str=query,
                    result_status=res["status"],
                    evidence_summary=(res.get("reason") or ", ".join(res.get("calculation", {}).get("requested", []))),
                    next_decision="Return structured result" if res["status"] == "success" else "Request employee identifier or record",
                    stop_reason="employee_case_complete" if res["status"] == "success" else "employee_data_missing",
                )
            )
            if res["status"] == "success":
                agent_status = "success"
                stop_reason = "employee_case_complete"
                final_answer = res["answer"]
            else:
                agent_status = "needs_input"
                stop_reason = "employee_data_missing"
                final_answer = (
                    "I need additional employee information before I can calculate this.\n- "
                    + "\n- ".join(res.get("missing_information", []))
                )
            break

        if current_workflow == "policy_comparison":
            tool_calls += 1
            res = policy_comparison(query, strategy=strategy, top_k=top_k)
            current_hits = res.get("hits", [])
            safety_findings = unsafe_hits(current_hits)
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision=initial_rationale if step_num == 1 else "Execute multi-policy comparison",
                    workflow="policy_comparison",
                    input_str=query,
                    result_status=res["status"],
                    evidence_summary=f"{len(current_hits)} comparison chunks aligned",
                    next_decision="Inspect evidence validation" if res["status"] == "success" else "Refuse",
                    stop_reason="" if res["status"] == "success" else "refused_in_comparison",
                )
            )
            if safety_findings:
                agent_status = "refused"
                stop_reason = "untrusted_document_instruction"
                final_answer = "I cannot use the retrieved document because it contains an unsafe instruction-like passage."
                break
            if res["status"] != "success":
                agent_status = "refused"
                stop_reason = "refused_in_comparison"
                final_answer = (
                    "I cannot answer this from the indexed policy documents.\n"
                    f"Reason: {res.get('reason', 'Comparative evidence insufficient')}."
                )
                break
            # Success -> Proceed to answer generation & audit
            current_workflow = "generate_and_audit"

        elif current_workflow == "policy_applicability":
            tool_calls += 1
            res = policy_applicability(query, strategy=strategy, top_k=top_k)
            current_hits = res.get("hits", [])
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision=initial_rationale if step_num == 1 else "Evaluate employee applicability",
                    workflow="policy_applicability",
                    input_str=query,
                    result_status=res["status"],
                    evidence_summary=(
                        f"Missing: {res.get('missing_information', [])}"
                        if res["status"] == "needs_input"
                        else f"{len(current_hits)} applicable rule chunks found"
                    ),
                    next_decision="Stop and request input" if res["status"] == "needs_input" else "Generate answer",
                    stop_reason="needs_employee_information" if res["status"] == "needs_input" else "",
                )
            )
            if res["status"] == "needs_input":
                agent_status = "needs_input"
                stop_reason = "needs_employee_information"
                final_answer = (
                    "Additional employee details are required to answer this question accurately:\n- "
                    + "\n- ".join(res.get("missing_information", []))
                    + "\n\nPlease specify these details so the applicable policy rules can be identified."
                )
                break
            if res["status"] != "success":
                agent_status = "refused"
                stop_reason = "applicability_refusal"
                final_answer = (
                    "I cannot answer this from the indexed policy documents.\n"
                    f"Reason: {res.get('reason', 'Policy rules not applicable')}."
                )
                break
            current_workflow = "generate_and_audit"

        elif current_workflow == "policy_lookup":
            tool_calls += 1
            res = policy_lookup(query, strategy=strategy, top_k=top_k)
            current_hits = res.get("hits", [])
            safety_findings = unsafe_hits(current_hits)
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision=initial_rationale if step_num == 1 else "Execute policy lookup",
                    workflow="policy_lookup",
                    input_str=query,
                    result_status=res["status"],
                    evidence_summary=f"{len(current_hits)} chunks retrieved; evidence_ok={res.get('evidence_ok')}",
                    next_decision="Generate answer and audit" if res["status"] == "success" else "Refuse question",
                    stop_reason="" if res["status"] == "success" else "refusal_gate_fired",
                )
            )
            if safety_findings:
                agent_status = "refused"
                stop_reason = "untrusted_document_instruction"
                final_answer = "I cannot use the retrieved document because it contains an unsafe instruction-like passage."
                break
            if res["status"] != "success":
                agent_status = "refused"
                stop_reason = "refusal_gate_fired"
                final_answer = (
                    "I cannot answer this from the indexed policy documents.\n"
                    f"Reason: {res.get('reason', 'Refusal gate fired')}."
                )
                break
            current_workflow = "generate_and_audit"

        elif current_workflow == "generate_and_audit":
            # Generate answer using LLM
            llm_calls += 1
            llm = ChatGroq(model=GENERATION_MODEL, temperature=0.0, max_retries=2)
            response = llm.invoke(
                [
                    ("system", SYSTEM_PROMPT),
                    ("human", f"Context chunks:\n{_context(current_hits)}\n\nQuestion: {query}"),
                ]
            )
            generated_text = response_text(response.content)
            parsed_citations = _citations(generated_text, current_hits)

            # Audit step
            audit_res = policy_audit(
                query,
                answer_text=generated_text,
                citations=parsed_citations,
                hits=current_hits,
            )
            steps.append(
                _agent_step_record(
                    step=step_num,
                    decision="Generate grounded answer and audit citations/claims",
                    workflow="policy_audit",
                    input_str=f"{len(current_hits)} chunks",
                    result_status=audit_res["status"],
                    evidence_summary=(
                        f"Answer generated ({len(parsed_citations)} citations); audit status={audit_res['status']}"
                    ),
                    next_decision="Terminate with approved answer",
                    stop_reason="answer_generated_and_audited",
                )
            )
            final_answer = generated_text
            citations = parsed_citations
            stop_reason = "answer_generated_and_audited"
            agent_status = "success"
            break

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    metrics = {
        "step_count": len(steps),
        "tool_calls": tool_calls,
        "llm_calls": llm_calls,
        "elapsed_ms": elapsed_ms,
    }
    write_trajectory(
        query=query, steps=steps, status=agent_status,
        workflows_called=workflows_called, answer=final_answer,
        stop_reason=stop_reason, metrics=metrics,
        safety_findings=safety_findings,
    )
    return {
        "mode": "agent",
        "status": agent_status,
        "steps": steps,
        "workflows_called": workflows_called,
        "answer": final_answer,
        "citations": citations,
        "hits": current_hits,
        "stop_reason": stop_reason,
        "metrics": metrics,
        "safety_findings": safety_findings,
    }
