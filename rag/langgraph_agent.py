"""LangGraph implementation of the HR Policy Agent (Week 7).

Implements the agent workflow as a state graph using the `langgraph` framework.
The 5 predefined workflows in `rag/workflows.py` serve as the graph's nodes,
and decision logic serves as conditional edges.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from .generate import (
    GENERATION_MODEL,
    SYSTEM_PROMPT,
    Citation,
    _citations,
    _context,
    response_text,
)
from .retrieve import Hit
from .workflows import (
    policy_applicability,
    policy_audit,
    policy_comparison,
    policy_lookup,
)


class HRPolicyState(TypedDict):
    """The shared state passed between nodes in the LangGraph."""
    query: str
    strategy: str
    top_k: int
    status: str
    workflows_called: list[str]
    steps: list[dict[str, Any]]
    hits: list[Hit]
    citations: list[Citation]
    answer: str
    missing_information: list[str]
    stop_reason: str
    tool_calls: int
    llm_calls: int
    start_time: float
    elapsed_ms: float


def _step_record(
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
# 1. Routing functions (Conditional Edges)
# ----------------------------------------------------------------------
def route_intent(state: HRPolicyState) -> str:
    """Classify user intent and route to the appropriate retrieval node."""
    lowered = state["query"].lower()

    # Multi-policy comparison intent
    if any(k in lowered for k in ("compare", "difference", "versus", "vs", "between", "how do", "similarities")):
        return "comparison_node"

    # Eligibility & applicability intent
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
        return "applicability_node"

    return "lookup_node"


def route_after_retrieval(state: HRPolicyState) -> str:
    """Decide whether to terminate or proceed to generation and audit."""
    if state["status"] in ("needs_input", "refused", "error"):
        return END
    return "generate_and_audit_node"


# ----------------------------------------------------------------------
# 2. Graph Nodes
# ----------------------------------------------------------------------
def lookup_node(state: HRPolicyState) -> dict[str, Any]:
    """Execute single-policy lookup workflow."""
    res = policy_lookup(state["query"], strategy=state["strategy"], top_k=state["top_k"])
    hits = res.get("hits", [])
    status = res["status"]
    stop_reason = "" if status == "success" else "refusal_gate_fired"

    step = _step_record(
        step=len(state["steps"]) + 1,
        decision="Execute policy lookup workflow",
        workflow="policy_lookup",
        input_str=state["query"],
        result_status=status,
        evidence_summary=f"{len(hits)} chunks retrieved; evidence_ok={res.get('evidence_ok')}",
        next_decision="Generate answer and audit" if status == "success" else "Refuse question",
        stop_reason=stop_reason,
    )

    answer = ""
    if status != "success":
        answer = (
            "I cannot answer this from the indexed policy documents.\n"
            f"Reason: {res.get('reason', 'Refusal gate fired')}."
        )

    return {
        "status": status,
        "workflows_called": state["workflows_called"] + ["policy_lookup"],
        "steps": state["steps"] + [step],
        "hits": hits,
        "answer": answer,
        "stop_reason": stop_reason,
        "tool_calls": state["tool_calls"] + 1,
    }


def comparison_node(state: HRPolicyState) -> dict[str, Any]:
    """Execute multi-policy comparison workflow."""
    res = policy_comparison(state["query"], strategy=state["strategy"], top_k=state["top_k"])
    hits = res.get("hits", [])
    status = res["status"]
    stop_reason = "" if status == "success" else "refused_in_comparison"

    step = _step_record(
        step=len(state["steps"]) + 1,
        decision="Execute multi-policy comparison workflow",
        workflow="policy_comparison",
        input_str=state["query"],
        result_status=status,
        evidence_summary=f"{len(hits)} comparison chunks aligned",
        next_decision="Generate answer and audit" if status == "success" else "Refuse question",
        stop_reason=stop_reason,
    )

    answer = ""
    if status != "success":
        answer = (
            "I cannot answer this from the indexed policy documents.\n"
            f"Reason: {res.get('reason', 'Comparative evidence insufficient')}."
        )

    return {
        "status": status,
        "workflows_called": state["workflows_called"] + ["policy_comparison"],
        "steps": state["steps"] + [step],
        "hits": hits,
        "answer": answer,
        "stop_reason": stop_reason,
        "tool_calls": state["tool_calls"] + 1,
    }


def applicability_node(state: HRPolicyState) -> dict[str, Any]:
    """Execute employee applicability workflow."""
    res = policy_applicability(state["query"], strategy=state["strategy"], top_k=state["top_k"])
    hits = res.get("hits", [])
    status = res["status"]
    missing = res.get("missing_information", [])

    if status == "needs_input":
        stop_reason = "needs_employee_information"
        answer = (
            "Additional employee details are required to answer this question accurately:\n- "
            + "\n- ".join(missing)
            + "\n\nPlease specify these details so the applicable policy rules can be identified."
        )
    elif status != "success":
        stop_reason = "applicability_refusal"
        answer = (
            "I cannot answer this from the indexed policy documents.\n"
            f"Reason: {res.get('reason', 'Policy rules not applicable')}."
        )
    else:
        stop_reason = ""
        answer = ""

    step = _step_record(
        step=len(state["steps"]) + 1,
        decision="Evaluate employee applicability workflow",
        workflow="policy_applicability",
        input_str=state["query"],
        result_status=status,
        evidence_summary=f"Missing: {missing}" if status == "needs_input" else f"{len(hits)} rule chunks found",
        next_decision="Stop and request input" if status == "needs_input" else "Generate answer and audit",
        stop_reason=stop_reason,
    )

    return {
        "status": status,
        "workflows_called": state["workflows_called"] + ["policy_applicability"],
        "steps": state["steps"] + [step],
        "hits": hits,
        "missing_information": missing,
        "answer": answer,
        "stop_reason": stop_reason,
        "tool_calls": state["tool_calls"] + 1,
    }


def generate_and_audit_node(state: HRPolicyState) -> dict[str, Any]:
    """Generate grounded answer using LLM and verify citations via audit."""
    hits = state["hits"]
    query = state["query"]

    # 1. LLM Generation
    llm = ChatGroq(model=GENERATION_MODEL, temperature=0.0, max_retries=2)
    response = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", f"Context chunks:\n{_context(hits)}\n\nQuestion: {query}"),
        ]
    )
    generated_text = response_text(response.content)
    parsed_citations = _citations(generated_text, hits)

    # 2. Audit
    audit_res = policy_audit(
        query,
        answer_text=generated_text,
        citations=parsed_citations,
        hits=hits,
    )

    step = _step_record(
        step=len(state["steps"]) + 1,
        decision="Generate grounded answer and audit citations/claims",
        workflow="policy_audit",
        input_str=f"{len(hits)} context chunks",
        result_status=audit_res["status"],
        evidence_summary=f"Answer generated ({len(parsed_citations)} citations); audit status={audit_res['status']}",
        next_decision="Terminate with approved answer",
        stop_reason="answer_generated_and_audited",
    )

    return {
        "status": "success",
        "workflows_called": state["workflows_called"] + ["policy_audit"],
        "steps": state["steps"] + [step],
        "answer": generated_text,
        "citations": parsed_citations,
        "stop_reason": "answer_generated_and_audited",
        "llm_calls": state["llm_calls"] + 1,
    }


# ----------------------------------------------------------------------
# 3. Construct and Compile Graph
# ----------------------------------------------------------------------
def create_hr_policy_graph():
    """Build and compile the LangGraph agent."""
    builder = StateGraph(HRPolicyState)

    # Add Nodes
    builder.add_node("lookup_node", lookup_node)
    builder.add_node("comparison_node", comparison_node)
    builder.add_node("applicability_node", applicability_node)
    builder.add_node("generate_and_audit_node", generate_and_audit_node)

    # Conditional Entry Point (Intent Router)
    builder.set_conditional_entry_point(
        route_intent,
        {
            "lookup_node": "lookup_node",
            "comparison_node": "comparison_node",
            "applicability_node": "applicability_node",
        },
    )

    # Conditional Transitions after retrieval
    for node in ("lookup_node", "comparison_node", "applicability_node"):
        builder.add_conditional_edges(
            node,
            route_after_retrieval,
            {
                END: END,
                "generate_and_audit_node": "generate_and_audit_node",
            },
        )

    builder.add_edge("generate_and_audit_node", END)
    return builder.compile()


_COMPILED_GRAPH = None


def get_graph():
    """Lazy initialization of compiled graph."""
    global _COMPILED_GRAPH
    if _COMPILED_GRAPH is None:
        _COMPILED_GRAPH = create_hr_policy_graph()
    return _COMPILED_GRAPH


# ----------------------------------------------------------------------
# 4. Public Execution Function
# ----------------------------------------------------------------------
def run_langgraph_agent(
    query: str,
    *,
    strategy: str = "structure",
    top_k: int = 5,
) -> dict[str, Any]:
    """Execute the HR Policy Agent via LangGraph.

    Returns the standard result shape matching `run_agent`.
    """
    start_time = time.perf_counter()
    initial_state: HRPolicyState = {
        "query": query,
        "strategy": strategy,
        "top_k": top_k,
        "status": "pending",
        "workflows_called": [],
        "steps": [],
        "hits": [],
        "citations": [],
        "answer": "",
        "missing_information": [],
        "stop_reason": "",
        "tool_calls": 0,
        "llm_calls": 0,
        "start_time": start_time,
        "elapsed_ms": 0.0,
    }

    graph = get_graph()
    final_state = graph.invoke(initial_state)

    elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
    return {
        "mode": "langgraph_agent",
        "status": final_state["status"],
        "steps": final_state["steps"],
        "workflows_called": final_state["workflows_called"],
        "answer": final_state["answer"],
        "citations": final_state["citations"],
        "hits": final_state["hits"],
        "stop_reason": final_state["stop_reason"],
        "missing_information": final_state.get("missing_information", []),
        "metrics": {
            "step_count": len(final_state["steps"]),
            "tool_calls": final_state["tool_calls"],
            "llm_calls": final_state["llm_calls"],
            "elapsed_ms": elapsed_ms,
        },
    }
