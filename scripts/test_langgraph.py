"""Unit test script for rag/langgraph_agent.py."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.langgraph_agent import run_langgraph_agent, get_graph


def test_langgraph() -> None:
    print("Testing LangGraph implementation (rag/langgraph_agent.py)...")

    # Verify graph compiles and exports structure
    graph = get_graph()
    assert graph is not None
    print("  [PASS] Graph compiled successfully")

    # 1. Single-policy lookup
    print("  Testing LangGraph: single-policy lookup...")
    res1 = run_langgraph_agent(
        "How many annual leave days does an Acme full-time employee receive per calendar year?"
    )
    assert res1["mode"] == "langgraph_agent"
    assert res1["status"] == "success"
    assert "policy_lookup" in res1["workflows_called"]
    assert "policy_audit" in res1["workflows_called"]
    assert len(res1["answer"]) > 0
    print("  [PASS] LangGraph single-policy lookup")

    # 2. Multi-policy comparison
    print("  Testing LangGraph: multi-policy comparison...")
    res2 = run_langgraph_agent(
        "Compare annual leave between Acme and SoftSuave employees."
    )
    assert res2["mode"] == "langgraph_agent"
    assert res2["status"] == "success"
    assert "policy_comparison" in res2["workflows_called"]
    print("  [PASS] LangGraph multi-policy comparison")

    # 3. Missing facts eligibility
    print("  Testing LangGraph: missing facts -> needs_input...")
    res3 = run_langgraph_agent("Am I eligible to work from home full-time?")
    assert res3["mode"] == "langgraph_agent"
    assert res3["status"] == "needs_input"
    assert res3["stop_reason"] == "needs_employee_information"
    assert len(res3["missing_information"]) > 0
    print("  [PASS] LangGraph missing facts handling")

    # 4. Refusal query
    print("  Testing LangGraph: refusal query...")
    res4 = run_langgraph_agent("What is SoftSuave's sabbatical leave entitlement?")
    assert res4["mode"] == "langgraph_agent"
    assert res4["status"] == "refused"
    print("  [PASS] LangGraph refusal query")

    print("\nALL LANGGRAPH TESTS PASSED!")


if __name__ == "__main__":
    test_langgraph()
