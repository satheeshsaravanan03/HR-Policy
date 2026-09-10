"""Validation test script for rag/agent.py (Week 7)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.agent import run_agent, run_fixed_workflow


def test_agent_and_workflow() -> None:
    print("Testing rag/agent.py execution paths...")

    # 1. Fixed workflow on answerable query
    print("  Testing run_fixed_workflow (answerable)...")
    res_fixed_ans = run_fixed_workflow(
        "How many annual leave days does an Acme full-time employee receive per calendar year?"
    )
    assert res_fixed_ans["mode"] == "fixed_workflow"
    assert res_fixed_ans["status"] == "success"
    assert res_fixed_ans["metrics"]["tool_calls"] == 1
    assert res_fixed_ans["metrics"]["llm_calls"] == 1
    assert len(res_fixed_ans["answer"]) > 0
    print("  [PASS] run_fixed_workflow (answerable)")

    # 2. Fixed workflow on refusal query
    print("  Testing run_fixed_workflow (refusal)...")
    res_fixed_ref = run_fixed_workflow("What is SoftSuave's sabbatical leave entitlement?")
    assert res_fixed_ref["mode"] == "fixed_workflow"
    assert res_fixed_ref["status"] == "refused"
    assert res_fixed_ref["metrics"]["llm_calls"] == 0
    assert "cannot answer" in res_fixed_ref["answer"].lower()
    print("  [PASS] run_fixed_workflow (refusal)")

    # 3. Standalone agent on single-policy query
    print("  Testing run_agent (single-policy)...")
    res_agent_single = run_agent(
        "How many annual leave days does an Acme full-time employee receive per calendar year?"
    )
    assert res_agent_single["mode"] == "agent"
    assert res_agent_single["status"] == "success"
    assert "policy_lookup" in res_agent_single["workflows_called"]
    assert res_agent_single["metrics"]["step_count"] <= 5
    assert res_agent_single["metrics"]["tool_calls"] <= 3
    print("  [PASS] run_agent (single-policy)")

    # 4. Standalone agent on multi-policy comparison
    print("  Testing run_agent (multi-policy comparison)...")
    res_agent_comp = run_agent("Compare annual leave between Acme and SoftSuave employees.")
    assert res_agent_comp["mode"] == "agent"
    assert res_agent_comp["status"] == "success"
    assert "policy_comparison" in res_agent_comp["workflows_called"]
    print("  [PASS] run_agent (multi-policy comparison)")

    # 5. Standalone agent on eligibility query with missing facts
    print("  Testing run_agent (eligibility missing facts)...")
    res_agent_elig = run_agent("Am I eligible to work from home full-time?")
    assert res_agent_elig["mode"] == "agent"
    assert res_agent_elig["status"] == "needs_input"
    assert res_agent_elig["stop_reason"] == "needs_employee_information"
    assert "policy_applicability" in res_agent_elig["workflows_called"]
    print("  [PASS] run_agent (eligibility missing facts -> needs_input)")

    # 6. Standalone agent on refusal query
    print("  Testing run_agent (refusal)...")
    res_agent_ref = run_agent("What is SoftSuave's sabbatical leave entitlement?")
    assert res_agent_ref["mode"] == "agent"
    assert res_agent_ref["status"] == "refused"
    assert res_agent_ref["metrics"]["llm_calls"] == 0
    print("  [PASS] run_agent (refusal)")

    print("\nALL AGENT AND FIXED WORKFLOW TESTS PASSED!")


if __name__ == "__main__":
    test_agent_and_workflow()
