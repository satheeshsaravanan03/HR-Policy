"""Unit-style validation script for the five predefined workflows (Week 7)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.workflows import (
    policy_lookup,
    policy_comparison,
    evidence_validation,
    policy_applicability,
    policy_audit,
)
from rag.generate import Citation


def test_workflows() -> None:
    print("Testing 5 predefined workflows...")

    # 1. Policy Lookup (answerable)
    res1 = policy_lookup("How many annual leave days does an Acme full-time employee receive per calendar year?")
    assert res1["workflow"] == "policy_lookup"
    assert res1["status"] == "success"
    assert res1["evidence_ok"] is True
    assert len(res1["hits"]) > 0
    assert len(res1["steps"]) >= 4
    print("  [PASS] policy_lookup (answerable)")

    # 2. Policy Lookup (refusal)
    res2 = policy_lookup("What is SoftSuave's sabbatical leave entitlement?")
    assert res2["workflow"] == "policy_lookup"
    assert res2["status"] == "refused"
    assert res2["evidence_ok"] is False
    assert res2["reason"] != ""
    print("  [PASS] policy_lookup (refusal)")

    # 3. Policy Comparison
    res3 = policy_comparison("Compare annual leave between Acme and SoftSuave employees.")
    assert res3["workflow"] == "policy_comparison"
    assert res3["status"] == "success"
    assert res3["evidence_ok"] is True
    assert len(res3["hits"]) > 0
    print("  [PASS] policy_comparison")

    # 4. Evidence Validation
    res4 = evidence_validation("How many days per week may an eligible Northstar employee work remotely?")
    assert res4["workflow"] == "evidence_validation"
    assert res4["status"] == "success"
    assert res4["evidence_ok"] is True
    print("  [PASS] evidence_validation")

    # 5. Policy Applicability (missing input)
    res5 = policy_applicability("Am I eligible to work from home full-time?")
    assert res5["workflow"] == "policy_applicability"
    assert res5["status"] == "needs_input"
    assert res5["evidence_ok"] is False
    assert len(res5["missing_information"]) > 0
    print("  [PASS] policy_applicability (needs_input)")

    # 6. Policy Applicability (with input / facts in query)
    res6 = policy_applicability(
        "Can a permanent employee in the Bangalore office take casual leave?",
    )
    assert res6["workflow"] == "policy_applicability"
    assert res6["status"] == "success"
    assert res6["evidence_ok"] is True
    print("  [PASS] policy_applicability (with facts)")

    # 7. Policy Audit (valid citation)
    hit = res1["hits"][0]
    valid_citation = Citation(
        chunk_id=hit.chunk_id,
        policy_id=hit.policy_id,
        section=hit.section,
        resolves=True,
    )
    res7 = policy_audit(
        "Acme leave",
        answer_text=f"Leave policy [CITE: {hit.chunk_id}]",
        citations=[valid_citation],
        hits=res1["hits"],
    )
    assert res7["workflow"] == "policy_audit"
    assert res7["status"] == "success"
    assert res7["evidence_ok"] is True
    print("  [PASS] policy_audit (valid citation)")

    # 8. Policy Audit (unresolved citation)
    bad_citation = Citation(
        chunk_id="non-existent-chunk-id",
        policy_id="ACME",
        section="1",
        resolves=False,
    )
    res8 = policy_audit(
        "Acme leave",
        citations=[bad_citation],
        hits=res1["hits"],
    )
    assert res8["workflow"] == "policy_audit"
    assert res8["status"] == "error"
    assert res8["evidence_ok"] is False
    print("  [PASS] policy_audit (unresolved citation caught)")

    print("\nALL 5 WORKFLOWS VALIDATED SUCCESSFULLY!")


if __name__ == "__main__":
    test_workflows()
