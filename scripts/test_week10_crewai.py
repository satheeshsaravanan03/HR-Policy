"""Offline tests for CrewAI's explicit specialist-output handoffs."""

from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.week10_a2a import _agent_work  # noqa: E402
from rag.week10_crewai import _SpecialistTool, _cost, _plan_with_crewai  # noqa: E402


class _FakeAgent:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeTask:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeProcess:
    sequential = "sequential"


class _FakeCrew:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def kickoff(self):
        return type("Output", (), {"raw": '{"specialists": ["policy_research"], "reason": "policy lookup"}'})()


class CrewAIHandoffTests(unittest.TestCase):
    def test_crewai_manager_plan_is_guarded_with_required_policy_review(self):
        roles, reason, _usage = _plan_with_crewai(
            "What does the ACME policy say about carry-forward?",
            _FakeAgent, _FakeCrew, _FakeProcess, _FakeTask, object(),
        )
        self.assertEqual(roles, ["policy_research", "evidence_review"])
        self.assertEqual(reason, "policy lookup")

    def test_crewai_manager_cannot_route_employee_activity_to_policy_search(self):
        roles, _reason, _usage = _plan_with_crewai(
            "List employees who used more than 2 days of leave",
            _FakeAgent, _FakeCrew, _FakeProcess, _FakeTask, object(),
        )
        self.assertEqual(roles, ["leave_activity", "record_validator"])

    def test_record_validator_receives_directory_specialist_output(self):
        directory_result = {
            "role": "employee_directory",
            "data": {"employee_id": "EMP-005", "policy_id": "ACME-EMP-2026"},
        }
        payload = _SpecialistTool.payload(
            "record_validator", "employee005@example.com", [directory_result]
        )
        self.assertEqual(payload["record"], directory_result["data"])

    def test_evidence_review_receives_previous_specialist_results(self):
        prior_results = [{"role": "policy_research", "data": {"results": [{"chunk_id": "chunk-1"}]}}]
        payload = _SpecialistTool.payload("evidence_review", "policy question", prior_results)
        self.assertIs(payload["specialist_results"], prior_results)

    def test_activity_records_are_passed_to_validator_and_checked(self):
        prior_results = [{
            "role": "leave_activity",
            "data": {"employees": [{"employee_id": "EMP-005", "current_leave_balance": 3}]},
        }]
        payload = _SpecialistTool.payload("record_validator", "list employees", prior_results)
        result = _agent_work("record_validator", payload)
        self.assertTrue(result["data"]["valid"])
        self.assertEqual(result["data"]["employee_ids"], ["EMP-005"])

    def test_cost_is_not_claimed_without_both_token_rates(self):
        self.assertIsNone(_cost({"input_tokens": 10, "output_tokens": 5}, {}))

    def test_cost_uses_reported_input_and_output_rates(self):
        result = _cost(
            {"input_tokens": 1_000_000, "output_tokens": 500_000},
            {"input_per_million": 2.0, "output_per_million": 4.0},
        )
        self.assertEqual(result, 4.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
