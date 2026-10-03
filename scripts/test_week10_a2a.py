"""Offline unit tests for the Week 10 A2A team and evaluation helpers."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
import sys
import threading
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag import employee_data, mcp_agent, week10_a2a as week10  # noqa: E402


class Week10PlanningTests(unittest.TestCase):
    def test_standalone_hello_does_not_select_policy_specialist(self):
        self.assertTrue(week10.is_greeting_query("Hello."))
        self.assertTrue(week10.is_greeting_query("Thank you!"))
        self.assertFalse(week10.is_greeting_query("Hello, what is ACME's carry-forward limit?"))

    def test_mcp_baseline_answers_hello_without_calling_tools(self):
        with patch("rag.mcp_agent.stdio_client", side_effect=AssertionError("MCP should not be called")):
            result = asyncio.run(mcp_agent._run_mcp_policy_agent("Hello."))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["stop_reason"], "greeting_handled_without_retrieval")
        self.assertEqual(result["tool_calls"], 0)
        self.assertFalse(result["hits"])

    def test_week10_team_answers_hello_without_starting_a2a_or_llm(self):
        with (
            patch.object(week10, "ensure_a2a_server", side_effect=AssertionError("A2A should not start")),
            patch.object(week10, "_append_task_trace"),
        ):
            result = week10.run_week10_team("Hello.")
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["llm_calls"], 0)
        self.assertEqual(result["selected_specialists"], [])
        self.assertIn("Hello!", result["answer"])

    def test_employee_balance_fallback_includes_record_and_calculation_stages(self):
        roles = week10._fallback_plan("How much leave balance does EMP-002 have?")
        self.assertIn("employee_directory", roles)
        self.assertIn("record_validator", roles)
        self.assertIn("policy_calculator", roles)
        self.assertIn("evidence_review", roles)

    def test_transaction_search_uses_activity_specialist(self):
        roles = week10._fallback_plan("List employees who used more than 2 days of leave")
        self.assertIn("leave_activity", roles)
        self.assertNotIn("policy_calculator", roles)

    def test_threshold_first_balance_query_routes_to_employee_activity(self):
        query = "list out the employee details who has more then 3 leave balance"
        self.assertEqual(mcp_agent._transaction_search_arguments(query), {
            "field": "current_leave_balance", "operator": "gt", "threshold": 3.0,
        })
        self.assertTrue(mcp_agent._looks_like_transaction_search(query))
        roles = week10._fallback_plan(query)
        self.assertIn("leave_activity", roles)
        self.assertNotIn("policy_research", roles)

    def test_experience_filter_returns_only_employees_over_two_years(self):
        query = "List out the employees who have more 2 years of experience"
        criteria = mcp_agent._transaction_search_arguments(query)
        self.assertEqual(criteria, {"field": "experience_years", "operator": "gt", "threshold": 2.0})
        self.assertTrue(mcp_agent._looks_like_transaction_search(query))
        matches = employee_data.search_leave_transactions(**criteria)
        self.assertEqual([row["employee_id"] for row in matches], ["EMP-002", "EMP-004"])
        roles = week10._fallback_plan(query)
        self.assertEqual(roles, ["leave_activity"])

    def test_planner_api_failure_uses_deterministic_employee_route(self):
        query = "list out the employee details who has more then 3 leave balance"
        with patch.object(week10, "_llm_json", side_effect=RuntimeError("offline")):
            roles, reason, usage = week10._plan(query)
        self.assertIn("leave_activity", roles)
        self.assertNotIn("policy_research", roles)
        self.assertIn("deterministic routing", reason)
        self.assertEqual(usage["llm_calls"], 0)

    def test_manager_plan_is_guarded_with_required_roles(self):
        with patch.object(week10, "_llm_json", return_value=({"specialists": ["policy_research"], "reason": "policy"}, {"input_tokens": 3, "output_tokens": 2})):
            roles, reason, usage = week10._plan("How much leave balance does EMP-002 have?")
        self.assertIn("employee_directory", roles)
        self.assertIn("policy_calculator", roles)
        self.assertIn("evidence_review", roles)
        self.assertEqual(reason, "policy")
        self.assertEqual(usage["input_tokens"], 3)


class Week10SpecialistTests(unittest.TestCase):
    def test_leave_activity_uses_find_then_details_tools_for_balance_filter(self):
        query = "list out the employee details who has more then 3 leave balance"
        matches = {"employee_ids": ["EMP-004"], "criteria": {"field": "current_leave_balance", "operator": "gt", "threshold": 3.0}}
        employee_rows = {"status": "success", "employees": [{
            "employee_id": "EMP-004", "name": "Demo Employee 004", "email": "employee004@example.com",
            "region": "US", "policy_id": "ACME-HR-2026", "experience_years": 4,
            "current_leave_balance": 8, "transaction": {"leave_used_days": 2},
        }]}
        with patch.object(week10, "_call_mcp", side_effect=[matches, employee_rows]) as mcp_call:
            result = week10._agent_work("leave_activity", {"query": query})
        self.assertEqual(mcp_call.call_args_list[0].args[0], "find_employee_ids_by_employee_criteria")
        self.assertEqual(mcp_call.call_args_list[0].args[1], matches["criteria"])
        self.assertEqual(mcp_call.call_args_list[1].args[0], "get_employee_details")
        self.assertEqual(mcp_call.call_args_list[1].args[1], {"employee_ids": ["EMP-004"]})
        self.assertEqual(result["data"]["employees"], employee_rows["employees"])

    def test_leave_activity_uses_employee_criteria_tool_for_experience_filter(self):
        query = "List out the employees who have more 2 years of experience"
        matches = {"employee_ids": ["EMP-002", "EMP-004"], "criteria": {"field": "experience_years", "operator": "gt", "threshold": 2.0}}
        employee_rows = {"status": "success", "employees": [{"employee_id": "EMP-002"}, {"employee_id": "EMP-004"}]}
        with patch.object(week10, "_call_mcp", side_effect=[matches, employee_rows]) as mcp_call:
            result = week10._agent_work("leave_activity", {"query": query})
        self.assertEqual(mcp_call.call_args_list[0].args[0], "find_employee_ids_by_employee_criteria")
        self.assertEqual(mcp_call.call_args_list[0].args[1], {"field": "experience_years", "operator": "gt", "threshold": 2.0})
        self.assertEqual(mcp_call.call_args_list[1].args[0], "get_employee_details")
        self.assertEqual(mcp_call.call_args_list[1].args[1], {"employee_ids": ["EMP-002", "EMP-004"]})
        self.assertEqual(result["tool_calls"], ["find_employee_ids_by_employee_criteria", "get_employee_details"])

    def test_final_answer_generation_accepts_plain_text_not_json(self):
        class FakeResponse:
            content = "ACME employees receive 18 annual leave days."
            usage_metadata = {"input_tokens": 11, "output_tokens": 7}

        class FakeLLM:
            def invoke(self, _messages):
                return FakeResponse()

        with (
            patch.object(week10, "ChatGroq", return_value=FakeLLM()),
            patch.object(week10, "response_text", side_effect=lambda content: content),
        ):
            answer, usage = week10._llm_text("Answer using evidence")
        self.assertEqual(answer, "ACME employees receive 18 annual leave days.")
        self.assertEqual(usage, {"input_tokens": 11, "output_tokens": 7})

    def test_manager_gets_real_citation_attached_when_it_omits_chunk_id(self):
        query = "What is the carry-forward limit in the ACME leave policy?"
        specialists = [{
            "role": "policy_research",
            "data": {"results": [
                {"rank": 1, "policy_id": "ACME-LEAVE-2026", "section": "3", "chunk_id": "acme-section-3", "text": "Employees may carry forward up to 10 unused annual-leave days."},
                {"rank": 2, "policy_id": "AZURE-HR-2026", "section": "4", "chunk_id": "azure-section-4", "text": "Unrelated carry-forward rule."},
            ]},
        }]
        answer, citation_ok = week10._ensure_resolvable_policy_citation(
            "Employees may carry forward up to 10 unused days.", query, specialists,
        )
        self.assertTrue(citation_ok)
        self.assertIn("ACME-LEAVE-2026 section 3, chunk_id acme-section-3", answer)
        self.assertNotIn("azure-section-4", answer)

    def test_manager_refuses_if_no_matching_policy_citation_exists(self):
        answer, citation_ok = week10._ensure_resolvable_policy_citation(
            "Some answer.",
            "What is the carry-forward limit in the ACME leave policy?",
            [{"role": "policy_research", "data": {"results": [
                {"policy_id": "AZURE-HR-2026", "section": "4", "chunk_id": "azure-section-4"},
            ]}}],
        )
        self.assertFalse(citation_ok)
        self.assertEqual(answer, "Some answer.")

    def test_acme_name_resolves_to_explicit_policy_filter(self):
        self.assertEqual(
            week10.understand_query("What is the carry-forward limit in the ACME leave policy?"),
            ("ACME-LEAVE-2026", None),
        )

    def test_policy_research_passes_explicit_acme_filter_to_mcp(self):
        expected = {"results": []}
        with patch.object(week10, "_call_mcp", return_value=expected) as mcp_call:
            result = week10._agent_work("policy_research", {
                "query": "What is the carry-forward limit in the ACME leave policy?",
            })
        mcp_call.assert_called_once_with("search_hr_policy", {
            "query": "What is the carry-forward limit in the ACME leave policy?",
            "strategy": "structure", "method": "hybrid", "top_k": 5,
            "policy_id": "ACME-LEAVE-2026",
        })
        self.assertEqual(result["data"], expected)

    def test_evidence_reviewer_ignores_unrelated_uncitable_hits(self):
        results = [{
            "role": "policy_research",
            "data": {"results": [
                {"policy_id": "ACME-LEAVE-2026", "section": "3", "chunk_id": "acme-s3", "text": "Carry forward up to 10 days."},
                {"policy_id": "AZURE-HR-2026", "section": None, "chunk_id": "azure-no-section", "text": "Unrelated policy result."},
                {"policy_id": None, "section": "2", "chunk_id": None, "text": "Incomplete unrelated hit."},
            ]},
        }]
        with patch.object(week10, "_llm_json", return_value=({"approved": True, "issues": []}, {"input_tokens": 1, "output_tokens": 1})) as judge:
            result = week10._agent_work("evidence_review", {
                "query": "What is the carry-forward limit in the ACME leave policy?",
                "specialist_results": results,
            })
        self.assertTrue(result["data"]["approved"])
        self.assertEqual(result["data"]["citation_count"], 1)
        self.assertIn('"policy_id": "ACME-LEAVE-2026"', judge.call_args.args[0])
        self.assertNotIn("AZURE-HR-2026", judge.call_args.args[0])

    def test_evidence_reviewer_rejects_when_requested_policy_has_no_citable_hit(self):
        results = [{"role": "policy_research", "data": {"results": [
            {"policy_id": "AZURE-HR-2026", "section": "3", "chunk_id": "azure-s3", "text": "Other policy."},
        ]}}]
        with patch.object(week10, "_llm_json", return_value=({"approved": True, "issues": []}, {"input_tokens": 1, "output_tokens": 1})):
            result = week10._agent_work("evidence_review", {
                "query": "What is the carry-forward limit in the ACME leave policy?",
                "specialist_results": results,
            })
        self.assertFalse(result["data"]["approved"])
        self.assertTrue(any("ACME-LEAVE-2026" in issue for issue in result["data"]["issues"]))

    def test_employee_directory_calls_mcp_with_normalized_email(self):
        expected = {"employee_id": "EMP-005", "policy_id": "ACME-EMP-2026"}
        with patch.object(week10, "_call_mcp", return_value=expected) as mcp_call:
            result = week10._agent_work("employee_directory", {"query": "employee005@example.com details"})
        mcp_call.assert_called_once_with("get_employee_record", {"identifier": "employee005@example.com"})
        self.assertEqual(result["data"], expected)

    def test_record_validator_rejects_negative_balance_and_missing_policy(self):
        result = week10._agent_work("record_validator", {
            "query": "validate", "record": {"employee_id": "EMP-002", "current_leave_balance": -1},
        })
        self.assertFalse(result["data"]["valid"])
        self.assertIn("Policy ID is missing", result["data"]["issues"])
        self.assertIn("Current leave balance must be a non-negative number", result["data"]["issues"])

    def test_evidence_reviewer_does_not_override_structural_policy_mismatch(self):
        results = [
            {"role": "employee_directory", "data": {"employee_id": "EMP-002", "policy_id": "AZURE-HR-2026"}},
            {"role": "policy_research", "data": {"results": [{"policy_id": "OTHER-POLICY", "section": "2", "chunk_id": "chunk-x", "text": "rule"}]}},
        ]
        with patch.object(week10, "_llm_json", return_value=({"approved": True, "issues": []}, {"input_tokens": 1, "output_tokens": 1})):
            result = week10._agent_work("evidence_review", {"query": "employee policy", "specialist_results": results})
        self.assertFalse(result["data"]["approved"])
        self.assertTrue(any("do not match" in issue for issue in result["data"]["issues"]))

    def test_a2a_task_response_returns_task_id_and_completion(self):
        message = {"messageId": "msg-1", "role": "user", "parts": [{"kind": "text", "text": json.dumps({"query": "hello"})}]}
        with patch.object(week10, "_agent_work", return_value={"tool_calls": [], "data": {"ok": True}}):
            envelope = week10._task_response("policy_research", 7, {"message": message})
        task = envelope["result"]
        self.assertEqual(envelope["id"], 7)
        self.assertEqual(task["status"]["state"], "completed")
        self.assertTrue(task["id"])
        payload = json.loads(task["artifacts"][0]["parts"][0]["text"])
        self.assertEqual(payload["data"], {"ok": True})


class Week10ScoringTests(unittest.TestCase):
    def test_answer_score_requires_expected_fact_and_real_chunk_reference(self):
        case = {"expected_terms": ["18 days"], "require_citation": True, "expected_refusal": False}
        result = {
            "status": "success",
            "answer": "18 days under policy P section 2; source chunk-123.",
            "specialist_results": [{"role": "policy_research", "data": {"results": [{"chunk_id": "chunk-123", "policy_id": "P", "section": "2"}]}}],
        }
        score = week10.score_answer(result, case)
        self.assertEqual(score["score"], 100)
        self.assertTrue(score["pass"])

    def test_answer_score_penalizes_missing_citation_and_expected_refusal(self):
        answerable = week10.score_answer({"status": "success", "answer": "18 days"}, {"expected_terms": ["18 days"], "require_citation": True})
        refusal = week10.score_answer({"status": "success", "answer": "The policy does not mention sabbatical."}, {"expected_refusal": True})
        self.assertEqual(answerable["score"], 80)
        self.assertFalse(answerable["citation_ok"])
        self.assertEqual(refusal["score"], 0)

    def test_frozen_eval_set_has_five_cases_and_refusal(self):
        cases = json.loads((ROOT / "data" / "week10_eval_cases.json").read_text(encoding="utf-8"))["cases"]
        self.assertEqual(len(cases), 5)
        self.assertTrue(any(case.get("expected_refusal") for case in cases))
        self.assertTrue(all(case.get("case_id") and case.get("query") for case in cases))


class Week10OrchestrationTests(unittest.TestCase):
    def test_employee_balance_filter_returns_structured_employee_details(self):
        query = "list out the employee details who has more then 3 leave balance"

        class FakeResponse:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(self.body).encode("utf-8")

        def fake_urlopen(url, timeout=5):
            del timeout
            if "/agent-card/" in url:
                role = url.rsplit("/", 1)[-1].removesuffix(".json")
                return FakeResponse(week10._a2a_card(role))
            return FakeResponse({"name": "test manager", "skills": []})

        activity = {
            "role": "leave_activity", "task_id": "task-activity", "state": "completed",
            "tool_calls": ["find_employee_ids_by_leave_transactions", "get_employee_details"],
            "data": {
                "status": "success",
                "criteria": {"field": "current_leave_balance", "operator": "gt", "threshold": 3.0},
                "employees": [{
                    "employee_id": "EMP-004", "name": "Demo Employee 004", "email": "employee004@example.com",
                    "region": "US", "policy_id": "ACME-HR-2026", "experience_years": 4,
                    "current_leave_balance": 8, "transaction": {"leave_used_days": 2},
                }],
            },
        }
        with (
            patch.object(week10, "ensure_a2a_server", return_value=week10.BASE_URL),
            patch.object(week10, "urlopen", side_effect=fake_urlopen),
            patch.object(week10, "_plan", return_value=(
                ["leave_activity"], "employee record filter", {"input_tokens": 0, "output_tokens": 0, "llm_calls": 0},
            )),
            patch.object(week10, "_a2a_send", return_value=activity),
            patch.object(week10, "_llm_text", side_effect=AssertionError("structured employee data needs no answer LLM")),
            patch.object(week10, "_append_task_trace"),
        ):
            result = week10.run_week10_team(query)

        self.assertEqual(result["status"], "success")
        self.assertIn("EMP-004", result["answer"])
        self.assertIn("current leave balance 8 days", result["answer"])
        self.assertEqual(result["metrics"]["llm_calls"], 0)
        self.assertEqual(result["stop_reason"], "employee_activity_data_returned_without_llm_reformatting")

    def test_team_runs_independent_specialists_in_parallel_then_reviews(self):
        roles = ["policy_research", "employee_directory", "leave_activity", "record_validator", "policy_calculator", "evidence_review"]
        active = 0
        peak_active = 0
        active_lock = threading.Lock()

        class FakeResponse:
            def __init__(self, body):
                self.body = body

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

            def read(self):
                return json.dumps(self.body).encode("utf-8")

        def fake_a2a_send(role, _payload):
            nonlocal active, peak_active
            if role in {"policy_research", "employee_directory", "leave_activity"}:
                with active_lock:
                    active += 1
                    peak_active = max(peak_active, active)
                time.sleep(0.04)
                with active_lock:
                    active -= 1
            base = {"role": role, "task_id": f"task-{role}", "state": "completed", "tool_calls": []}
            if role == "policy_research":
                base.update({"tool_calls": ["search_hr_policy"], "data": {"results": [{"policy_id": "AZURE-HR-2026", "section": "4", "chunk_id": "azure-chunk-4", "text": "Carry-forward limit is supported."}]}})
            elif role == "employee_directory":
                base.update({"tool_calls": ["get_employee_record"], "data": {"employee_id": "EMP-002", "policy_id": "AZURE-HR-2026", "current_leave_balance": 14}})
            elif role == "leave_activity":
                base.update({"tool_calls": ["get_employee_leave_transactions"], "data": {"leave_used_days": 2}})
            elif role == "record_validator":
                base.update({"data": {"valid": True, "issues": []}})
            elif role == "policy_calculator":
                base.update({"tool_calls": ["get_employee_leave_summary"], "data": {"status": "success", "policy_id": "AZURE-HR-2026", "values": {"current_leave_balance": 14}, "citations": [{"policy_id": "AZURE-HR-2026", "section": "4", "chunk_id": "azure-chunk-4"}]}})
            elif role == "evidence_review":
                base.update({"llm_usage": {"input_tokens": 4, "output_tokens": 2}, "data": {"approved": True, "issues": [], "citation_count": 1}})
            return base

        def fake_urlopen(url, timeout=5):
            del timeout
            if "/agent-card/" in url:
                role = url.rsplit("/", 1)[-1].removesuffix(".json")
                return FakeResponse(week10._a2a_card(role))
            return FakeResponse({"name": "test manager", "skills": []})

        final_answer = {"answer": "EMP-002's leave balance is 14 days. Source: AZURE-HR-2026 section 4, chunk azure-chunk-4."}
        with (
            patch.object(week10, "ensure_a2a_server", return_value=week10.BASE_URL),
            patch.object(week10, "urlopen", side_effect=fake_urlopen),
            patch.object(week10, "_plan", return_value=(roles, "parallel test plan", {"input_tokens": 5, "output_tokens": 2})),
            patch.object(week10, "_a2a_send", side_effect=fake_a2a_send),
            patch.object(week10, "_llm_text", return_value=(final_answer["answer"], {"input_tokens": 7, "output_tokens": 3})),
            patch.object(week10, "_append_task_trace"),
        ):
            result = week10.run_week10_team("How much leave balance does EMP-002 have?")

        self.assertGreaterEqual(peak_active, 2)
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["metrics"]["llm_calls"], 3)
        self.assertEqual(result["metrics"]["total_tokens"], 23)
        self.assertIn("azure-chunk-4", result["answer"])
        self.assertTrue(all(item["task_id"].startswith("task-") for item in result["tasks"]))
        self.assertEqual(result["steps"][-1]["step"], "Final response")


if __name__ == "__main__":
    unittest.main(verbosity=2)
