"""Offline tests for Week 11 request tracing and cost metrics."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.generate import _estimated_cost, _stream_groq, refusal_check  # noqa: E402
from rag.tracing import redact, write_trace  # noqa: E402


class Week11TracingTests(unittest.TestCase):
    def test_conversational_policy_question_passes_absent_term_gate(self):
        hit = SimpleNamespace(
            rank=1, chunk_id="ACME-LEAVE-2026#section-2", policy_id="ACME-LEAVE-2026",
            section="2", score=0.9073, semantic_score=0.9073, content="ACME leave policy",
        )
        refused, gate, _ = refusal_check("tell me about ACME leave policy", [hit])
        self.assertFalse(refused)
        self.assertEqual(gate, "")

    def test_unknown_policy_topic_still_trips_absent_term_gate(self):
        hit = SimpleNamespace(
            rank=1, chunk_id="ACME-LEAVE-2026#section-2", policy_id="ACME-LEAVE-2026",
            section="2", score=0.9073, semantic_score=0.9073, content="ACME leave policy",
        )
        refused, gate, reason = refusal_check("tell me about ACME sabbatical policy", [hit])
        self.assertTrue(refused)
        self.assertEqual(gate, "absent_from_corpus")
        self.assertIn("sabbatical", reason)

    def test_colloquial_kinda_company_question_passes_unknown_word_gate(self):
        hit = SimpleNamespace(
            rank=1, chunk_id="SS-HB-2025#section-1", policy_id="SS-HB-2025",
            section="1", score=0.896, semantic_score=0.896, content="Company handbook",
        )
        refused, gate, _ = refusal_check("what kinda company softsuave", [hit])
        self.assertFalse(refused)
        self.assertEqual(gate, "")

    def test_redacts_email_employee_id_and_synthetic_name(self):
        safe = redact("EMP-005 Demo Employee 005 employee005@example.com employee John Doe")
        self.assertNotIn("EMP-005", safe)
        self.assertNotIn("Demo Employee 005", safe)
        self.assertNotIn("employee005@example.com", safe)
        self.assertNotIn("John Doe", safe)

    def test_general_policy_question_is_not_mistaken_for_personal_employee_query(self):
        hit = SimpleNamespace(
            rank=1, chunk_id="policy#chunk-1", policy_id="POLICY-1", section="2",
            score=0.8, retrieval_rank=1, retrieval_score=0.8, content="Policy clause",
        )
        with TemporaryDirectory() as tmp, patch("rag.tracing.TRACE_PATH", Path(tmp) / "trace.jsonl"):
            write_trace(
                query="What does an employee who works remotely need to do?",
                hits=[hit], output="Notify the manager [CITE: policy#chunk-1]", refused=False,
            )
            record = json.loads((Path(tmp) / "trace.jsonl").read_text(encoding="utf-8"))
        self.assertFalse(record["employee_specific"])
        self.assertEqual(record["raw_output"], "Notify the manager [CITE: policy#chunk-1]")

    def test_employee_specific_trace_omits_personal_answer_and_context(self):
        hit = SimpleNamespace(
            rank=1, chunk_id="policy#chunk-1", policy_id="POLICY-1", section="2",
            score=0.8, retrieval_rank=1, retrieval_score=0.8,
            content="Sensitive record content",
        )
        with TemporaryDirectory() as tmp, patch("rag.tracing.TRACE_PATH", Path(tmp) / "trace.jsonl"):
            trace_id = write_trace(
                query="Show leave balance for EMP-005",
                hits=[hit], output="Demo Employee 005 has 7 leave days.", refused=False,
                timings_ms={"retrieval": 12.0, "first_token": 90.0},
                usage={"input_tokens": 100, "output_tokens": 20},
                estimated_cost=0.0001, status="success",
            )
            record = json.loads((Path(tmp) / "trace.jsonl").read_text(encoding="utf-8"))

        self.assertEqual(record["trace_id"], trace_id)
        self.assertTrue(record["employee_specific"])
        self.assertTrue(record["employee_details_redacted"])
        self.assertEqual(record["raw_output"], "[EMPLOYEE-SPECIFIC ANSWER REDACTED]")
        self.assertEqual(record["retrieved"][0]["content"], "[EMPLOYEE-SPECIFIC CONTEXT REDACTED]")
        self.assertEqual(record["timings_ms"]["first_token"], 90.0)
        self.assertEqual(record["usage"]["input_tokens"], 100)

    def test_policy_trace_keeps_answer_and_cost_metadata(self):
        with TemporaryDirectory() as tmp, patch("rag.tracing.TRACE_PATH", Path(tmp) / "trace.jsonl"):
            write_trace(
                query="What is the notice period?", hits=[], output="30 days [CITE: policy#1]",
                refused=False, status="success", estimated_cost=0.0002,
            )
            record = json.loads((Path(tmp) / "trace.jsonl").read_text(encoding="utf-8"))
        self.assertEqual(record["raw_output"], "30 days [CITE: policy#1]")
        self.assertEqual(record["estimated_cost_usd"], 0.0002)

    def test_cost_accounts_for_provider_reported_cached_input(self):
        usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 400}
        rates = {"input_per_million": 1.0, "output_per_million": 2.0, "cached_input_per_million": 0.5}
        self.assertAlmostEqual(_estimated_cost(usage, rates), 0.001)

    def test_cost_is_unavailable_if_cached_rate_is_missing(self):
        usage = {"input_tokens": 1000, "output_tokens": 100, "cached_input_tokens": 400}
        rates = {"input_per_million": 1.0, "output_per_million": 2.0}
        self.assertIsNone(_estimated_cost(usage, rates))

    def test_streaming_captures_first_visible_token_and_final_usage(self):
        chunks = [
            {"choices": [{"delta": {"content": "30 calendar days"}}]},
            {"choices": [], "usage": {
                "prompt_tokens": 200, "completion_tokens": 30,
                "prompt_tokens_details": {"cached_tokens": 40},
                "completion_tokens_details": {"reasoning_tokens": 5},
            }},
        ]

        class FakeCompletions:
            def create(self, **kwargs):
                self.kwargs = kwargs
                return iter(chunks)

        completions = FakeCompletions()

        class FakeGroq:
            def __init__(self):
                self.chat = SimpleNamespace(completions=completions)

        with patch.dict(sys.modules, {"groq": SimpleNamespace(Groq=FakeGroq)}):
            text, first_token_ms, usage = _stream_groq([{"role": "user", "content": "question"}])

        self.assertEqual(text, "30 calendar days")
        self.assertIsNotNone(first_token_ms)
        self.assertEqual(usage, {
            "input_tokens": 200,
            "output_tokens": 30,
            "cached_input_tokens": 40,
            "reasoning_tokens": 5,
        })
        self.assertTrue(completions.kwargs["stream"])
        self.assertTrue(completions.kwargs["stream_options"]["include_usage"])

    def test_streaming_falls_back_when_sdk_rejects_stream_options(self):
        chunks = [{"choices": [{"delta": {"content": "Answer"}}]}]

        class OlderSdkCompletions:
            def create(self, **kwargs):
                if "stream_options" in kwargs:
                    raise TypeError("Completions.create() got an unexpected keyword argument 'stream_options'")
                self.kwargs = kwargs
                return iter(chunks)

        completions = OlderSdkCompletions()

        class FakeGroq:
            def __init__(self):
                self.chat = SimpleNamespace(completions=completions)

        with patch.dict(sys.modules, {"groq": SimpleNamespace(Groq=FakeGroq)}):
            text, first_token_ms, usage = _stream_groq([{"role": "user", "content": "question"}])

        self.assertEqual(text, "Answer")
        self.assertIsNotNone(first_token_ms)
        self.assertTrue(completions.kwargs["stream"])
        self.assertNotIn("stream_options", completions.kwargs)
        self.assertIsNone(usage["input_tokens"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
