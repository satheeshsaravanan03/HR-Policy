"""Validation tests for the editable Week 7 employee-data prototype."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.agent import run_agent  # noqa: E402
from rag.employee_data import identifier_from_query, load_records, lookup_record  # noqa: E402
from rag.workflows import employee_case  # noqa: E402


def main() -> None:
    data = load_records()
    assert len(data["records"]) >= 2
    assert identifier_from_query("show EMP-001 leave balance") == "EMP-001"
    assert identifier_from_query("employee001@example.com carry forward") == "employee001@example.com"
    assert lookup_record("EMP-001")["current_leave_balance"] == 7

    result = employee_case("How much leave can EMP-001 carry forward?")
    assert result["status"] == "success"
    assert result["calculation"]["values"]["carry_forward"] == 5

    missing = run_agent("What is the leave balance for an employee?")
    assert missing["status"] == "needs_input"
    assert missing["stop_reason"] == "employee_data_missing"

    answer = run_agent("How much leave can employee001@example.com carry forward?")
    assert answer["status"] == "success"
    assert "employee_case" in answer["workflows_called"]
    print("Employee structured-data tests: PASS")


if __name__ == "__main__":
    main()
