"""Small editable structured-data store for the Week 7 employee prototype.

This data is intentionally separate from Qdrant policy embeddings. It models
the kind of live record an ERP API would provide later; no employee record is
embedded or sent to the vector store.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "employee_records.json"
TRANSACTION_PATH = Path(__file__).resolve().parent.parent / "data" / "leave_transactions.json"
KEY_FIELDS = ("employee_id", "email")
ALLOWED_RECORD_FIELDS = frozenset({
    "employee_id", "email", "name", "region", "policy_id",
    "experience_years", "current_leave_balance",
})


def load_records() -> dict[str, Any]:
    if not DATA_PATH.exists():
        return {"version": 1, "records": []}
    with DATA_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError(f"Invalid employee data file: {DATA_PATH}")
    return data


def save_records(data: dict[str, Any]) -> None:
    records = data.get("records")
    if not isinstance(records, list):
        raise ValueError("employee data must contain a records list")
    for record in records:
        validate_record(record)
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = DATA_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(DATA_PATH)


def load_leave_transactions() -> dict[str, Any]:
    """Load mutable leave transactions separately from employee identity data."""
    if not TRANSACTION_PATH.exists():
        return {"version": 1, "records": []}
    with TRANSACTION_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("records"), list):
        raise ValueError(f"Invalid leave transaction file: {TRANSACTION_PATH}")
    return data


def lookup_leave_transaction(employee_id: str, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    value = str(employee_id or "").strip().lower()
    for item in (data or load_leave_transactions()).get("records", []):
        if value == str(item.get("employee_id", "")).strip().lower():
            return dict(item)
    return None


def validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("each employee record must be an object")
    unknown = set(record) - ALLOWED_RECORD_FIELDS
    if unknown:
        raise ValueError(f"employee record contains unsupported fields: {', '.join(sorted(unknown))}")
    for field in ("employee_id", "email", "name", "region", "policy_id"):
        if not str(record.get(field, "")).strip():
            raise ValueError(f"employee record requires {field}")
    for field in ("experience_years", "current_leave_balance"):
        try:
            if float(record.get(field)) < 0:
                raise ValueError(f"{field} cannot be negative")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field} must be a non-negative number") from exc


def lookup_record(identifier: str, data: dict[str, Any] | None = None) -> dict[str, Any] | None:
    value = str(identifier or "").strip().lower()
    if not value:
        return None
    records = (data or load_records()).get("records", [])
    for record in records:
        if any(value == str(record.get(field, "")).strip().lower() for field in KEY_FIELDS):
            return dict(record)
    return None


def identifier_from_query(query: str) -> str | None:
    text = query or ""
    match = re.search(r"\b(?:EMP[-_ ]?\d+|CUST[-_ ]?\d+)\b", text, re.I)
    if match:
        return match.group(0).replace(" ", "-").upper()
    email = re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
    return email.group(0).lower() if email else None


def identifiers_from_query(query: str) -> list[str]:
    """Return every employee ID or email mentioned, in order."""
    text = query or ""
    values = []
    for match in re.finditer(r"\b(?:EMP[-_ ]?\d+|CUST[-_ ]?\d+)\b", text, re.I):
        values.append(match.group(0).replace(" ", "-").upper())
    for match in re.finditer(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        values.append(match.group(0).lower())
    return list(dict.fromkeys(values))


def calculate_employee_case(
    record: dict[str, Any],
    query: str,
    *,
    policy_rules: dict[str, float | None] | None = None,
    transaction: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Calculate values from live data plus explicitly retrieved policy rules."""
    lowered = (query or "").lower()
    balance = float(record["current_leave_balance"])
    rules = policy_rules or {}
    carry_cap = rules.get("carry_forward_cap")
    annual_entitlement = rules.get("annual_leave_entitlement")
    comp_cap = rules.get("compensatory_leave_cap")
    comp_balance = (transaction or {}).get("compensatory_leave_balance")
    requested: list[str] = []
    values: dict[str, float] = {}
    if any(term in lowered for term in ("balance", "available", "holding", "left")):
        requested.append("current_leave_balance")
        values["current_leave_balance"] = balance
    if any(term in lowered for term in ("carry", "forward", "next year")) and carry_cap is not None:
        requested.append("carry_forward")
        values["carry_forward"] = min(balance, float(carry_cap))
    if any(term in lowered for term in ("compensatory", "comp off", "comp-off", "comp off")) and comp_balance is not None and comp_cap is not None:
        requested.append("compensatory_leave")
        values["compensatory_leave"] = min(float(comp_balance), float(comp_cap))
    if any(term in lowered for term in ("annual entitlement", "annual leave", "entitlement")) and annual_entitlement is not None:
        requested.append("annual_leave_entitlement")
        values["annual_leave_entitlement"] = float(annual_entitlement)
    if not requested:
        requested = ["current_leave_balance"]
        values = {"current_leave_balance": balance}
        if carry_cap is not None:
            requested.append("carry_forward")
            values["carry_forward"] = min(balance, float(carry_cap))
        if comp_balance is not None and comp_cap is not None:
            requested.append("compensatory_leave")
            values["compensatory_leave"] = min(float(comp_balance), float(comp_cap))
    return {"requested": requested, "values": values}


def format_employee_answer(record: dict[str, Any], calculation: dict[str, Any]) -> str:
    labels = {
        "current_leave_balance": "Current leave balance",
        "carry_forward": "Carry-forward available",
        "compensatory_leave": "Compensatory leave available",
        "annual_leave_entitlement": "Annual leave entitlement",
    }
    lines = [f"Employee `{record['employee_id']}` ({record['region']}) — structured record result:"]
    for key in calculation["requested"]:
        value = calculation["values"][key]
        shown = int(value) if float(value).is_integer() else value
        lines.append(f"- {labels[key]}: **{shown} days**")
    lines.append("\nSource: editable structured employee record; policy calculations are not embedded in Qdrant.")
    return "\n".join(lines)


def format_employee_comparison(records: list[dict[str, Any]]) -> str:
    lines = ["Employee structured-record comparison:", ""]
    for record in records:
        carry = min(float(record["current_leave_balance"]), float(record["carry_forward_cap"]))
        comp = min(float(record["compensatory_leave_balance"]), float(record["compensatory_leave_cap"]))
        lines.append(
            f"- **{record['employee_id']}** ({record['region']}, {record['experience_years']} years): "
            f"balance {record['current_leave_balance']} days; carry-forward {carry:g} days; "
            f"compensatory leave {comp:g} days; policy `{record['policy_id']}`."
        )
    lines.append("\nThese are structured demo-record values. Policy evidence must be available in Qdrant before treating them as authoritative policy rules.")
    return "\n".join(lines)
