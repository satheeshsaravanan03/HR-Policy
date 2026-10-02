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


def search_leave_transactions(
    field: str,
    *,
    operator: str = "gt",
    threshold: float = 0,
    second_field: str | None = None,
    second_operator: str | None = None,
    second_threshold: float | None = None,
    region: str | None = None,
    policy_id: str | None = None,
) -> list[dict[str, Any]]:
    """Find employee IDs matching one or two AND-combined numeric conditions."""
    allowed_fields = {
        "compensatory_leave_balance",
        "current_leave_balance",
        "leave_filed_days",
        "leave_used_days",
        "pending_leave_days",
    }
    conditions = [(field, operator, threshold)]
    if second_field is not None:
        if second_threshold is None or second_operator is None:
            raise ValueError("second_field requires second_operator and second_threshold")
        conditions.append((second_field, second_operator, second_threshold))
    for condition_field, condition_operator, _ in conditions:
        if condition_field not in allowed_fields:
            raise ValueError(f"field must be one of: {', '.join(sorted(allowed_fields))}")
        if condition_operator not in {"gt", "gte", "lt", "lte", "eq"}:
            raise ValueError("operator must be gt, gte, lt, lte, or eq")
    conditions = [(name, op, float(value)) for name, op, value in conditions]
    records = load_records().get("records", [])
    by_id = {str(item.get("employee_id", "")).upper(): item for item in records}
    matches: list[dict[str, Any]] = []

    def condition_matches(actual: float, condition_operator: str, expected: float) -> bool:
        return {
            "gt": actual > expected,
            "gte": actual >= expected,
            "lt": actual < expected,
            "lte": actual <= expected,
            "eq": actual == expected,
        }[condition_operator]

    transaction_rows = load_leave_transactions().get("records", [])
    if any(name == "current_leave_balance" for name, _, _ in conditions):
        candidates = [
            (record, {
                **(next((item for item in transaction_rows if str(item.get("employee_id", "")).upper() == str(record.get("employee_id", "")).upper()), {})),
                "current_leave_balance": record.get("current_leave_balance"),
            }) for record in records
        ]
    else:
        candidates = [
            (by_id.get(str(item.get("employee_id", "")).upper()), item)
            for item in transaction_rows
        ]

    for record, transaction in candidates:
        if record is None:
            continue
        if record is None:
            continue
        if region and str(record.get("region", "")).casefold() != region.casefold():
            continue
        if policy_id and str(record.get("policy_id", "")).casefold() != policy_id.casefold():
            continue
        if all(
            transaction.get(condition_field) is not None
            and condition_matches(float(transaction[condition_field]), condition_operator, expected)
            for condition_field, condition_operator, expected in conditions
        ):
            matches.append({
                "employee_id": record["employee_id"],
                "email": record["email"],
                "name": record["name"],
                "region": record["region"],
                "policy_id": record["policy_id"],
                "experience_years": record["experience_years"],
                "current_leave_balance": record["current_leave_balance"],
            })
    return matches


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
        # Names are accepted for conversational queries, but the returned
        # record remains keyed by the canonical employee_id.
        record_name = " ".join(str(record.get("name", "")).split()).strip().lower()
        if record_name and value == record_name:
            return dict(record)
    return None


def identifier_from_query(query: str) -> str | None:
    text = query or ""
    match = re.search(r"\b(?:EMP[-_ ]?\d+|CUST[-_ ]?\d+)\b", text, re.I)
    if match:
        return match.group(0).replace(" ", "-").upper()
    email = re.search(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text)
    if email:
        return email.group(0).lower()
    lowered = " ".join(text.casefold().split())
    # Resolve a full employee name to its canonical ID. Match longest names
    # first so a future shared prefix cannot select the wrong employee.
    for record in sorted(load_records().get("records", []), key=lambda item: len(str(item.get("name", ""))), reverse=True):
        name = " ".join(str(record.get("name", "")).casefold().split()).strip()
        if name and name in lowered:
            return str(record["employee_id"])
    return None


def identifiers_from_query(query: str) -> list[str]:
    """Return every employee ID or email mentioned, in order."""
    text = query or ""
    values = []
    for match in re.finditer(r"\b(?:EMP[-_ ]?\d+|CUST[-_ ]?\d+)\b", text, re.I):
        values.append(match.group(0).replace(" ", "-").upper())
    for match in re.finditer(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", text):
        values.append(match.group(0).lower())
    lowered = " ".join(text.casefold().split())
    for record in sorted(load_records().get("records", []), key=lambda item: len(str(item.get("name", ""))), reverse=True):
        name = " ".join(str(record.get("name", "")).casefold().split()).strip()
        if name and name in lowered:
            values.append(str(record["employee_id"]))
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
