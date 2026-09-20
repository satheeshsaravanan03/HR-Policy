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
KEY_FIELDS = ("employee_id", "customer_id", "email")


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


def validate_record(record: dict[str, Any]) -> None:
    if not isinstance(record, dict):
        raise ValueError("each employee record must be an object")
    for field in ("employee_id", "email", "region", "policy_id"):
        if not str(record.get(field, "")).strip():
            raise ValueError(f"employee record requires {field}")
    for field in (
        "experience_years", "current_leave_balance", "compensatory_leave_balance",
        "annual_leave_entitlement", "carry_forward_cap", "compensatory_leave_cap",
    ):
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


def calculate_employee_case(record: dict[str, Any], query: str) -> dict[str, Any]:
    """Calculate requested values from exact record fields, without guessing."""
    lowered = (query or "").lower()
    balance = float(record["current_leave_balance"])
    carry_cap = float(record["carry_forward_cap"])
    comp_balance = float(record["compensatory_leave_balance"])
    requested: list[str] = []
    values: dict[str, float] = {}
    if any(term in lowered for term in ("balance", "available", "holding", "left")):
        requested.append("current_leave_balance")
        values["current_leave_balance"] = balance
    if any(term in lowered for term in ("carry", "forward", "next year")):
        requested.append("carry_forward")
        values["carry_forward"] = min(balance, carry_cap)
    if any(term in lowered for term in ("compensatory", "comp off", "comp-off", "comp off")):
        requested.append("compensatory_leave")
        values["compensatory_leave"] = min(comp_balance, float(record["compensatory_leave_cap"]))
    if any(term in lowered for term in ("annual entitlement", "annual leave", "entitlement")):
        requested.append("annual_leave_entitlement")
        values["annual_leave_entitlement"] = float(record["annual_leave_entitlement"])
    if not requested:
        requested = ["current_leave_balance", "carry_forward", "compensatory_leave"]
        values = {
            "current_leave_balance": balance,
            "carry_forward": min(balance, carry_cap),
            "compensatory_leave": min(comp_balance, float(record["compensatory_leave_cap"])),
        }
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
