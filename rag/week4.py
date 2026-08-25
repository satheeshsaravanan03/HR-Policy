"""Pre-registered Week 4 retrieval challenge set and inspection cases.

The retrieval cases are deliberately phrased with policy codes, section ids,
office names and exact entitlement terms.  Those are the terms a vector-only
search can underweight and that BM25 is designed to preserve.  The script
evaluates every case identically before and after hybrid RRF.

The generation cases are kept separate: their required context spans two
sections, so a retrieval hit alone does not prove that a generated answer is
safe.  They are evidence for a future parent-context change, not evidence that
hybrid retrieval worked.
"""

from __future__ import annotations

from dataclasses import dataclass

from .questions import Question


RETRIEVAL_CHALLENGES: tuple[Question, ...] = (
    Question(
        "W4-R1", "Under policy USM 3-113.5, what is the carry-forward cap?",
        "USM-3-113", "3-113.5", "320 accrued hours", r"320 accrued hours",
    ),
    Question(
        "W4-R2", "USM 3-113.3: maximum accrual rate for an 80-hour pay period?",
        "USM-3-113", "3-113.3", "6.77 hours", r"6\.77",
    ),
    Question(
        "W4-R3", "SS-HB-2025 section 9.8 maternity leave before delivery?",
        "SS-HB-2025", "9.8", "26 weeks; eight before delivery", r"maximum of\s*26\s*\n?\s*weeks",
    ),
    Question(
        "W4-R4", "SS-HB-2025 9.3 non-technical leave carry-over limit?",
        "SS-HB-2025", "9.3", "nine leaves", r"nine leaves will be carried over",
    ),
    Question(
        "W4-R5", "Bangalore Office casual/sick leave entitlement in the calendar year?",
        "SS-HB-2025", "9.2", "15 casual/sick leaves", r"Bangalore Office\s*\n?\s*15",
        True,
    ),
    Question(
        "W4-R6", "Chennai Office privilege leave entitlement?",
        "SS-HB-2025", "9.2", "3 privilege leaves", r"Chennai Office\s*\n?\s*15\s*\n?\s*3",
        True,
    ),
    Question(
        "W4-R7", "Faculty handbook 16.D.1: how many floating holidays?",
        "UM-FH-16", "16.D.1", "one floating holiday", r"one floating holiday",
    ),
    Question(
        "W4-R8", "Research Foundation jury duty voucher and pay policy?",
        "RF-LEAVE", "Jury Duty", "full pay and jury duty voucher", r"jury duty voucher",
    ),
)


@dataclass(frozen=True)
class GenerationInspection:
    qid: str
    query: str
    required_sections: tuple[str, ...]
    expected_policy_id: str
    diagnosis: str


GENERATION_INSPECTIONS: tuple[GenerationInspection, ...] = (
    GenerationInspection(
        "W4-G1",
        "I joined Soft Suave eight months ago. How many unused leaves can I carry forward?",
        ("9.2", "9.3"),
        "SS-HB-2025",
        "Generation/context risk: 9.3 states the nine-leave cap, while 9.2 holds "
        "the eligibility conditions. A correct answer needs both sections.",
    ),
)
