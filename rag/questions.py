"""The eight known-answer questions, and the six generation questions.

Written from the documents BEFORE any search was run, which is the ordering
the task insists on: questions authored after looking at retrieval output
measure the question-writing, not the chunker.

Scoring note. A hit is NOT chunk_id equality. The two strategies produce
different chunk_ids by construction, so comparing ids would make the two
numbers incomparable and the 30-point criterion unmeasurable. Instead a hit
requires that a top-5 chunk (a) comes from the expected policy_id and
(b) actually contains the answer, matched by answer_pattern. That test is
identical for both strategies, which is what makes the comparison fair.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Question:
    qid: str
    query: str
    policy_id: str
    section: str
    known_answer: str
    answer_pattern: str
    from_table_row: bool = False

    def matches(self, text: str) -> bool:
        return re.search(self.answer_pattern, text, re.I | re.S) is not None


# Three of these depend on a row inside an eligibility table, per requirement 2.
QUESTIONS: tuple[Question, ...] = (
    Question(
        qid="Q1",
        query="How many casual or sick leaves does an employee at the Bangalore office get in a calendar year?",
        policy_id="SS-HB-2025",
        section="9.2",
        known_answer="15 casual/sick leaves per calendar year",
        answer_pattern=r"Bangalore Office\s*\n?\s*15",
        from_table_row=True,
    ),
    Question(
        qid="Q2",
        query="How many privilege leaves are listed for the Chennai office?",
        policy_id="SS-HB-2025",
        section="9.2",
        known_answer="3 privilege leaves, at the discretion of the company",
        answer_pattern=r"Chennai Office\s*\n?\s*15\s*\n?\s*3",
        from_table_row=True,
    ),
    Question(
        qid="Q3",
        query="What is the maximum number of vacation hours accrued per 80-hour pay period for Arizona university staff?",
        policy_id="USM-3-113",
        section="3-113.3",
        known_answer="6.77 hours per 80-hour pay period",
        answer_pattern=r"6\.77",
        from_table_row=True,
    ),
    Question(
        qid="Q4",
        query="For non-technical staff, how many unused leaves can be carried over to the next year?",
        policy_id="SS-HB-2025",
        section="9.3",
        known_answer="A maximum of nine leaves carry over; technical staff get encashment instead",
        answer_pattern=r"nine leaves will be carried over",
    ),
    Question(
        qid="Q5",
        query="How many accrued vacation hours may Arizona staff carry forward each year?",
        policy_id="USM-3-113",
        section="3-113.5",
        known_answer="Up to 320 accrued hours, prorated by FTE; excess is forfeited",
        answer_pattern=r"320 accrued hours",
    ),
    Question(
        qid="Q6",
        query="What is the maximum maternity leave, and how much of it may be taken before delivery?",
        policy_id="SS-HB-2025",
        section="9.8",
        known_answer="26 weeks maximum, of which 8 weeks may precede delivery",
        answer_pattern=r"maximum of\s*26\s*\n?\s*weeks",
    ),
    Question(
        qid="Q7",
        query="How many university-designated holidays do Michigan faculty receive, plus any floating holiday?",
        policy_id="UM-FH-16",
        section="16.D.1",
        known_answer="Seven University-designated holidays plus one floating holiday",
        answer_pattern=r"seven University-\s*\n?\s*designated holidays",
    ),
    Question(
        qid="Q8",
        query="Do Research Foundation employees get paid for jury duty, and what documentation is required?",
        policy_id="RF-LEAVE",
        section="Jury Duty",
        known_answer="Full pay for necessary time off; a jury duty voucher and advance notice are required",
        answer_pattern=r"jury duty voucher",
    ),
)


@dataclass(frozen=True)
class RefusalCase:
    qid: str
    query: str
    why_unanswerable: str
    difficulty: str


# Graded by how hard they are to refuse. The first two are traps: the retriever
# returns confident, topically adjacent chunks, so a similarity threshold alone
# will not catch them.
REFUSALS: tuple[RefusalCase, ...] = (
    RefusalCase(
        qid="R1",
        query="What is Apple's parental leave entitlement for new parents?",
        why_unanswerable=(
            "Apple's Business Conduct Policy is in the corpus but defines no leave "
            "entitlement; its only mention of leave is an anti-retaliation clause. "
            "'Parental leave' appears once, in the Michigan faculty handbook, which "
            "is a different employer."
        ),
        difficulty="hard - entity mismatch, high-similarity Apple chunks retrieved",
    ),
    RefusalCase(
        qid="R2",
        query="What is the gratuity payout formula for a Soft Suave employee on resignation?",
        why_unanswerable=(
            "Soft Suave's separation policy (section 8) covers notice period, "
            "clearance and full-and-final settlement including Provident Fund, but "
            "'gratuity' appears nowhere in the corpus."
        ),
        difficulty="hard - topically adjacent, correct document retrieved",
    ),
    RefusalCase(
        qid="R3",
        query="What is the annual tuition reimbursement cap for employees?",
        why_unanswerable="'Tuition reimbursement' is absent from all six documents.",
        difficulty="baseline - no adjacent content anywhere",
    ),
)

# Requirement 5 wants three answerable questions through generation. These
# reuse Q4, Q1 and Q7 so the cited answers rest on already-known ground truth.
CITED_ANSWER_QIDS = ("Q4", "Q1", "Q7")

BY_QID = {q.qid: q for q in QUESTIONS}
