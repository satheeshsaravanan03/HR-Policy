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


# These questions are authored from the current three-document corpus.
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
        query="How many annual leave days does an Acme full-time employee receive per calendar year?",
        policy_id="ACME-LEAVE-2026",
        section="2",
        known_answer="18 days of annual leave per calendar year",
        answer_pattern=r"18 days of annual leave",
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
        query="How many unused Acme annual-leave days may be carried into the next year?",
        policy_id="ACME-LEAVE-2026",
        section="3",
        known_answer="Up to 10 unused annual-leave days",
        answer_pattern=r"10 unused annual-leave days",
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
        query="How many days per week may an eligible Northstar employee work remotely?",
        policy_id="NORTHSTAR-REMOTE-2026",
        section="2",
        known_answer="Up to three days per week",
        answer_pattern=r"three days per week",
    ),
    Question(
        qid="Q8",
        query="How much notice must a permanent Acme employee give before resigning?",
        policy_id="ACME-EMP-2026",
        section="3",
        known_answer="30 calendar days of written notice",
        answer_pattern=r"30 calendar days of written notice",
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
        query="What is SoftSuave's sabbatical leave entitlement?",
        why_unanswerable=(
            "The current SoftSuave handbook does not define a sabbatical entitlement."
        ),
        difficulty="hard - correct employer, unsupported leave type",
    ),
    RefusalCase(
        qid="R2",
        query="What is Acme's dental insurance reimbursement limit?",
        why_unanswerable=(
            "The Acme leave policy contains no dental insurance reimbursement rule."
        ),
        difficulty="hard - correct employer, unrelated benefits topic",
    ),
    RefusalCase(
        qid="R3",
        query="Does Northstar provide a relocation bonus for remote employees?",
        why_unanswerable="Relocation bonuses are absent from the current three documents.",
        difficulty="baseline - no supporting content anywhere",
    ),
)

# Requirement 5 wants three answerable questions through generation. These
# reuse Q4, Q1 and Q7 so the cited answers rest on already-known ground truth.
CITED_ANSWER_QIDS = ("Q4", "Q1", "Q3")

BY_QID = {q.qid: q for q in QUESTIONS}
