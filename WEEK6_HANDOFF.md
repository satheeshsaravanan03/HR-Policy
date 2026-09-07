# Week 6 Handoff — Evals and Error Analysis

This document records the Week 6 changes currently present in the working tree
and explains how to test them from another system.

## Goal

Week 6 turns retrieval changes into repeatable measurements. The app now has a
one-command evaluation suite rather than relying on manual inspection.

## What was added

### 1. Evaluation sets

`scripts/08_week6_eval.py` evaluates three sets:

- **Answerable:** the eight known HR-policy questions from `rag/questions.py`.
- **Refusal:** the three out-of-corpus questions that must be refused.
- **Regression:** three permanent tests based on earlier failures:
  - misspelled notice-period question;
  - Northstar remote-work retrieval;
  - unsupported SoftSuave sabbatical question.

### 2. Assertion checks

Each case checks:

- retrieval returned chunks;
- the expected answer pattern was found, when answerable;
- refusal happened when refusal was expected;
- supporting source context was available.

The assertions are reported per case and by test set.

### 3. Before/after comparison

The evaluator compares:

- **Baseline:** semantic retrieval, no reranking.
- **Improved:** hybrid semantic + BM25 RRF retrieval with local reranking.

The report includes overall, answerable, refusal, regression, and per-case
scores.

### 4. LLM judge

`scripts/09_llm_judge.py` runs a six-case calibration set:

- three answerable questions;
- three refusal questions.

The Groq judge scores faithfulness, relevance, completeness, refusal
correctness, and overall pass/fail. It compares the judge decision with the
human-labelled expected outcome before marking the judge as validated.

The current calibration achieved 100% agreement on six cases. Expand the
calibration set before treating this as a production-quality metric.

### 5. Streamlit trigger

The sidebar now contains:

`Week 6 evaluations → Run Week 6 evaluation`

The button runs both evaluation scripts and displays their output. It also
offers downloads for the generated Markdown and JSON reports.

## Files

| File | Purpose |
|---|---|
| `scripts/08_week6_eval.py` | Rule-based eval sets, regression tests, assertions, before/after scores |
| `scripts/09_llm_judge.py` | Groq LLM judge and human calibration agreement |
| `output/week6_eval.md` | Human-readable evaluation report |
| `output/week6_eval.json` | Machine-readable evaluation results |
| `output/week6_judge.json` | Judge calibration result |
| `app.py` | Streamlit Week 6 evaluation button |
| `requirements.txt` | Reproducible Python dependency list |

## Setup on another system

```powershell
cd D:\Project\HR-Policy
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Copy `.env` separately. Do not commit it because it contains secrets. Required
values for the current setup are:

```text
VECTOR_STORE=qdrant
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=hr_policy_v1
GROQ_API_KEY=your_groq_key
GROQ_GENERATION_MODEL=openai/gpt-oss-120b
```

The Qdrant collections must contain the same indexed corpus:

- `hr_policy_v1_recursive`
- `hr_policy_v1_structure`

## Test from the terminal

Run the rule-based evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\08_week6_eval.py
```

Run the LLM judge calibration:

```powershell
.\.venv\Scripts\python.exe scripts\09_llm_judge.py
```

Start the UI:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Then click the Week 6 evaluation button in the sidebar.

## How to compare a future change

1. Run `scripts/08_week6_eval.py` before changing code.
2. Save or copy `output/week6_eval.md` as the baseline report.
3. Make exactly one retrieval or generation change.
4. Run the same command again.
5. Compare overall and per-set scores.
6. Inspect any failed case in `output/week6_eval.json`.
7. Run `scripts/09_llm_judge.py` if the change affects generated answers.
8. Keep a change only when the target category improves without unacceptable
   regressions in refusals, citations, or other categories.

The important comparison is:

```text
baseline score → improved score → delta
```

Do not compare runs with different questions, documents, embedding models, or
retrieval settings unless that difference is the experiment being measured.

## Current recorded result

The current rule-based run recorded:

```text
Baseline: 96.4%
Improved: 98.2%
Delta: +1.8 percentage points
```

The current six-case LLM-judge calibration recorded:

```text
Human–judge agreement: 100%
Validated: true
```

These numbers are evaluation results for the current case set, not a guarantee
of accuracy on every future HR question. Add new real failures as regression
cases whenever they are discovered.
