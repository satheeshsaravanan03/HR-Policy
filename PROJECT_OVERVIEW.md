# HR Policy RAG Learning Project

## Purpose of this document

This document explains what the HR Policy application does today, what we plan
to add next, and why the project is being built. It is intended for review and
confirmation by a technical lead before the next implementation phase begins.

## What the application does now

The application is a retrieval-augmented generation (RAG) assistant for HR
policy documents. It retrieves policy evidence before generating an answer, so
the answer is grounded in the indexed documents instead of being based only on
the language model's general knowledge.

Current capabilities include:

- PDF and Markdown policy document ingestion.
- Local chunking with recursive and structure-aware strategies.
- Embedding and retrieval through Qdrant Cloud.
- Semantic vector search.
- BM25 keyword search.
- Hybrid search using semantic and keyword results.
- Reciprocal-rank fusion for merging retrieval results.
- Metadata filtering by region, policy ID, and source file.
- Local cross-encoder reranking.
- Query normalization for common spelling and spacing errors.
- Parent-context expansion for surrounding policy clauses.
- Citation-aware answer generation.
- Refusal of unsupported or out-of-corpus questions.
- JSONL traces containing questions, retrieved chunks, scores, settings, and
  generated outputs.
- Week 6 evaluation sets, regression tests, assertion checks, and an
  LLM-as-judge calibration.

The existing Streamlit RAG modes are:

- **Ask:** retrieve evidence and generate a cited answer.
- **Retrieve:** show retrieved chunks and metadata without generation.
- **Rerank:** show the original retrieval order and reranked order.

## Current evaluation status

The project now has repeatable evaluation instead of relying only on visual
inspection.

The Week 6 evaluator measures:

- Answerable questions.
- Questions that must be refused.
- Regression cases from earlier failures.
- Retrieval and evidence assertions.
- Baseline versus improved retrieval performance.

The current comparison is:

- **Baseline:** semantic retrieval without reranking.
- **Improved:** hybrid semantic + BM25 retrieval with local reranking.

The current recorded rule-based result is:

```text
Baseline: 96.4%
Improved: 98.2%
Delta: +1.8 percentage points
```

The LLM judge was calibrated against six human-labelled cases and achieved
100% agreement in the current calibration set. This is encouraging, but the
calibration set should grow as more real failures are found.

## What we plan to build next: Week 7

Week 7 introduces workflows and agents. The goal is not to replace the current
RAG pipeline. The goal is to understand when a fixed workflow is better and
when an agent is useful because the next step depends on the previous result.

Only two Week 7 modes will be added:

### 1. Compare mode

The same user question runs through two independent paths:

```text
User question
   ├── Separate fixed workflow → Answer A
   └── Separate standalone agent → Answer B
```

The two paths must not reuse each other's answer or retrieved context. The UI
will compare:

- Final answers.
- Retrieved evidence.
- Citations.
- Number of steps.
- Number of tool calls.
- Number of LLM calls.
- Runtime.
- Refusal correctness.
- Citation validity.
- Failure or timeout reasons.

### 2. Agent + Workflow mode

In this mode, the agent dynamically chooses and runs predefined workflows.
The workflows contain reliable fixed steps; the agent decides which workflow
to call next based on the result.

```text
Question
  ↓
Agent chooses a workflow
  ↓
Workflow executes fixed retrieval/evidence steps
  ↓
Agent observes the result
  ↓
Agent answers, retries, calls another workflow, or refuses
```

## Planned predefined workflows

### Policy lookup workflow

```text
Normalize query
→ Infer metadata
→ Hybrid retrieval
→ Rerank
→ Expand context
→ Validate evidence
```

### Policy comparison workflow

```text
Identify the policies or regions
→ Retrieve each policy
→ Align relevant sections
→ Validate evidence
→ Prepare comparison context
```

### Evidence validation workflow

```text
Inspect retrieved chunks
→ Check policy ID and section
→ Check citation support
→ Approve answer or refuse
```

The agent will not directly bypass these safety checks. It will select and
combine the workflows, while retrieval, citation validation, and refusal logic
remain grounded in the existing RAG implementation.

## Safety limits for the agent

The hand-built agent will have explicit limits:

- Maximum number of agent steps.
- Maximum retrieval/tool calls.
- Maximum execution time.
- A clear stop condition when evidence is sufficient.
- A safe refusal when evidence is insufficient.
- A visible log of every decision, tool call, result, and stop reason.

No complex agent framework, long-term memory system, or autonomous external
actions are required for the first implementation. The purpose is to make the
agent loop understandable and debuggable.

## Planned implementation files

- `rag/workflows.py` — predefined HR policy workflows.
- `rag/agent.py` — hand-built planning and execution loop.
- `scripts/10_week7_agent_eval.py` — agent versus workflow measurements.
- Streamlit additions for `Compare` and `Agent + Workflow` modes.
- `output/week7_agent_race.md` — speed, cost, reliability, and answer results.

## How the Week 7 result will be judged

The same test questions will run through both paths. We will compare:

| Metric | Fixed workflow | Standalone agent |
|---|---:|---:|
| Average runtime | | |
| Average steps | | |
| Tool calls | | |
| LLM calls | | |
| Estimated cost | | |
| Correct answers | | |
| Citation validity | | |
| Correct refusals | | |
| Failures/timeouts | | |

The final recommendation will state which approach should be used for normal
HR questions and which approach is useful for dynamic multi-policy questions.

## Why this learning project is being built

The project follows a staged AI engineering learning path:

1. Understand RAG fundamentals and context grounding.
2. Improve retrieval using chunking, semantic search, keyword search, hybrid
   search, metadata filtering, and reranking.
3. Record real traces and analyze retrieval and generation failures.
4. Build automatic evaluation sets, regression tests, assertions, and a
   calibrated LLM judge.
5. Learn workflows and agents by implementing both a fixed process and a
   dynamic decision-making process.

The purpose is not merely to make a chatbot that appears to work. The purpose
is to understand why it works, identify when it fails, measure whether a change
actually helps, and choose the simplest reliable architecture for each task.

## Approval requested before implementation

Before Week 7 coding begins, the technical lead should confirm:

- The two Week 7 modes are **Compare** and **Agent + Workflow**.
- Compare mode runs an independent fixed workflow and an independent
  standalone agent.
- Agent + Workflow mode lets the agent select predefined workflows.
- Existing Ask, Retrieve, and Rerank modes remain unchanged.
- The current HR corpus and Qdrant Cloud index remain the data source.
- The evaluation metrics above are sufficient for the Week 7 comparison.
