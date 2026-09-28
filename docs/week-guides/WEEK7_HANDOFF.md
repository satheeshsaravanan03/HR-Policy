# Week 7 Handoff — Workflows and Agents

This document records what the HR Policy RAG application has completed and the
planned Week 7 implementation. It is intended as a working handoff and review
document before coding begins.

## 1. What we have completed so far

The application is an HR-policy RAG assistant that retrieves policy evidence
before generating an answer.

Completed capabilities include:

- PDF and Markdown policy ingestion.
- Recursive and structure-aware chunking.
- Embedding and retrieval through Qdrant Cloud.
- Semantic vector search.
- BM25 keyword search.
- Hybrid semantic + keyword retrieval using RRF.
- Region, policy ID, and source-file metadata filtering.
- Local cross-encoder reranking.
- Query normalization for common spelling errors.
- Parent-context expansion.
- Citation-aware generation.
- Refusal gates for unsupported questions.
- JSONL request traces.
- Week 5 trace analysis foundation.
- Week 6 evaluation sets.
- Regression tests from earlier failures.
- Rule-based assertion checks.
- Before/after evaluation reports.
- LLM-as-judge calibration.

The current Streamlit modes are:

- **Ask:** retrieve evidence and generate a cited answer.
- **Retrieve:** show retrieved chunks and metadata.
- **Rerank:** show original versus reranked results.

These existing modes will remain unchanged during Week 7.

## 2. Why Week 7 is being added

The earlier work uses a mostly known sequence of RAG steps. Week 7 introduces
two important concepts:

- A **workflow** executes a known sequence reliably.
- An **agent** chooses the next action dynamically based on the previous result.

The goal is not to use an agent everywhere. The goal is to learn when a fixed
workflow is faster and safer, and when a dynamic agent is useful because the
required path changes with the question.

## 3. Week 7 modes

Only two new modes will be added.

### Mode A: Compare

The same question runs through two independent paths:

```text
User question
   ├── Separate fixed workflow → Answer A
   └── Separate standalone agent → Answer B
```

The workflow and agent must not reuse each other's answer or retrieved context.
The UI will compare:

- Final answers.
- Retrieved chunks.
- Citations.
- Steps taken.
- Tool calls.
- LLM calls.
- Runtime.
- Correctness.
- Refusal behavior.
- Failure or timeout reasons.

### Mode B: Agent + Workflow

The agent controls the sequence, but the actual work is performed by reliable
predefined workflows.

```text
Question
  ↓
Agent selects a workflow
  ↓
Workflow executes fixed steps
  ↓
Agent observes the result
  ↓
Agent answers, retries, calls another workflow, or refuses
```

## 4. Five predefined workflows

### Workflow 1: Policy lookup

For ordinary questions about one policy.

```text
Normalize query
→ Infer metadata
→ Hybrid retrieval
→ Rerank
→ Expand context
→ Validate evidence
```

### Workflow 2: Policy comparison

For questions involving multiple policies, regions, or employee types.

```text
Identify policies or regions
→ Retrieve each policy
→ Align relevant sections
→ Validate evidence
→ Prepare comparison context
```

### Workflow 3: Evidence validation

For checking whether a response is safe to generate.

```text
Inspect retrieved chunks
→ Check policy ID and section
→ Check citation support
→ Approve answer or refuse
```

### Workflow 4: Policy applicability

For determining which rule applies to a specific employee situation.

```text
Extract employee conditions
→ Identify region and employee type
→ Check tenure or eligibility requirements
→ Retrieve applicable policy sections
→ Return applicable rules and missing information
```

This workflow should not guess missing employee facts. It should identify what
is unknown and ask for clarification or refuse safely.

### Workflow 5: Policy audit

For auditing an answer or retrieved evidence before delivery.

```text
Check answer claims
→ Resolve every citation
→ Check policy ID and section
→ Detect conflicting or outdated evidence
→ Report unsupported claims
→ Approve, revise, or refuse
```

## 5. How the workflows will be included

Each workflow will be implemented as a normal Python function that returns a
structured result rather than only a text response.

Example result shape:

```python
{
    "workflow": "policy_lookup",
    "status": "success",
    "evidence": [...],
    "citations": [...],
    "missing_information": [],
    "evidence_ok": True,
    "reason": ""
}
```

The proposed implementation files are:

- `rag/workflows.py` — the five predefined workflows.
- `rag/agent.py` — the hand-built agent loop.
- `scripts/10_week7_agent_eval.py` — the comparison evaluation.
- `output/week7_agent_race.md` — the final performance report.

The Streamlit application will call these functions and display their returned
steps and results.

## 6. Agent loop

The agent will be deliberately small and visible:

```text
1. Read the question.
2. Select the next workflow.
3. Execute the workflow.
4. Inspect the structured result.
5. Decide whether to answer, retry, call another workflow, or refuse.
6. Stop and return the final result.
```

Every step will be logged with:

- Step number.
- Agent decision.
- Workflow name.
- Input.
- Result status.
- Evidence summary.
- Next decision.
- Stop reason.

## 7. Safety limits

The agent will have explicit limits:

- Maximum five agent steps.
- Maximum three retrieval/tool calls.
- Maximum 60 seconds runtime.
- No action outside the indexed policy system.
- Refusal when evidence is insufficient.
- No answer based on unsupported employee assumptions.
- Stop on timeout, repeated failure, or completed evidence validation.

## 8. Week 7 test cases

The same test set will be used for both Compare and Agent + Workflow modes:

- Normal single-policy question.
- Region-filtered question.
- Multi-policy comparison question.
- Eligibility question with missing information.
- Question requiring a retry.
- Typo-heavy question.
- Unsupported question that must be refused.
- Question with potentially conflicting policy evidence.

## 9. Measurements

The Week 7 evaluator will compare the independent fixed workflow and standalone
agent using:

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

The evaluation command will be:

```powershell
.\.venv\Scripts\python.exe scripts\10_week7_agent_eval.py
```

## 10. Expected Streamlit presentation

The Week 7 UI should show:

### Compare mode

- Question.
- Fixed workflow steps and answer.
- Standalone agent steps and answer.
- Side-by-side metrics.
- Difference or failure explanation.

### Agent + Workflow mode

- Agent decision at each step.
- Selected workflow.
- Workflow result.
- Follow-up decision.
- Final answer or refusal.
- Resolved citations.

## 11. Expected final recommendation

The final report will explain:

- Where the fixed workflow wins on speed, cost, or reliability.
- Where the agent succeeds because the path changes by question.
- Whether the agent introduced unnecessary calls or failures.
- Which architecture should be used for normal HR questions.
- Which architecture should be used for dynamic multi-policy tasks.

The expected learning outcome is not that the agent always wins. The expected
outcome is a measured decision about when to use a workflow, when to use an
agent, and how to combine them safely.

## 12. Implementation order

1. Freeze and record the current Week 6 baseline.
2. Implement the five workflows independently.
3. Unit-test each workflow with known questions.
4. Implement the hand-built agent loop.
5. Add step logging and safety limits.
6. Add Agent + Workflow mode to Streamlit.
7. Add independent fixed workflow and standalone agent paths.
8. Add Compare mode.
9. Run the Week 7 evaluation race.
10. Review the metrics and write the architecture recommendation.

No Week 7 code should change the existing retrieval or evaluation behavior
without rerunning the Week 6 regression suite.

## 13. Implementation instructions for another developer

Follow these instructions in order. Do not skip ahead to the Streamlit work.

### Prerequisites

1. Work from the project root: `D:\Project\HR-Policy`.
2. Install dependencies:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Confirm `.env` contains valid Qdrant and Groq settings.
4. Confirm the Qdrant collections contain the current corpus.
5. Run the existing Week 6 checks before editing:

   ```powershell
   .\.venv\Scripts\python.exe scripts\08_week6_eval.py
   .\.venv\Scripts\python.exe -m compileall -q rag scripts app.py
   ```

6. Save the Week 6 report as the Week 7 baseline.

### Required workflow interface

Implement each workflow with the same interface:

```python
def workflow_name(query: str, *, strategy: str = "structure",
                  top_k: int = 5) -> dict:
    ...
```

Each function must return:

```python
{
    "workflow": "workflow_name",
    "status": "success" | "refused" | "needs_input" | "error",
    "steps": [
        {"name": "...", "status": "success", "details": "..."}
    ],
    "hits": [],
    "citations": [],
    "answer_context": "",
    "missing_information": [],
    "evidence_ok": True,
    "reason": ""
}
```

Do not return an unstructured string from a workflow. The agent and Compare
mode need the structured fields to display and measure the execution.

### Workflow implementation order

1. Implement `policy_lookup` by composing existing normalization, search,
   reranking, parent-context, and citation validation functions.
2. Test it with Q1, Q3, Q7, and Q8 from `rag/questions.py`.
3. Implement `policy_comparison` by running lookup independently for each
   named policy and combining only their returned evidence.
4. Test it with a question comparing Acme and SoftSuave.
5. Implement `evidence_validation` using the existing refusal and citation
   checks. Test it with one supported and one unsupported question.
6. Implement `policy_applicability`. If employee information is absent, return
   `needs_input`; do not infer eligibility.
7. Implement `policy_audit` to report unsupported claims, unresolved
   citations, conflicting policy IDs, and missing sections.
8. Add unit-style checks for every workflow before building the agent.

### Required agent interface

Implement the agent as:

```python
def run_agent(query: str, *, max_steps: int = 5,
              timeout_seconds: int = 60) -> dict:
    ...
```

The agent result must contain:

```python
{
    "mode": "agent",
    "status": "success" | "refused" | "needs_input" | "error" | "timeout",
    "steps": [...],
    "workflows_called": [],
    "answer": "",
    "citations": [],
    "stop_reason": "",
    "metrics": {
        "step_count": 0,
        "tool_calls": 0,
        "llm_calls": 0,
        "elapsed_ms": 0
    }
}
```

The loop must stop when any of these conditions occurs:

- Evidence validation succeeds and an answer is generated.
- A workflow returns `needs_input`.
- A refusal is confirmed.
- `max_steps` is reached.
- The timeout is reached.
- The same failed action repeats.

### Compare mode instructions

1. Call the fixed workflow with the original query.
2. Call the standalone agent with the original query separately.
3. Do not pass the first result into the second path.
4. Record start/end time around each path.
5. Validate both answers using the same Week 6 assertions.
6. Display both step logs and both answers side by side.
7. Write one comparison record containing both metrics and the winner for each
   criterion.

### Streamlit acceptance criteria

The `Compare` mode is complete only when the UI visibly shows:

- The original question.
- Fixed workflow steps and answer.
- Standalone agent steps and answer.
- Citations and citation resolution status.
- Runtime and call counts for both paths.
- Refusal or error status for both paths.

The `Agent + Workflow` mode is complete only when the UI visibly shows:

- The agent decision at every step.
- The selected workflow.
- The workflow result.
- The next agent decision.
- The final stop reason.

### Required evaluation command

Create and run:

```powershell
.\.venv\Scripts\python.exe scripts\10_week7_agent_eval.py
```

The command must run the same cases through both paths and write:

```text
output/week7_agent_race.json
output/week7_agent_race.md
```

The Markdown report must contain one row per test case and aggregate results
for runtime, calls, answer correctness, citation validity, refusals, and
failures.

### Definition of done

Week 7 is complete only when all of the following are true:

- All five workflows return the documented structured result.
- Each workflow has at least one passing test.
- The agent visibly completes at least one multi-step task.
- The agent stops safely on an unsupported question.
- Compare mode runs independent workflow and agent paths.
- Agent + Workflow mode shows dynamic workflow selection.
- The Week 7 evaluation command completes successfully.
- The report contains real before/after path metrics.
- Week 6 regression tests still pass after the Week 7 changes.
- No secrets are written to traces or reports.
