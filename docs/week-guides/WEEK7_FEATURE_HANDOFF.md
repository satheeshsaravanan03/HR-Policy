# Week 7 Feature Handoff: Workflows and Agents

## 1. Executive Summary

Week 7 expands the HR Policy RAG assistant from a static linear retrieval pipeline into a **Workflows and Agents architecture**. 

The goal of Week 7 is not to replace deterministic RAG with an unconstrained agent. Instead, it demonstrates the boundary between **fixed workflows** (which execute a known sequence reliably and cheaply) and **autonomous agents** (which dynamically inspect intermediate results and select the next action when the path depends on the user's question).

---

## 2. Core Architecture & Concepts

```text
                                 User Question
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
       Mode 1: Compare Mode                         Mode 2: Agent + Workflow
  ┌───────────────────────────┐                   ┌───────────────────────────┐
  │ Path A: Fixed Workflow    │                   │ Agent Intent Analysis     │
  │ • Normalize & Infer       │                   │          │                │
  │ • Single Hybrid Search    │                   │   Selects Workflow        │
  │ • Refusal Gate            │                   │          │                │
  │ • Direct LLM Generation   │                   │ Executes Structured Step  │
  ├───────────────────────────┤                   │          │                │
  │ Path B: Standalone Agent  │                   │ Inspects Result Dict      │
  │ • Dynamic Intent Analysis │                   │          │                │
  │ • Specialized Workflow    │                   │ • needs_input -> clarify  │
  │ • Evidence Validation     │                   │ • refused -> stop         │
  │ • LLM Gen + Policy Audit  │                   │ • success -> LLM + Audit  │
  └───────────────────────────┘                   └───────────────────────────┘
```

### The Two New Modes in Streamlit
1. **Compare Mode:** Runs the exact same user query through two independent paths without shared state or context, displaying side-by-side metrics, step execution tables, final answers, citation validity, and speed/accuracy winner analysis.
2. **Agent + Workflow Mode:** The agent acts as an orchestrator, determining the sequence dynamically and delegating execution to predefined, reliable workflows with an inspectable decision trace.

---

## 3. The Five Predefined Workflows (`rag/workflows.py`)

Every workflow is implemented as an independent Python function that returns a standardized dictionary rather than raw text:

```python
{
    "workflow": str,                  # Name of the workflow
    "status": str,                    # "success" | "refused" | "needs_input" | "error"
    "steps": list[dict],              # Detailed trace of internal actions taken
    "hits": list[Hit],                # Retrieved policy chunk dataclasses
    "citations": list[Citation],      # Extracted citation objects
    "answer_context": str,            # Formatted context blocks for generation
    "missing_information": list[str], # Populated if status == "needs_input"
    "evidence_ok": bool,              # True if all validation gates pass
    "reason": str                     # Refusal, clarification, or failure reason
}
```

### Summary of Workflows
| Workflow | Function Signature | Purpose & Execution Flow |
|---|---|---|
| **1. Policy Lookup** | `policy_lookup(query, *, strategy="structure", top_k=5)` | Standard single-policy lookup. Normalizes query, infers region/policy metadata, runs hybrid vector + BM25 search with local cross-encoder reranking, applies parent-context expansion, and validates against refusal gates. |
| **2. Policy Comparison** | `policy_comparison(query, *, strategy="structure", top_k=6)` | Multi-entity comparison. Identifies distinct policies or regions (e.g., Acme vs. SoftSuave), executes **isolated retrieval for each entity**, aligns and interleaves relevant clauses, and validates comparative evidence. |
| **3. Evidence Validation** | `evidence_validation(query, *, hits=None, strategy="structure", top_k=5)` | Pre-generation gatekeeper. Checks candidate chunk relevance against the `0.30` semantic floor, confirms explicit policy ID and section presence, and validates terms against corpus vocabulary. |
| **4. Policy Applicability** | `policy_applicability(query, *, employee_info=None, strategy="structure", top_k=5)` | Determines employee eligibility. Extracts `location/region`, `employment_type` (permanent/probation/contractor), and `tenure`. If essential conditions are missing, returns `status="needs_input"` without hallucinating employee facts. |
| **5. Policy Audit** | `policy_audit(query, *, answer_text="", citations=None, hits=None)` | Post-generation auditor. Verifies that every `[CITE: chunk_id]` in the generated text resolves to a supplied chunk and detects cross-policy conflicting evidence. |

---

## 4. Standalone Agent & Independent Fixed Workflow (`rag/agent.py`)

### Standalone Agent (`run_agent`)
- **Intent Dispatcher:** Analyzes query semantics to select initial workflow:
  - Comparison keywords (`compare`, `versus`, `difference`, `between`) $\rightarrow$ `policy_comparison`.
  - Eligibility keywords (`am i eligible`, `can i work`, `do i qualify`) $\rightarrow$ `policy_applicability`.
  - General policy questions $\rightarrow$ `policy_lookup`.
- **Dynamic Decision Loop:**
  - If workflow returns `needs_input` $\rightarrow$ stops and prompts user for missing facts.
  - If workflow returns `refused` $\rightarrow$ stops and outputs grounded refusal without calling LLM.
  - If workflow returns `success` $\rightarrow$ invokes Groq LLM (`openai/gpt-oss-120b`) with verified evidence, then executes `policy_audit` before approving.
- **Strict Safety Ceilings:**
  - Maximum **5 steps**.
  - Maximum **3 tool/retrieval calls**.
  - Maximum **60 seconds** execution timeout.
  - Halts on completed audit, refusal, clarification, or safety limit breach.

### Fixed Workflow (`run_fixed_workflow`)
- Executes the linear sequence: `normalize` $\rightarrow$ `hybrid search + rerank` $\rightarrow$ `refusal check` $\rightarrow$ `LLM generation`.
- Produces an identical metric signature to allow direct side-by-side comparison in Compare mode.

---

## 5. Walkthrough of Scenarios (Why Fixed Workflow Fails vs. Why Agent Succeeds)

### Scenario A: Multi-Policy Comparison
- **Query:** *"Compare annual leave between Acme and SoftSuave employees."*
- **Fixed Workflow:** Performs a single query. It retrieves chunks predominantly from one document, fails entity grounding, and triggers code-level refusal.
- **Standalone Agent:** Selects `policy_comparison`, retrieves 3 chunks from Acme and 3 chunks from SoftSuave separately, interleaves them, and generates an accurate comparative table with valid citations.
- **Winner:** **Agent**.

### Scenario B: Missing Employee Information
- **Query:** *"Am I eligible to work from home full-time?"*
- **Fixed Workflow:** Blindly retrieves general remote-work clauses and generates an answer making ungrounded assumptions about the employee's role and location.
- **Standalone Agent:** Selects `policy_applicability`, detects that office location and employment type are missing, halts with `needs_input`, and asks for clarification.
- **Winner:** **Agent** (prevents legal and compliance hallucinations).

### Scenario C: Routine Lookup
- **Query:** *"How many annual leave days does an Acme full-time employee receive per calendar year?"*
- **Fixed Workflow:** Completes in 1 retrieval step and 1 generation call (~7 seconds).
- **Standalone Agent:** Takes 2 steps (Lookup $\rightarrow$ Audit) with comparable accuracy (~7 seconds).
- **Winner:** **Fixed Workflow** (simpler, lower overhead on routine queries).

---

## 6. Evaluation Benchmark Results (`scripts/10_week7_agent_eval.py`)

The evaluation race ran 8 canonical test cases representing single-policy, region-filtered, comparative, ambiguous, typo-heavy, and refusal queries.

### Aggregate Comparison Table
| Metric | Fixed Workflow | Standalone Agent | Analysis |
|---|---:|---:|---|
| **Average Runtime** | 12,048.6 ms | **11,411.4 ms** | Comparable overall latency across all test cases |
| **Average Steps** | 3.0 | 1.8 | Fixed workflow uses fixed steps; Agent short-circuits on clarification |
| **Total Tool Calls** | **8** | **8** | Both paths stayed strictly within tool call budgets |
| **Total LLM Calls** | 6 | 6 | Comparable LLM usage |
| **Correct Answers** | 5 / 8 (62.5%) | **6 / 8 (75.0%)** | **Agent won** on comparison and missing fact handling |
| **Citation Validity** | **100.0%** | **100.0%** | 100% of generated citations resolved to valid chunks |
| **Correct Refusals** | **1 / 1 (100%)** | **1 / 1 (100%)** | Both cleanly refused unsupported out-of-corpus topics |
| **Failures / Timeouts** | **0** | **0** | Zero crashes, exceptions, or timeouts |

### Per-Case Benchmark Table
| ID | Type | Query | Fixed Status | Agent Status | Workflows Called | Winner / Rationale |
|---|---|---|:---:|:---:|---|---|
| **TC-01** | Single-policy | Acme annual leave days | `success` | `success` | `policy_lookup`, `policy_audit` | **Tie** (both accurate) |
| **TC-02** | Region-filtered | Bangalore casual/sick leaves | `success` | `success` | `policy_lookup`, `policy_audit` | **Tie** (both accurate) |
| **TC-03** | Multi-policy comparison | Compare Acme vs. SoftSuave | `refused` | `success` | `policy_comparison`, `policy_audit` | **Agent** (interleaved comparative context) |
| **TC-04** | Missing facts | Eligible to work from home | `success` (guessed) | `needs_input` | `policy_applicability` | **Agent** (requested facts instead of guessing) |
| **TC-05** | Reformulation / retry | Notice periond permanent terms | `success` | `success` | `policy_lookup`, `policy_audit` | **Tie** (normalized spelling noise) |
| **TC-06** | Typo-heavy | Nottice periond permanent Acme employe | `success` | `success` | `policy_lookup`, `policy_audit` | **Tie** (both resolved) |
| **TC-07** | Unsupported refusal | SoftSuave sabbatical entitlement | `refused` | `refused` | `policy_lookup` | **Tie** (both refused safely in code) |
| **TC-08** | Conflicting policies | Core working hours & attendance | `success` | `success` | `policy_lookup`, `policy_audit` | **Fixed** (faster on general policy text) |

*Full evaluation reports:*
- Markdown: `output/week7_agent_race.md`
- JSON: `output/week7_agent_race.json`

---

## 7. Streamlit User Interface (`app.py`)

### New Controls & Features
1. **Sidebar Mode Radio:**
   - `Ask`: Standard single RAG generation.
   - `Retrieve`: Search-only chunk inspector.
   - `Rerank`: Candidate comparison before/after cross-encoder.
   - **`Compare`**: Side-by-side execution of Fixed Workflow vs. Standalone Agent.
   - **`Agent + Workflow`**: Interactive agent orchestration trace and audited output.
2. **Presets Tab:**
   - **Preset: Week 7 scenarios**: Instant 1-click execution for comparison (`W7-01`), missing information (`W7-02`), spelling noise (`W7-03`), and multi-policy overlap (`W7-04`).
3. **Sidebar Week 7 Evaluation Runner:**
   - Expandable section to trigger `scripts/10_week7_agent_eval.py` asynchronously and download generated JSON/Markdown reports.

---

## 8. Codebase File Index

| File | Status | Description |
|---|---|---|
| `rag/workflows.py` | New | Contains all 5 predefined workflows returning structured dictionaries. |
| `rag/agent.py` | New | Implements `run_agent` with safety limits and `run_fixed_workflow` for Compare mode. |
| `scripts/test_workflows.py` | New | Automated unit test suite verifying all 5 workflows pass. |
| `scripts/test_agent.py` | New | Automated test suite verifying the agent loop, safety limits, and stopping conditions. |
| `scripts/10_week7_agent_eval.py` | New | Evaluation suite running the 8 benchmark cases and recording metrics. |
| `output/week7_agent_race.md` | Generated | Human-readable race report comparing Fixed Workflow vs. Agent. |
| `output/week7_agent_race.json` | Generated | Machine-readable race report with execution details. |
| `app.py` | Modified | Streamlit console with Compare and Agent+Workflow modes, presets, and race runner. |

---

## 9. Verification Commands

### 9.1 Prerequisites & Setup

```powershell
# 1. Create and activate virtual environment
cd D:\Project\HR-Policy
py -3.13 -m venv .venv

# 2. Install dependencies
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Create a `.env` file in the project root with the following keys:
```text
VECTOR_STORE=qdrant
QDRANT_URL=your_qdrant_cloud_url
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_COLLECTION=hr_policy_v1
GROQ_API_KEY=your_groq_api_key
GROQ_GENERATION_MODEL=openai/gpt-oss-120b
```

### 9.2 Verification Commands

Run the following commands in PowerShell from the project root:

```powershell
# 1. Run workflow unit tests
.\.venv\Scripts\python.exe scripts\test_workflows.py

# 2. Run agent unit tests
.\.venv\Scripts\python.exe scripts\test_agent.py

# 3. Run Week 7 evaluation race benchmark
.\.venv\Scripts\python.exe scripts\10_week7_agent_eval.py

# 4. Verify zero regressions on Week 6 baseline (98.2%)
.\.venv\Scripts\python.exe scripts\08_week6_eval.py

# 5. Launch the Streamlit application
.\.venv\Scripts\python.exe -m streamlit run app.py
```

---

## 10. Architecture Recommendation for Production

1. **Do not use an unconstrained agent for routine queries:** Fixed workflows are faster, cost less, and eliminate unnecessary LLM reasoning steps when the question maps to a single known policy.
2. **Use agents for dynamic routing and multi-step reasoning:** Agents excel at identifying multi-policy comparisons, recognizing when user-supplied facts are insufficient, and performing post-generation audits.
3. **The Recommended Hybrid Pattern (Agent + Workflow):** In production, employ a lightweight classifier agent that routes simple queries directly into fast deterministic workflows (`policy_lookup`) and reserves multi-step orchestration (`policy_comparison`, `policy_applicability`, `policy_audit`) for complex scenarios.

---

## 11. Dynamic employee-data prototype

The Week 7 agent now supports a separate structured employee-record layer for
learning ERP-style questions such as leave balance, carry-forward, and
compensatory leave.

### Storage boundary

- Policy documents and embeddings remain in Qdrant Cloud.
- Synthetic editable employee records are stored in
  `data/employee_records.json`.
- Employee records are not embedded into Qdrant.
- A future production integration can replace the JSON lookup with an
  authenticated ERP/database API without changing policy retrieval.

### New workflow and tools

- `rag/employee_data.py` loads, validates, saves, and looks up records by
  employee ID, customer ID, or email.
- `employee_case` resolves the record and calculates requested values from
  exact structured fields.
- The agent routes employee-specific questions to `employee_case` and asks for
  an identifier when one is missing.
- Streamlit provides a **Dynamic employee records** editor with save and
  download controls.

### Example

```text
Question: How much leave can EMP-001 carry forward?
→ lookup EMP-001
→ read current balance and carry-forward cap
→ calculate min(balance, cap)
→ return the structured result
```

Run the prototype test with:

```powershell
.\.venv\Scripts\python.exe scripts\test_employee_data.py
```

This prototype is for synthetic learning data. A production version must add
authentication, authorization, audit logging, and live ERP data freshness
checks before exposing personal employee information.
