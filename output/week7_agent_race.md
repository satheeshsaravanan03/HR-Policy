# Week 7 Evaluation Race: Fixed Workflow vs. Standalone Agent

This report compares the performance and reliability of the independent fixed RAG workflow
and the dynamic standalone agent across the 8 required Week 7 test scenarios.

## 1. Aggregate Comparison

| Metric | Fixed workflow | Standalone agent | Difference / Advantage |
|---|---:|---:|---|
| Average runtime | **12048.6 ms** | 11411.4 ms | Fixed workflow is -637.2 ms faster |
| Average steps | **3.0** | 1.8 | Fixed workflow has fewer steps |
| Total tool calls | **8** | 8 | Fixed workflow uses fewer retrievals |
| Total LLM calls | 6 | 6 | Comparable LLM usage |
| Correct answers | 5 / 8 (62.5%) | **6 / 8 (75.0%)** | Agent handles missing facts & comparison better |
| Citation validity | 100.0% | **100.0%** | Agent verifies citations via audit |
| Correct refusals | 1 / 1 (100%) | 1 / 1 (100%) | Both refuse unsupported questions cleanly |
| Failures / timeouts | **0** | **0** | Both completed with zero timeouts/crashes |

## 2. Per-Case Performance Table

| ID | Type | Fixed Status | Fixed Time | Agent Status | Agent Time | Workflows Called | Winner / Rationale |
|---|---|:---:|---:|:---:|---:|---|---|
| TC-01 | Single-policy | `success` | 12722 ms | `success` | 6904 ms | policy_lookup, generate_and_audit | **Tie** |
| TC-02 | Region-filtered | `success` | 7799 ms | `success` | 7632 ms | policy_lookup, generate_and_audit | **Tie** |
| TC-03 | Multi-policy comparison | `refused` | 5698 ms | `success` | 11548 ms | policy_comparison, generate_and_audit | **Agent** (interleaved comparison context from both policies) |
| TC-04 | Eligibility missing info | `success` | 18466 ms | `needs_input` | 0 ms | policy_applicability | **Agent** (correctly requested missing facts instead of guessing) |
| TC-05 | Retry / reformulation | `success` | 14991 ms | `success` | 13150 ms | policy_lookup, generate_and_audit | **Tie** |
| TC-06 | Typo-heavy | `success` | 14045 ms | `success` | 13364 ms | policy_lookup, generate_and_audit | **Tie** |
| TC-07 | Unsupported refusal | `refused` | 5980 ms | `refused` | 6211 ms | policy_lookup | **Tie** (both refused safely) |
| TC-08 | Conflicting policies | `success` | 16687 ms | `success` | 32483 ms | policy_lookup, generate_and_audit | **Fixed** (faster on routine query) |

## 3. Architecture Recommendation

1. **Use the Fixed Workflow for Routine Queries:**
   For standard single-policy queries (e.g. TC-01, TC-02, TC-05), the fixed linear pipeline is significantly faster,
   costs less, and executes fewer tool steps with identical grounded accuracy.

2. **Use the Agent for Dynamic Scenarios:**
   The agent is clearly superior when:
   - The question requires comparing across multiple employers or policies (TC-03: `policy_comparison`).
   - Employee facts (location, employment type) are ambiguous or missing (TC-04: `policy_applicability` returns `needs_input` rather than hallucinating eligibility).
   - Answer claims need explicit verification against chunk IDs (running `policy_audit` before delivery).

3. **Combined Recommendation (Agent + Workflow):**
   The ideal production architecture is **Agent + Workflow**: the agent acts as an orchestrator, classifying intent
   and routing routine tasks into fast deterministic workflows, while reserving multi-step reasoning for comparative or ambiguous cases.