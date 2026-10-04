# Week 10 implementation — CrewAI specialists, task handoffs, and MCP

## What is now in the app

The existing **MCP stdio** mode remains the single-agent baseline. The new **Week 10 Agent Race** mode runs that baseline beside a manager-led team. It is an extension to the HR-policy app; policy chunks remain in Qdrant and synthetic employee/transaction records remain in the existing JSON files.

The team has one manager and six role-specific specialist agents:

| Role | Responsibility | MCP access |
|---|---|---|
| Manager/orchestrator | Routes to relevant tasks and synthesizes after specialist outputs/review | CrewAI task orchestration; no direct employee-data tool |
| Policy research | Retrieves policy evidence and source metadata | `search_hr_policy` |
| Employee directory | Resolves one ID/email to a limited employee record | `get_employee_record` |
| Leave activity | Gets a person's transactions or searches matching employees and fetches their details | `get_employee_leave_transactions`, `find_employee_ids_by_leave_transactions`, `get_employee_details` |
| Record validator | Checks employee ID, policy ID, and numeric balance fields | No MCP data access; receives only the candidate record |
| Policy calculator | Requests the existing policy-backed summary/calculation | `get_employee_leave_summary` |
| Evidence review | Checks retrieved evidence/citations and employee-policy consistency before synthesis | No broad data access; reviews specialist results |

The current Streamlit and race-script team runner is CrewAI. A CrewAI manager selects applicable specialist roles; a deterministic router adds required safety stages, and CrewAI runs the selected tasks sequentially. To reduce Groq TPM pressure, specialist and evidence-review tasks do not automatically inherit every earlier narrative output; Python retains the complete structured specialist evidence for validation, while the final manager receives only the latest compact task summaries. Record validation follows employee lookup; evidence review audits the structured specialist results before the manager drafts the response. GPT-OSS uses low reasoning effort and a 512-token completion cap to reduce hidden-reasoning token use and preserve room for visible output. The previous custom Python A2A HTTP runner remains in `rag/week10_a2a.py` for reference, but it is no longer the primary team implementation.

## Legacy A2A protocol reference

The diagram below documents the previous custom A2A implementation only; it is not the active Week 10 race path.

```text
Streamlit host
  ├─ Single-agent baseline ─ MCP client ─ MCP server ─ policy + employee tools
  └─ Week 10 manager (Groq)
       ├─ A2A JSON-RPC message/send ─ policy research ─ MCP server
       ├─ A2A JSON-RPC message/send ─ employee directory ─ MCP server
       ├─ A2A JSON-RPC message/send ─ leave activity ─ MCP server
       ├─ A2A JSON-RPC message/send ─ record validator
       ├─ A2A JSON-RPC message/send ─ policy calculator ─ MCP server
       └─ A2A JSON-RPC message/send ─ evidence review
```

The active Week 10 path demonstrates CrewAI agent-to-agent handoff through selected task outputs; it does not claim this is the A2A standard protocol. Full evidence remains available to Python validation without being repeatedly copied into every model prompt. CrewAI is orchestration, MCP remains the tool/data protocol, and the single-agent MCP baseline remains unchanged. The retired custom A2A service code is retained only as a reference implementation.

CrewAI is a Python framework. Project-specific MCP adapters, deterministic policy checks, employee-record validation, and citation checks remain Python code; this is normal even in framework-based applications. This learning implementation is not production authorization for real employee data.

## Week 10 race and scoring

The frozen cases are in `data/week10_eval_cases.json`. The same question is run through both systems. `scripts/16_week10_agent_race.py` produces:

- `output/week10_agent_race.md` — aggregate table plus per-case answers and CrewAI specialist task statuses.
- `output/week10_agent_race.json` — raw per-system/per-case metrics and scores.
- `output/week10_a2a_tasks.jsonl` — legacy filename for the existing redacted Week 10 task trace.

Quality is a deterministic expected-fact/refusal/citation proxy, not a semantic LLM judge. Review the raw answers and case scores before declaring a winner. Mean latency, token totals, LLM-call counts, and estimated cost are reported. Actual cost is omitted unless input and output prices per million tokens are configured; token rates are inputs, not inferred from a changing provider price sheet.

To run interactively:

1. Activate the project's `.venv` and run `streamlit run app.py` as usual.
2. Select **Week 10 Agent Race**, enter one question, and press **Run**. Compare the baseline and CrewAI team columns and expand the task handoff trace.
3. Optionally enter token prices in the sidebar if you want estimated cost.
4. Open **Week 10 multi-agent race** in the sidebar and click **Run Week 10 comparison set**. Download the Markdown/JSON report and task log there.

Command-line alternative from the project root:

```powershell
py scripts/16_week10_agent_race.py
```

## Comparison and interpretation

The race uses the same fixed cases, corpus, MCP tools, configured generation model, and application state for both approaches. The single-agent route is the current Week 9 MCP host, not an artificially weakened baseline. The team adds manager planning and CrewAI task handoffs, so it may use more tokens and time. The saved report is the evidence for whether the extra coordination improved this task; no result is claimed before the race is run.

The deterministic scorer favors transparent checks: expected facts present, expected refusals honored, and citations resolvable to retrieved chunks. It can miss paraphrases and semantic errors. It does not replace human review or the Week 6 LLM-judge calibration.
