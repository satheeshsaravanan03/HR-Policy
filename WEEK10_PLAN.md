# Week 10 implementation — manager, specialists, A2A and MCP

## What is now in the app

The existing **MCP stdio** mode remains the single-agent baseline. The new **Week 10 Agent Race** mode runs that baseline beside a manager-led team. It is an extension to the HR-policy app; policy chunks remain in Qdrant and synthetic employee/transaction records remain in the existing JSON files.

The team has one manager and six role-specific specialist agents:

| Role | Responsibility | MCP access |
|---|---|---|
| Manager/orchestrator | Uses the configured Groq model to choose specialist skills, coordinates tasks, and synthesizes only after review | Calls specialist agents over A2A; no direct employee-data tool |
| Policy research | Retrieves policy evidence and source metadata | `search_hr_policy` |
| Employee directory | Resolves one ID/email to a limited employee record | `get_employee_record` |
| Leave activity | Gets a person's transactions or searches matching employees and fetches their details | `get_employee_leave_transactions`, `find_employee_ids_by_leave_transactions`, `get_employee_details` |
| Record validator | Checks employee ID, policy ID, and numeric balance fields | No MCP data access; receives only the candidate record |
| Policy calculator | Requests the existing policy-backed summary/calculation | `get_employee_leave_summary` |
| Evidence review | Checks retrieved evidence/citations and employee-policy consistency before synthesis | No broad data access; reviews specialist results |

The LLM manager plans which workers are relevant. Independent retrieval roles run concurrently; record validation and evidence review run after their required inputs exist. A calculation is skipped if record validation fails. The team has a fixed specialist limit and HTTP task timeout; it does not repeat indefinitely.

## Protocol boundaries

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

The local A2A HTTP service is started automatically in the Streamlit process on `127.0.0.1:8120` (override with `WEEK10_A2A_HOST`/`WEEK10_A2A_PORT`). It publishes the team and specialist AgentCards at `/.well-known/agent-card.json` and `/.well-known/agent-card/{role}.json`; `/tasks` exposes in-memory task status. Specialists accept JSON-RPC `message/send`, return a task ID and completed/failed state, and use MCP STDIO sessions to discover/call tools. A redacted task summary is appended to `output/week10_a2a_tasks.jsonl`.

This is a small local A2A JSON-RPC learning transport implemented with Python's standard library so no new framework dependency is required. It implements the message/task patterns used in this demo; it is not a production deployment or a substitute for deployment-grade authentication, TLS, tenant-aware authorization, durable task storage, streaming, or the full A2A SDK feature set. The A2A service binds to loopback by default and should not be exposed to the network as-is.

## Week 10 race and scoring

The frozen cases are in `data/week10_eval_cases.json`. The same question is run through both systems. `scripts/16_week10_agent_race.py` produces:

- `output/week10_agent_race.md` — aggregate table plus per-case answers and team task IDs/statuses.
- `output/week10_agent_race.json` — raw per-system/per-case metrics and scores.
- `output/week10_a2a_tasks.jsonl` — task execution summaries.

Quality is a deterministic expected-fact/refusal/citation proxy, not a semantic LLM judge. Review the raw answers and case scores before declaring a winner. Mean latency, token totals, LLM-call counts, and estimated cost are reported. Actual cost is omitted unless input and output prices per million tokens are configured; token rates are inputs, not inferred from a changing provider price sheet.

To run interactively:

1. Activate the project's `.venv` and run `streamlit run app.py` as usual.
2. Select **Week 10 Agent Race**, enter one question, and press **Run**. Compare the baseline and team columns and expand the AgentCards/task trace.
3. Optionally enter token prices in the sidebar if you want estimated cost.
4. Open **Week 10 multi-agent race** in the sidebar and click **Run Week 10 comparison set**. Download the Markdown/JSON report and task log there.

Command-line alternative from the project root:

```powershell
py scripts/16_week10_agent_race.py
```

## Comparison and interpretation

The race uses the same fixed five cases, corpus, MCP tools, configured generation model, and application state for both approaches. The single-agent route is the current Week 9 MCP host, not an artificially weakened baseline. The team adds manager planning and A2A hand-offs, so it is expected to use more tokens and time. The saved report is the evidence for whether the extra coordination improved this task; no result is claimed before the race is run.

The deterministic scorer favors transparent checks: expected facts present, expected refusals honored, and citations resolvable to retrieved chunks. It can miss paraphrases and semantic errors. It does not replace human review or the Week 6 LLM-judge calibration.
