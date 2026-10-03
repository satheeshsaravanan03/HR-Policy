# Week 10 Presentation - CrewAI Multi-Agent HR Assistant

## What we built

Week 10 adds a CrewAI manager-and-specialist team to the existing HR policy app. It runs beside the existing single-agent MCP baseline so both approaches can be compared on the same question or evaluation set.

The team combines three pieces:

- **CrewAI** defines agents, assigns tasks, passes task outputs as context, and runs the team.
- **MCP** provides the existing HR policy and employee-data tools. CrewAI does not replace the MCP server.
- **Python** connects CrewAI tools to MCP operations and applies deterministic routing, validation, citation checks, and safe response handling.

This is a learning implementation using synthetic employee records. It is not production authorization for real HR data.

## Tools and libraries

| Component | Use in this app |
|---|---|
| CrewAI (`crewai[litellm]==0.203.2`) | Agent, task, and crew orchestration; LiteLLM integration enables Groq as the model provider. |
| Groq LLM | Language model used by the CrewAI agents for planning, tool use, and response generation. |
| MCP Python SDK / FastMCP | Exposes policy and employee operations as MCP tools. The CrewAI adapter calls those tools through existing Python operations. |
| Streamlit | UI for asking a question, comparing single-agent and CrewAI responses, and inspecting the run steps. |
| Qdrant and local JSON data | Policy chunks are retrieved from the configured vector store; demo employee and leave-transaction data are local JSON-backed records. |

CrewAI is installed from `requirements.txt`; it is not copied into the project. The pinned dependency is `crewai[litellm]==0.203.2`.

## How many agents are there?

There are **six specialist role definitions**; a question uses only the roles the planner selects (with Python safety routing adding any required roles):

1. **Policy Research** - retrieves policy passages and source metadata using `search_hr_policy`.
2. **Employee Directory** - resolves an employee ID or email using `get_employee_record`.
3. **Leave Activity** - searches leave activity or employee criteria, then retrieves matching details.
4. **Record Validator** - checks returned employee records and numeric fields before relying on them.
5. **Policy Calculator** - obtains a policy-backed employee leave summary using the existing MCP operation.
6. **Evidence Review** - checks policy identity, evidence, and citations before policy claims are answered.

The implementation also creates a planning-manager agent and a final response-manager agent. In presentation terms, describe this as **one manager role plus selected specialists**; technically, planning and final synthesis are separate CrewAI `Agent` instances. The number of instantiated agents therefore depends on the question and its selected roles - not every request uses all six specialists.

## How one question flows through the system

```text
User question in Streamlit
          |
          v
CrewAI planning manager selects specialist roles
          |
          v
Python safety routing adds required validation/review roles
          |
          v
Selected specialist agents run their assigned tasks
          |
          +--> MCP-backed policy / employee tools
          |
          v
Task outputs are passed forward as CrewAI context
          |
          v
Evidence reviewer checks policy evidence when needed
          |
          v
Response manager synthesizes the answer
          |
          v
Python applies citation / record safety checks; Streamlit shows answer and steps
```

The active CrewAI path uses `Process.sequential`: tasks run one after another, and later tasks receive earlier outputs through `Task.context`. It is **not parallel specialist execution**. The previous custom A2A implementation remains in the repository as reference code, but this CrewAI path uses CrewAI task/context handoffs; it does not claim to use the A2A protocol for these handoffs.

For employee-list results, the application also has a deterministic Python formatting path after the validated structured records are returned. That avoids asking the LLM to recalculate or invent record values.

## How it is constructed in code

The central implementation is `rag/week10_crewai.py`:

- `run_week10_crewai_team(...)` is the entry point called by the Streamlit comparison mode.
- CrewAI `LLM(...)` configures the Groq-backed model.
- `Agent(...)` creates the planner, selected specialists, reviewer, and final response manager.
- `Task(...)` describes each assignment and its expected output.
- `MCPDispatchTool` adapts a specialist task to the existing Python/MCP-backed operation.
- `Crew(...)` gathers agents and tasks with `Process.sequential`.
- `crew.kickoff()` executes the tasks and returns the final manager output.

The specialist operations are implemented in `rag/week10_a2a.py` under `_agent_work(...)`; despite that file's historical name, it contains the Python dispatch to the existing MCP tools used by this CrewAI path.

## How to install and run

From the project root in PowerShell:

```powershell
# Install project dependencies into the existing virtual environment
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# Start Streamlit
.\.venv\Scripts\python.exe -m streamlit run app.py
```

In Streamlit, choose **Week 10 Agent Race**, enter a question, and run the comparison. To run the saved comparison set from a terminal:

```powershell
.\.venv\Scripts\python.exe scripts\16_week10_agent_race.py
```

The Groq API key must be present in the local environment or `.env`; never paste or commit it. Cost is shown only when token prices are configured. A CrewAI run makes multiple LLM calls, so provider rate limits can interrupt the run even when MCP retrieval succeeds.

## What to show during the presentation

1. Ask a policy question and show the single-agent baseline beside the CrewAI team.
2. Expand the CrewAI steps to show the planner, selected specialists, MCP tool calls, evidence review, and final response.
3. Ask an employee-record question and show that structured record data is validated and then formatted.
4. Ask an unsupported policy question and show that the system refuses unsupported claims.
5. Explain the trade-off honestly: CrewAI makes role/task handoffs explicit, but it adds LLM calls, latency, and rate-limit exposure. Keep the multi-agent approach only where measured quality or task clarity justifies that cost.

## Source files

- Dependency: `requirements.txt`
- CrewAI agents, tasks, tool adapter, and execution: `rag/week10_crewai.py`
- Existing specialist/MCP-backed operations and role definitions: `rag/week10_a2a.py`
- MCP tools: `mcp_server.py`
- Streamlit Week 10 UI: `app.py`
- Saved evaluation set: `data/week10_eval_cases.json`
- Evaluation runner: `scripts/16_week10_agent_race.py`
- Evaluation report: `output/week10_agent_race.md`
