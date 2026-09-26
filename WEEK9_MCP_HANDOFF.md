# Week 9 MCP Handoff

This document explains the MCP feature built in this HR Policy project: the
main concepts, how each request travels through the code, how to run it, what
has been verified, and what is still only a learning-demo security measure.
It complements the ordered exercises in [WEEK9_STEPS.md](WEEK9_STEPS.md).

## 1. MCP in this project

MCP (Model Context Protocol) is a standard way for an AI application to connect
to external capabilities. It defines how a client initializes a connection,
discovers tools and their input schemas, calls a tool, and receives its result.

MCP does not itself decide which tool the model should use, make a tool safe,
or authenticate a person. The host application still owns those decisions and
checks.

### The roles

| Role | Meaning here |
| --- | --- |
| Host | The application that handles the user interaction and runs agent/model logic. The command-line script or Streamlit app can be a host. |
| MCP client | A component inside the host that speaks MCP, initializes a session, discovers tools, and calls them. The client is not the server and is not the whole MCP system. |
| MCP server | `mcp_server.py`, which publishes named tools and runs their Python implementations. |
| Tool | One named operation with a description, an input schema, validation, and a result. |
| Transport | How MCP messages travel. The primary exercise uses stdio pipes; an optional local demo uses Streamable HTTP. |

The stdio client and server do **not** communicate over two network ports. The
client starts the server as a child process and sends protocol messages through
the child's standard input and reads responses from standard output. In HTTP
mode, the server listens on `127.0.0.1:8000` at `/mcp`.

## 2. Implemented request paths

### Stdio: command-line host

```text
Question
  -> scripts/14_week9_mcp_policy_agent.py (host entry point)
  -> rag/mcp_agent.py (host-side route, MCP client, safety and answer generation)
  -> MCP ClientSession over stdio
  -> mcp_server.py (MCP server)
  -> search_hr_policy tool
  -> scripts/15_week9_policy_search_worker.py
  -> existing rag.retrieve.search and local policy index
  <- structured search evidence with source IDs and text
  -> host safety checks -> Groq answer -> citation audit -> response
```

### Stdio: Streamlit host

The **MCP stdio** option in the Streamlit sidebar calls the same
`run_mcp_policy_agent` host flow. Streamlit is the host; it doesn't call the
server tool directly. Its MCP client starts the stdio server, and the UI then
shows the answer, citations, and retrieved chunks. The MCP question presets
fill the shared question field and run immediately, so select **MCP stdio**
before clicking an MCP preset.

### Streamable HTTP: optional local demo

```text
Inspector or other MCP client
  -> HTTP Authorization: Bearer <local demo token>
  -> mcp_http_server.py token gate
  -> mcp_server.py registered MCP tools
```

The HTTP entry point reuses the same FastMCP object and registered tools. It
does not replace the stdio entry point. The demo gate returns HTTP 401 if the
bearer token is absent or wrong.

## 3. What the host does for a policy question

`rag/mcp_agent.py` follows this sequence:

1. Reject an empty question and use the existing rule-based router to check
   that the question is a single-policy lookup. The current MCP example does
   not support every existing workflow, such as comparisons and employee
   calculations.
2. Start `mcp_server.py` as a child process using the same Python interpreter
   as the host. The host forwards only a limited set of environment variables
   needed by retrieval (including Qdrant settings), not the entire environment.
3. Initialize an MCP session and call `list_tools` to discover the live server
   tools. It checks that `search_hr_policy` exists.
4. Check that the call's arguments fit the discovered input schema, then call
   the discovered tool. The current host arguments are normalized question,
   `strategy="structure"`, `method="hybrid"`, and `top_k=5`.
5. Check for an MCP tool error and parse the structured result. A JSON-text
   fallback supports clients/servers that don't populate structured content.
6. Convert evidence into the application's `Hit` records.
7. Run `unsafe_hits` to reject instruction-like unsafe policy passages, then
   run `refusal_check` to block questions without sufficient relevant evidence.
8. If the evidence passes, send it with the question to Groq for answer
   generation. The model call is configured with a 60-second timeout and no
   retries.
9. Extract citations and call `policy_audit`. The host returns an answer only
   when that audit succeeds; otherwise it returns a refusal.
10. Close the MCP session and the child process through context managers.

Tool selection is currently rule-based in the host. MCP provides discovery and
the call protocol; it does not make an LLM choose a tool in this exercise.
The model stays in the host and generates the final answer there.

## 4. Tools on the server

### `explain_mcp_demo`

A harmless starter tool that returns a short fixed string. It was used to
learn registration, discovery, calling, and stdio lifecycle before connecting
the HR retrieval system.

### `search_hr_policy`

The policy retrieval capability exposed over MCP. Its inputs are:

| Argument | Required? | Validation / purpose |
| --- | --- | --- |
| `query` | Yes | Non-empty question to search for. |
| `policy_id` | No | Optional exact-policy filter; if provided, non-empty string. |
| `region` | No | Optional region filter; if provided, non-empty string. |
| `strategy` | No | `structure` or `recursive`; defaults to `structure`. |
| `method` | No | `semantic`, `bm25`, or `hybrid`; defaults to `semantic`. |
| `top_k` | No | Integer from 1 to 5; defaults to 5. Booleans are rejected even though Python treats them as integers. |

The server delegates actual retrieval to
`scripts/15_week9_policy_search_worker.py`. The worker uses existing
`rag.retrieve.search`, so MCP wraps/reuses the retriever rather than
reimplementing it. It returns result count and source metadata such as policy
ID, section, source file, chunk ID, scores, and matching policy text. Text is
needed by the host to ground the final answer; it is not copied into the
trajectory's evidence summary.

The worker runs separately so retrieval/native vector-store output cannot
corrupt the MCP server's stdout protocol stream. The worker writes one JSON
payload to stdout; diagnostics belong on stderr. It uses `ensure_ascii=True`
to safely transfer unusual PDF characters.

### `get_employee_record`

A second, read-only capability used to demonstrate adding a tool without
changing the generic MCP discovery client or the policy agent's routing.
It requires one explicit identifier in `EMP-001` format, looks up the existing
synthetic record, and returns a limited set of fields. It omits name, email,
and other identifiers.

**Entering an employee ID is not authorization.** This project has no verified
caller identity or per-employee access-control check. The records are synthetic
and this tool must not be connected to real employee data as-is.

## 5. Safety, evidence, and trajectory logging

MCP standardizes tool exchange; it does not make tool output trustworthy. The
host treats retrieved policy text as untrusted input and keeps the existing
application checks in charge of answering:

- Instruction-like unsafe passages trigger a refusal.
- The refusal/relevance gate can stop unsupported questions before generation.
- Generated citations are checked against retrieved evidence by `policy_audit`.
- Tool errors are raised and reported to the host rather than used as policy
  evidence.

The Week 9 MCP wrapper writes records through the existing append-only Week 8
logger, `rag.trajectory.write_trajectory`, to
`output/agent_trajectories.jsonl`. MCP-specific steps record the tool name,
validated arguments, status, result count, policy/chunk source IDs, and stop
reason. They omit retrieved policy chunk text and environment values. Failures
record an error status and exception class, not the full exception contents.

The shared trajectory logger also stores the query and final answer using its
existing redaction. That redaction covers common employee IDs when explicitly
labelled and email addresses; it is not a general-purpose privacy filter.
Avoid sending sensitive user content to this learning app and review logging
before adapting it to production.

## 6. Streamlit controls and MCP presets

In `app.py`, choose **MCP stdio** from the Mode radio control. The mode:

- supports single-policy questions only;
- calls the existing MCP-backed host function;
- uses structure chunking, hybrid retrieval, and five results;
- disables the normal chunking, method, reranking, region, and top-k controls
  because those controls are not currently passed into the MCP agent;
- displays status, number of retrieved chunks, MCP tool name, answer,
  citations, and retrieved chunks.

The **Preset: MCP stdio** tab contains:

1. What is the annual leave policy for ACME in the United States?
2. How much annual leave does a confirmed Azure employee with at least one
   year of service receive?
3. How many casual and privilege leaves are provided in the SoftSuave
   handbook?
4. What is Northstar's annual leave entitlement and accrual rate?
5. What is the carry-forward limit in the ACME leave policy?

Select **MCP stdio** first. Clicking a preset sets the question and submits it
immediately. Presets are examples, not guaranteed answers; inspect the
organization/region in each answer because policies can differ.

## 7. Run instructions

### Streamlit setup on Windows PowerShell

The `.venv` folder was not present at the last check. Create the virtual
environment once, then install requirements into it:

```powershell
cd C:\Test\HR-Policy
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the virtual environment already exists and activates successfully, skip the
creation and installation commands. Running Streamlit as `python -m streamlit`
avoids depending on the `streamlit` launcher being available on PATH. The
globally selected `py` interpreter previously reported Streamlit 1.64.0; the
Streamlit app should use whichever interpreter has the project's dependencies
installed.

### Discover tools over stdio

From the project root, using the active environment:

```powershell
python scripts\13_week9_discover_tools.py
```

### Ask a policy question through the MCP host

```powershell
python scripts\14_week9_mcp_policy_agent.py "What is the annual leave policy?"
```

The same query can be asked in Streamlit with **MCP stdio** selected.

### Optional Streamable HTTP demo

Use two PowerShell terminals from the project root.

Terminal A:

```powershell
$env:MCP_DEMO_TOKEN = [guid]::NewGuid().ToString("N")
Write-Host $env:MCP_DEMO_TOKEN
python mcp_http_server.py
```

Copy the printed random demo token to Terminal B. Check denial without a
token, then list/call tools using the separately installed/pinned Inspector
CLI (Node 22.21.0 and Inspector 2.8.0 were used for the recorded demonstration):

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/mcp -H "Content-Type: application/json" --data "{}"
$token = Read-Host "Paste the demo token from Terminal A"
npx --yes @modelcontextprotocol/inspector@2.8.0 --cli --transport http --server-url http://127.0.0.1:8000/mcp --header "Authorization: Bearer $token" --method tools/list
npx --yes @modelcontextprotocol/inspector@2.8.0 --cli --transport http --server-url http://127.0.0.1:8000/mcp --header "Authorization: Bearer $token" --method tools/call --tool-name get_employee_record --tool-arg employee_id=EMP-001 --format json
```

The request without a token should receive HTTP 401. Stop Terminal A with
Ctrl+C. This is a local shared-token gate, not OAuth, not a real identity
provider, and not per-employee authorization. It binds only to loopback and is
for synthetic data and learning.

## 8. Code map

| File | Responsibility |
| --- | --- |
| `mcp_server.py` | Registers the tools and runs the MCP server over stdio. |
| `mcp_http_server.py` | Optional Streamable HTTP entry point with a local demo bearer-token gate; reuses the same registered tools. |
| `rag/mcp_agent.py` | Host-side route, MCP client lifecycle, discovery/schema checks, tool call, evidence checks, answer generation, citation audit, and trajectory write. |
| `scripts/13_week9_discover_tools.py` | Starts the stdio server, initializes an MCP client session, lists tools/schemas, and closes the session/server. |
| `scripts/14_week9_mcp_policy_agent.py` | CLI host entry point for one single-policy MCP question. |
| `scripts/15_week9_policy_search_worker.py` | Internal child process that invokes the existing retriever and serializes evidence as JSON. |
| `rag/employee_data.py` | Existing structured synthetic-record lookup used by `get_employee_record`. |
| `rag/trajectory.py` | Existing append-only trajectory writer used by the MCP host. |
| `app.py` | Streamlit UI; adds the MCP stdio mode and the MCP question presets. |
| `WEEK9_STEPS.md` | Ordered learning guide and recorded Step 8/HTTP demonstrations. |

## 9. What has been verified and what has not

Verified during this work:

- Python MCP dependencies were installed; the observed SDK version was 1.30.0.
- The stdio discovery script listed `explain_mcp_demo`, `search_hr_policy`,
  and `get_employee_record` with their schemas.
- The MCP client called `get_employee_record` and got the synthetic `EMP-001`
  record.
- The MCP policy CLI completed a policy question, retrieved five results,
  passed its safety checks, and returned a cited/audited answer.
- MCP Inspector CLI 2.8.0 independently discovered and called the stdio
  employee-record tool.
- The optional local HTTP gate returned 401 without a token; Inspector with
  the token discovered tools and called the synthetic employee tool.
- `py_compile` succeeded for the MCP agent and the updated Streamlit app.

Not yet confirmed end-to-end:

- The Streamlit app has not yet been launched and exercised through its new
  **MCP stdio** mode after those UI changes. Syntax compilation alone does not
  prove the UI flow works.
- The trajectory-logging changes have not been checked by rerunning the final
  Groq policy-agent call and inspecting the newest JSONL record.
- Step 6 demonstrates an ID-scoped synthetic lookup, but real identity and
  per-record authorization are not implemented.

## 10. Learning summary

The main advantages demonstrated are a discoverable tool contract, a reusable
server capability, and the ability for a different MCP client (Inspector) to
use the same server without copying the host agent's code. The main boundary
to remember is that MCP is a protocol, not a security policy: the host still
chooses and validates calls, the server validates inputs, and returned text
must still be checked before the model relies on it.
