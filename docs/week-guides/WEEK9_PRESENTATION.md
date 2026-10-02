# Week 9 Presentation — MCP HR Policy Integration

## 1. Week 9 objective

This week converts the HR-policy capabilities into reusable Model Context Protocol (MCP) tools. An MCP client can discover the tools, call them with JSON-RPC, and receive structured HR data and policy citations without hard-coding every tool into the client.

The implementation has two transports:

- **STDIO** for the local Streamlit host and local MCP clients.
- **Streamable HTTP** for Postman, MCP Inspector, or another remote-capable MCP client.

## 2. What was implemented

### MCP server — `mcp_server.py`

- `FastMCP` server creation: [`mcp_server.py:17-22`](mcp_server.py:17)
- `search_hr_policy`: [`mcp_server.py:34-88`](mcp_server.py:34)
- `get_employee_record`: [`mcp_server.py:90-115`](mcp_server.py:90)
- `get_employee_leave_transactions`: [`mcp_server.py:117-129`](mcp_server.py:117)
- `get_employee_leave_summary`: [`mcp_server.py:131-153`](mcp_server.py:131)
- `compare_employee_leave_transactions`: [`mcp_server.py:155-177`](mcp_server.py:155)

The six available tools are:

| Tool | Function |
|---|---|
| `explain_mcp_demo` | Connectivity/demo check |
| `search_hr_policy` | Semantic, BM25, or hybrid policy search with filters |
| `get_employee_record` | Lookup by employee ID or email |
| `get_employee_leave_transactions` | Read filed, used, pending, and compensatory leave data |
| `get_employee_leave_summary` | Combine employee data with policy rules and calculate a summary |
| `compare_employee_leave_transactions` | Compare two employees’ leave transactions |

### MCP client/host — `rag/mcp_agent.py`

- MCP client imports: [`rag/mcp_agent.py:11-13`](rag/mcp_agent.py:11)
- Server process configuration: [`rag/mcp_agent.py:53-73`](rag/mcp_agent.py:53)
- STDIO session creation and tool discovery: [`rag/mcp_agent.py:73-80`](rag/mcp_agent.py:73)
- Host-level MCP routing and result handling: [`rag/mcp_agent.py:234-296`](rag/mcp_agent.py:234)

The host starts the server as a subprocess, initializes an MCP session, discovers tools, selects the appropriate tool for the question, and converts the structured result into the application response.

### Streamlit integration — `app.py`

- MCP mode is listed in the UI: [`app.py:165-166`](app.py:165)
- MCP request execution: [`app.py:674-682`](app.py:674)
- MCP error handling: [`app.py:719-726`](app.py:719)

The user selects **MCP stdio** in Streamlit and asks a policy or employee question. The UI displays the returned answer and citations.

### Tool discovery script — `scripts/13_week9_discover_tools.py`

- MCP STDIO client setup: [`scripts/13_week9_discover_tools.py:14-31`](scripts/13_week9_discover_tools.py:14)

This script proves that the client discovers the server’s tools instead of relying only on a manually maintained list.

### HTTP server — `mcp_http_server.py`

- Bearer-token middleware: [`mcp_http_server.py:21-56`](mcp_http_server.py:21)
- Streamable HTTP MCP application: [`mcp_http_server.py:58-68`](mcp_http_server.py:58)

The local HTTP endpoint is:

```text
http://127.0.0.1:8000/mcp
```

### Starting the HTTP server

The HTTP server is **not started automatically by Streamlit**. Start it from a separate PowerShell terminal at the project root:

```powershell
cd D:\Project\HR-Policy
$env:MCP_DEMO_TOKEN = "replace-with-a-random-32-character-token"
.\.venv\Scripts\python.exe mcp_http_server.py
```

Keep this terminal running. The server will be available at:

```text
http://127.0.0.1:8000/mcp
```

In another terminal, start the Streamlit UI if needed:

```powershell
cd D:\Project\HR-Policy
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Use the HTTP URL with Postman or MCP Inspector. Use **MCP stdio** in Streamlit for the local subprocess flow; that mode starts `mcp_server.py` through the MCP client and does not require the HTTP server.

The client sequence is `initialize` → `tools/list` → `tools/call`. HTTP requests use a bearer token and the returned `Mcp-Session-Id`.

## 3. Frameworks and packages used

- **Python 3.13** — application language.
- **MCP Python SDK / FastMCP** — server and client implementation (`requirements.txt:16`).
- **Streamlit** — user interface (`requirements.txt:1`).
- **LangChain** — RAG and generation integration (`requirements.txt:3-10`).
- **FastEmbed** — local embedding generation (`requirements.txt:2`).
- **Qdrant client** — vector retrieval (`requirements.txt:11`).
- **Uvicorn** — local Streamable HTTP server (`mcp_http_server.py:17`).
- **Python virtual environment** — dependencies are installed in `.venv` and tracked in `requirements.txt`; this is not a standard-library-only application.

## 4. MCP technique used

MCP uses JSON-RPC messages over a standard transport:

```text
MCP host/client
    -> initialize
    -> tools/list
    -> tools/call
MCP server
    -> validates arguments
    -> runs the selected HR tool
    -> returns structured data and citations
```

The LLM and application orchestration remain on the host side. The MCP server exposes controlled capabilities; it does not independently decide the final natural-language answer.

## 5. Event/request-driven functionality

The MCP interaction is request-driven: a client sends `tools/call`, the server validates and executes the request, and the client receives structured content. The host may then display the result, call another tool, or pass the evidence to an LLM for final wording.

The current tools are read/calculation tools. Any future write or edit tool must add explicit authorization, validation, confirmation, and audit logging.

## 6. How it was tested

- **MCP Inspector:** connect to `mcp_server.py` over STDIO, inspect `tools/list`, and call tools.
- **Postman/curl:** run `mcp_http_server.py`, call `/mcp`, then send JSON-RPC `initialize`, `tools/list`, and `tools/call` requests.
- **Streamlit:** select **MCP stdio** and test policy and employee questions.

Verified example:

```text
get_employee_leave_summary(employee002@example.com)
```

It returned the employee’s balance, annual entitlement, carry-forward, compensatory leave, policy ID, and resolvable policy chunk citations with `isError: false`.

## 7. Production direction

The current HTTP bearer-token gate is a local learning/demo security layer. A production deployment should use HTTPS, OAuth2/OIDC or signed JWTs, role-based employee access, secret-vault storage, audit logs, rate limits, timeouts, redaction, monitoring, and restricted tool permissions. Qdrant and the MCP server should also be placed behind controlled network access.

## Presentation summary

Week 9 implemented a reusable HR MCP server, a discovering MCP client, six policy/employee tools, STDIO and Streamable HTTP transports, JSON-RPC tool calls, session handling, structured citations, and a path for secure production deployment.
