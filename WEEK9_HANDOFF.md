# Week 9 Handoff — MCP, Multi-Agent Connectivity, and A2A

## Purpose

Week 9 moves the HR-policy agent from manually wired Python tools to the Model
Context Protocol (MCP). MCP is the standard connection layer between an AI
host, an MCP client, and external tools or data servers.

MCP improves reuse and integration. It does not make the model smarter by
itself; it gives the agent a discoverable, replaceable tool interface.

## Current application before Week 9

The application currently has:

- Streamlit Ask, Retrieve, Rerank, Compare, and Agent + Workflow modes.
- Semantic, BM25, and hybrid retrieval with reranking.
- Recursive and structure-aware chunking.
- Qdrant policy collections.
- JSON employee records with policy-first calculations.
- Fixed workflows and a dynamic agent loop.
- Week 8 trajectory logging, prompt-injection detection, and route evaluation.

The remaining limitation is that the agent still reaches its tools through
application-owned Python imports. Another agent cannot discover or reuse those
tools through a standard protocol.

## Week 9 target architecture

```text
Streamlit / Agent host
        ↓
Generic MCP client
        ↓ discovers tools and schemas
MCP server
  ├── search_hr_policy
  ├── get_employee_record
  └── calculate_leave_entitlement
```

The AI reasoning runs in the host application. The MCP server exposes tools and
data; it does not run the AI reasoning.

## MCP roles

- **Host:** our Streamlit/agent application and the model that makes decisions.
- **Client:** the reusable MCP connector inside the host.
- **Server:** a process or service exposing tools, resources, and prompts.

The agent should discover available tools through the client rather than
hard-coding HR tool imports.

## Transport plan

### Local stdio — first implementation

Use stdio for the local demonstration:

```text
Agent process
  stdin  → MCP JSON-RPC requests
  stdout ← MCP JSON-RPC responses
```

This requires no port or network configuration and is appropriate for the
first working version.

### Streamable HTTP — reusable integration

Support Streamable HTTP through the same client abstraction:

```text
MCP client → HTTPS/HTTP → approved MCP server
```

The agent logic and tool schemas should not change when switching transports.
Only the server connection configuration changes.

Example configuration concept:

```json
{
  "name": "local_hr",
  "transport": "stdio",
  "command": "python",
  "args": ["mcp_server.py"]
}
```

```json
{
  "name": "remote_hris",
  "transport": "streamable-http",
  "url": "https://example.internal/mcp"
}
```

## Authentication and authorization

Local stdio relies mainly on process and operating-system permissions. HTTP
connections require stronger controls:

- Bearer token or API key for development.
- OAuth2 or service credentials for production.
- HTTPS for network transport.
- Approved-server allowlist.
- Tool-level authorization and input validation.
- Read-only employee tools by default.

Authentication should be layered:

```text
Connection authentication
  → Server allowlist
  → Tool authorization
  → Input/output validation
  → Tool execution
```

Secrets belong in `.env` or a secret manager, never in source code or traces.

## Implementation sequence

1. Build a small local MCP server using FastMCP or the MCP Python SDK.
2. Expose `search_hr_policy` as the first real capability.
3. Add `get_employee_record` and `calculate_leave_entitlement`.
4. Build a generic MCP client that performs initialization and tool discovery.
5. Let the existing agent select discovered tools from their names and schemas.
6. Pass tool results through the existing policy audit and trajectory logger.
7. Add a second tool without changing the agent’s routing code.
8. Add optional Streamable HTTP configuration and bearer-token validation.
9. Demonstrate that another MCP client could call our HR-policy server.

## Expected tool flow

```text
Question: “How much can EMP-001 carry forward?”
  ↓
Discover tools
  ↓
get_employee_record(EMP-001)
  ↓
search_hr_policy(policy_id=AZURE-HR-2026)
  ↓
calculate_leave_entitlement(balance=7, cap=5)
  ↓
Policy audit and cited answer
```

The existing safety rules remain active. The MCP layer must not bypass employee
identifier checks, policy-evidence validation, refusal behavior, or trajectory
logging.

## Week 9 completion criteria

Week 9 is complete when:

1. The agent discovers at least one tool through MCP instead of importing it directly.
2. The local HR-policy server works over stdio.
3. A second tool can be added without changing agent decision logic.
4. The same client design can connect to an approved HTTP MCP server.
5. Authentication and tool authorization are demonstrated for HTTP.
6. A tool call appears in the trajectory with input, output, status, and stop reason.
7. Another MCP client could call our HR-policy server.

## Presentation explanation

MCP is a standard socket for AI tools. The model remains in the host. The client
discovers tools. The server performs the approved operation. This separation
lets another agent reuse the HR-policy capability without copying our internal
Python implementation.

