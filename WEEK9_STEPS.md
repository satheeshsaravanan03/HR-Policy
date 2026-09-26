# Week 9 Learning Steps — Model Context Protocol (MCP)

## How to use this guide

Work through one step at a time. At the end of each step, review the concepts,
code, and checks before moving on. The goal is to understand why each piece is
needed, not just to make a demo run.

The main path uses a local MCP server over stdio. Streamable HTTP and
authentication are advanced follow-up topics; they are not required to learn
the core MCP flow.

## Week 9 goal

Connect the existing HR-policy agent to tools exposed by an MCP server. The
agent should discover the tools and their input schemas through an MCP client,
then call an appropriate tool and handle its result safely.

```text
User → Host application and agent → MCP client → MCP server → HR capability
                         ← tool result ←
```

- **Host:** application that runs the agent and model.
- **MCP client:** connects to a server, negotiates the protocol, discovers
  tools, and sends tool calls.
- **MCP server:** advertises and runs approved capabilities.
- **Tool:** a callable capability with a name, description, input schema, and
  result.
- **Transport:** how client and server exchange protocol messages. This guide
  starts with local stdio.

MCP standardizes the connection and discovery. It does not make the model more
capable or make tool calls safe automatically. The host remains responsible for
agent decisions, access rules, validation, and answer quality.

## Step 1 — Understand the baseline and MCP boundary

**Topic:** host, client, server, tool, transport, and the difference between
ordinary Python function calls and MCP calls.

**Activity:** inspect the current agent and its existing tool/workflow wiring.
Draw or write down where the host ends and where a future MCP server begins.
Do not change application code in this step.

**Why this step matters:** a clear boundary prevents moving model reasoning or
policy decisions into the server by accident. It also gives us a baseline to
compare against after the MCP integration.

**Review questions:**

1. Which code currently decides what action to take?
2. Which code performs retrieval or employee calculations?
3. What should the MCP server expose, and what should remain in the host?
4. What information does a client need in order to call a tool?

**Done when:** you can explain the roles above in your own words and sketch the
request/result path. Review the explanation before starting Step 2.

## Step 2 — Create the smallest MCP server

**Topic:** MCP SDK, server lifecycle, tool registration, tool metadata, and
stdio transport.

**Activity:** create a small local server with one harmless demonstration tool.
Run it as a child process using stdio. Keep diagnostic logs on stderr; stdout is
reserved for MCP protocol messages.

**In this project:** `mcp_server.py` registers `explain_mcp_demo`. Install the
new dependency with `python -m pip install -r requirements.txt`, then start the
server with `python mcp_server.py`. When run directly, it waits for an MCP
client on stdin/stdout; it is not expected to print a normal message or open a
browser. Stop it with Ctrl+C. In the next step we will connect an MCP client
and verify discovery and invocation.

**Why this step matters:** isolates server setup and transport from HR retrieval
logic, making protocol and startup errors easier to understand.

**Review questions:** Can the server start and stop cleanly? Where do protocol
messages go? What does its tool name and description communicate to a client?

**Done when:** the SDK dependency is installed, you can explain the server
startup and stdio behavior, and you have reviewed the tool registration. The
live client connection and tool listing are verified in Step 4, where we build
the client rather than introducing another dependency in this first server
exercise.

## Step 3 — Expose HR policy search as an MCP tool

**Topic:** tool schemas, required/optional arguments, validation, result shape,
and errors.

**Activity:** `mcp_server.py` now wraps the existing policy-search capability
as `search_hr_policy`. It accepts a query, optional policy ID and region,
chunking strategy, retrieval method, and a bounded result count. The wrapper
delegates to `scripts/15_week9_policy_search_worker.py`, which calls the
existing `rag.retrieve.search` implementation and returns evidence with source
information for citations. The worker keeps retrieval output separate from
the MCP server's stdout protocol stream.

**Why this step matters:** demonstrates reuse of existing functionality while
giving another MCP client a standard way to discover and call it.

**Review questions:** Are inputs constrained and described? Are source IDs or
citations preserved? What happens for empty or malformed input?

**Done when:** review the tool's input schema, validation, call into the
existing retriever, and citation fields. Live invocation will be done in Step 4
after we build the MCP client. A working local policy index and its embedding
dependencies are needed for a real search result.

## Step 4 — Build a reusable MCP client and discover tools

**Topic:** client initialization, capability negotiation, tool listing, tool
schemas, session lifecycle, and cleanup.

**Activity:** `scripts/13_week9_discover_tools.py` starts the local server as a
child process, initializes an MCP session, and prints the discovered tools,
descriptions, and input schemas. Run it from the project root with
`python scripts/13_week9_discover_tools.py`. The script uses the same Python
interpreter to launch the server, and context managers close the session and
child process when discovery finishes.

**Why this step matters:** discovery is the main architectural change: the host
can learn the available capabilities from the server rather than importing
each server implementation directly.

**Review questions:** Does the client inspect tool descriptions and schemas?
Does it close the process/session on success and failure? What if the server is
unavailable?

**Done when:** run the discovery script and confirm both
`explain_mcp_demo` and `search_hr_policy` appear with their schemas. Review
initialization, the discovered-versus-hard-coded distinction, error handling,
and cleanup before moving on.

## Step 5 — Call a discovered tool from the agent

**Topic:** host-side tool selection, checking discovered schemas, sending a
tool request, and passing validated results into answer generation.

**Activity:** `rag/mcp_agent.py` uses the existing rule-based workflow router
to select the policy-search capability. It verifies that
`search_hr_policy` is present in the server's live tool list, calls it through
the MCP client, converts returned evidence into the app's retrieval records,
then applies the existing prompt-injection/refusal checks and cited answer
audit. Groq generation has a 60-second timeout. Run it with
`python scripts/14_week9_mcp_policy_agent.py "What is the annual leave policy?"`.
The same host flow is also available from the Streamlit UI: run
`streamlit run app.py`, select **MCP stdio**, and submit a single-policy
question. In that mode Streamlit is the host and starts the local MCP server
through the client.
This step demonstrates the current agent choosing a discovered capability;
the Week 7 router is rule-based, so an LLM does not choose tools in this
exercise. The model remains in the host for answer generation.

**Why this step matters:** completes the basic MCP loop and shows that MCP
connectivity and agent reasoning are separate concerns.

**Review questions:** Does the host confirm the tool was discovered before
calling it? Are arguments constrained by both the tool schema and server-side
validation? Is returned text treated as untrusted and checked by existing
safety/evidence safeguards?

**Done when:** a single-policy question triggers a discovered MCP call and the
answer still follows the existing refusal and citation checks. Review the full
path, including why routing here is rule-based, not only the final answer.

## Step 6 — Add a second tool without changing routing logic

**Topic:** extensibility, tool contracts, least privilege, and separating
read-only data access from policy calculations.

**Activity:** `mcp_server.py` exposes a read-only `get_employee_record` tool.
It requires one explicit ID in `EMP-001` format, returns only a limited set of
synthetic demo fields, and rejects empty, malformed, or unknown IDs. This
project currently has no authenticated caller identity or per-employee access
control, so ID validation is not real authorization. Use synthetic data only;
add an identity/authorization check before connecting real employee records.
Leave calculation remains separate in the existing host workflow.

**Why this step matters:** proves that discovery is useful beyond a single demo
and makes the server's capabilities easier to reuse and maintain.

**Review questions:** Did adding the tool require edits to generic discovery or
routing? Is access scoped to an explicitly requested employee? Are mutable or
sensitive operations excluded?

**Done when:** the new tool appears through discovery and its input is
validated and scoped to one synthetic record without adding a tool-specific
branch to generic client code. For real employee data, a verified caller
identity and per-record authorization check are also required.

## Step 7 — Preserve safety, citations, and trajectory records

**Topic:** trust boundaries, input/output validation, authorization, audit
logging, failures, and stop reasons.

**Activity:** `rag/mcp_agent.py` routes MCP results through the existing unsafe
document checks, refusal gate, and citation audit, then writes an append-only
record using the Week 8 trajectory logger. It records the tool name, validated
arguments, status, policy/chunk identifiers, and stop reason. Retrieved text
and environment values are omitted. The normal trajectory record still stores
the query and answer with the project's existing redaction; review that policy
before logging sensitive user content in a real deployment.

**Why this step matters:** MCP provides a standard interface, not a security
boundary by itself. The existing policy-first behavior must continue to guard
the answer.

**Review questions:** Does missing policy evidence block a calculation? Are
unauthorized requests refused? Can failures be diagnosed without exposing
sensitive values?

**Done when:** successful and failed MCP requests are represented in the
trajectory, and a successful answer still passes safety and citation checks.

## Step 8 — Demonstrate interoperability with another MCP client

**Topic:** protocol interoperability, server/client independence, and
repeatable setup instructions.

**Activity completed:** connected the independent [MCP Inspector CLI](https://github.com/modelcontextprotocol/inspector)
to the local stdio server from Windows PowerShell. The environment has Node
22.21.0; Inspector 2.8.0 was pinned for repeatability. From the project root:

```powershell
npx --yes @modelcontextprotocol/inspector@2.8.0 --cli py mcp_server.py --method tools/list
npx --yes @modelcontextprotocol/inspector@2.8.0 --cli py mcp_server.py --method tools/call --tool-name get_employee_record --tool-arg employee_id=EMP-001 --format json
```

The first command discovered all three tools. The second called
`get_employee_record` and returned the synthetic `EMP-001` record. The MCP
server and its tool did not need host-agent code. This is a developer inspection
client; the employee tool still does not authenticate callers, so use only
the repository's synthetic records.

**Why this step matters:** shows that the server exposes a reusable capability,
not an interface that only works with this application's own agent.

**Review questions:** Can the other client discover the same schema? Does the
tool work without copying the host's agent code? Are setup and invocation
instructions reproducible?

**Done when:** complete. Inspector 2.8.0 discovered the schemas and invoked the
read-only demo employee tool successfully; exact PowerShell commands are above.

## Optional follow-up — Streamable HTTP and authentication

Do this after the local stdio flow is understood and reviewed. This is a
local bearer-token exercise, not OAuth or production authentication. The MCP
Python SDK's HTTP authorization guidance describes OAuth 2.1 resource-server
validation; use a real identity provider and token verifier for deployed
servers.

**Topics:** Streamable HTTP transport, endpoint configuration, HTTPS, bearer
tokens or OAuth, server allowlists, authorization, secret handling, and
deployment concerns.

**Activity completed locally:** `mcp_http_server.py` exposes the same tools on
Streamable HTTP at `http://127.0.0.1:8000/mcp`, guarded by a random bearer
token supplied through `MCP_DEMO_TOKEN`. It does not change the stdio server.
Use two PowerShell terminals from the project root.

Terminal A (copy the generated token for Terminal B):

```powershell
$env:MCP_DEMO_TOKEN = [guid]::NewGuid().ToString("N")
Write-Host $env:MCP_DEMO_TOKEN
py mcp_http_server.py
```

Terminal B, first check that a request without a bearer token is denied with
HTTP 401, then connect and call the synthetic record tool with the token:

```powershell
curl.exe -i -X POST http://127.0.0.1:8000/mcp -H "Content-Type: application/json" --data "{}"
$token = Read-Host "Paste the demo token from Terminal A"
npx --yes @modelcontextprotocol/inspector@2.8.0 --cli --transport http --server-url http://127.0.0.1:8000/mcp --header "Authorization: Bearer $token" --method tools/list
npx --yes @modelcontextprotocol/inspector@2.8.0 --cli --transport http --server-url http://127.0.0.1:8000/mcp --header "Authorization: Bearer $token" --method tools/call --tool-name get_employee_record --tool-arg employee_id=EMP-001 --format json
```

In this environment the request without a token returned **401**; Inspector
2.8.0 with the token discovered the tools and successfully called
`get_employee_record` for synthetic `EMP-001`. Stop Terminal A with Ctrl+C.
The demonstration token is shared-secret validation only: it does not identify
an employee or authorize access to a particular employee record. Never use it
for real data.

**Why it matters:** HTTP supports service-to-service integration, but adds
network and identity concerns that stdio avoids for a local learning demo.

**Done when:** the tool contract is reachable over HTTP and the local demo
returns 401 without a token while allowing a valid-token call. Production OAuth
and deployment hardening remain future work.

## Week 9 completion checklist

- [ ] Explain host, client, server, tool, and transport.
- [ ] Start a local MCP server over stdio.
- [ ] Discover tools and schemas through an MCP client.
- [ ] Call policy search through MCP from the host agent.
- [ ] Add a second tool without changing generic routing/client logic.
- [ ] Preserve policy evidence checks, authorization, and trajectory logging.
- [x] Demonstrate the server with another MCP client.
- [ ] Review each step's code and explain its purpose before moving forward.
- [ ] (Optional) Demonstrate HTTP transport and authentication.

## Short explanation to present

“The host runs the AI reasoning. Its MCP client connects to a server and
discovers the server's tools and input schemas. The server performs approved
operations and returns results. MCP standardizes this connection, while the
host still validates tool use and checks policy evidence before answering.”
