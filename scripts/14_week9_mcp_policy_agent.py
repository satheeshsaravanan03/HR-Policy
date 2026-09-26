"""Ask a single-policy question through the MCP-backed host flow.

Run from the project root:
    python scripts/14_week9_mcp_policy_agent.py "What is the annual leave policy?"
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.util
import sys
import traceback
from pathlib import Path

# Running a file under scripts/ puts that directory, rather than the project
# root, at sys.path[0]. Add the root so the sibling rag package is importable.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

required_modules = ("mcp", "langchain_groq")
missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
if missing_modules:
    missing_text = ", ".join(missing_modules)
    print(
        f"Missing project dependencies: {missing_text}. Install this project's "
        "requirements with `py -m pip install -r requirements.txt`, then rerun.",
        file=sys.stderr,
    )
    raise SystemExit(1)

from rag.mcp_agent import run_mcp_policy_agent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="A single-policy question")
    args = parser.parse_args()

    try:
        result = asyncio.run(run_mcp_policy_agent(args.question))
    except Exception as exc:
        traceback.print_exception(exc, file=sys.stderr)
        parser.exit(1, f"MCP policy agent failed: {exc}\n")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    print(result["answer"])
    print(f"\nStatus: {result['status']} ({result['stop_reason']})")
    for citation in result.get("citations", []):
        print(f"Citation: {citation.chunk_id} | {citation.policy_id} | section {citation.section}")
    if result.get("discovered_tool"):
        print(f"MCP tool: {result['discovered_tool']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
