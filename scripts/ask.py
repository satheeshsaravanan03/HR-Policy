"""Interactive console for trying the pipeline by hand.

    python scripts/ask.py                      # interactive
    python scripts/ask.py "how many sick leaves in Bangalore?"
    python scripts/ask.py --search "carry over cap"        # retrieval only
    python scripts/ask.py --region India "carry over cap"
    python scripts/ask.py --strategy recursive "carry over cap"

--search performs no generation, so it spends no generation quota and shows
exactly what the retriever found, which is usually what you want when
diagnosing a wrong answer.
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.chunkers import RECURSIVE, STRUCTURE  # noqa: E402
from rag.generate import answer  # noqa: E402
from rag.retrieve import search  # noqa: E402


# Someone at the '>' prompt pasting the whole shell command instead of just the
# question. The stray 'python'/'scripts' tokens then trip the corpus-vocabulary
# refusal gate, which looks like a retrieval bug but is not one.
PASTED_COMMAND = re.compile(r"^\s*(?:py|python3?)\s+\S*ask\.py\b(?P<rest>.*)$", re.I)


def strip_pasted_command(text: str) -> str:
    """Recover the question from an accidentally pasted command line."""
    match = PASTED_COMMAND.match(text)
    if not match:
        return text
    rest = match.group("rest").strip()
    quoted = re.findall(r"[\"'](.+?)[\"']", rest)
    recovered = quoted[0] if quoted else re.sub(r"--?\S+(?:\s+\S+)?", "", rest).strip()
    if recovered:
        print(f"  (read as a pasted command; using just the question: {recovered!r})")
        return recovered
    return text


def show_hits(hits) -> None:
    if not hits:
        print("  (nothing retrieved)")
        return
    print(f"  {'#':<3}{'score':<9}{'policy_id':<13}{'section':<10}{'region':<12}source")
    for h in hits:
        print(f"  {h.rank:<3}{h.score:<9.4f}{h.policy_id:<13}{(h.section or '-'):<10}"
              f"{h.region:<12}{h.source_file}")


def run(query: str, args) -> None:
    print(f"\nQ: {query}")

    if args.search:
        hits = search(args.strategy, query, top_k=args.k, region=args.region)
        print(f"\nRETRIEVED (strategy={args.strategy}"
              f"{f', region={args.region}' if args.region else ''}):")
        show_hits(hits)
        if hits and args.show_text:
            print(f"\nTOP CHUNK TEXT:\n{hits[0].content[:800]}")
        return

    result = answer(args.strategy, query, top_k=args.k, region=args.region)
    print(f"\nA: {result.text}")

    if result.is_refusal:
        print(f"\n[REFUSED by gate: {result.gate}]")
    elif result.citations:
        print("\nCITATIONS:")
        for c in result.citations:
            flag = "ok" if c.resolves else "UNRESOLVED"
            print(f"  [{flag}] {c.chunk_id} -> {c.policy_id} section {c.section or '-'}")

    print("\nRETRIEVED:")
    show_hits(result.hits)


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the HR policy index.")
    parser.add_argument("query", nargs="*", help="question; omit for interactive mode")
    parser.add_argument("--strategy", choices=[STRUCTURE, RECURSIVE], default=STRUCTURE)
    parser.add_argument("--region", default=None, help="e.g. India, Arizona, Michigan")
    parser.add_argument("-k", type=int, default=5, help="how many chunks to retrieve")
    parser.add_argument("--search", action="store_true", help="retrieval only, no generation")
    parser.add_argument("--show-text", action="store_true", help="print the top chunk's text")
    args = parser.parse_args()

    if args.query:
        run(" ".join(args.query), args)
        return

    print("HR policy RAG. Type a question, or 'quit' to exit.")
    print(f"strategy={args.strategy} region={args.region or 'none'} "
          f"mode={'search-only' if args.search else 'generate'}")
    while True:
        try:
            query = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.lower() in {"quit", "exit", "q"}:
            return
        if query:
            run(strip_pasted_command(query), args)


if __name__ == "__main__":
    main()
