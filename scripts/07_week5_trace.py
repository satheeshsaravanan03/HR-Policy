"""Week 5 trace sampler/replayer.

This tool only selects and displays real stored traces; it never invents
observations or edits the application behavior.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TRACE_PATH = Path(__file__).resolve().parent.parent / "output" / "traces.jsonl"


def load() -> list[dict]:
    if not TRACE_PATH.exists():
        return []
    return [json.loads(line) for line in TRACE_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260901)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--replay", help="trace_id to display for replay evidence")
    args = parser.parse_args()
    traces = load()
    if not traces:
        print(f"No traces found at {TRACE_PATH}. Run real app questions first.")
        return
    rng = random.Random(args.seed)
    chosen = rng.sample(traces, min(args.sample, len(traces)))
    print(f"trace_file={TRACE_PATH}")
    print(f"seed={args.seed} sample_size={len(chosen)}")
    print("sample_trace_ids=")
    for trace in chosen:
        print(f"  {trace['trace_id']}")
    if args.replay:
        trace = next((item for item in traces if item.get("trace_id") == args.replay), None)
        if trace is None:
            raise SystemExit(f"Trace not found: {args.replay}")
        print("\nREPLAY FROM TRACE ALONE")
        print(json.dumps({"trace_id": trace["trace_id"], "query": trace["query"], "raw_output": trace["raw_output"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

