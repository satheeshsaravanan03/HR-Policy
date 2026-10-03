"""One-command Week 10 single-agent vs A2A team race."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag.mcp_agent import run_mcp_policy_agent  # noqa: E402
from rag.week10_a2a import run_week10_team, score_answer  # noqa: E402


def _run_baseline(query: str) -> dict[str, Any]:
    started = time.perf_counter()
    result = asyncio.run(run_mcp_policy_agent(query))
    elapsed_ms = (time.perf_counter() - started) * 1000
    usage = result.get("llm_metrics", {})
    input_tokens = int(usage.get("input_tokens", 0))
    output_tokens = int(usage.get("output_tokens", 0))
    input_rate = os.getenv("WEEK10_INPUT_COST_PER_MILLION")
    output_rate = os.getenv("WEEK10_OUTPUT_COST_PER_MILLION")
    cost = None
    if input_rate is not None and output_rate is not None:
        cost = (input_tokens * float(input_rate) + output_tokens * float(output_rate)) / 1_000_000
    result["metrics"] = {
        "elapsed_ms": elapsed_ms,
        "llm_calls": int(bool(input_tokens or output_tokens)),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost": cost,
    }
    return result


def run_race() -> dict[str, Any]:
    case_file = ROOT / "data" / "week10_eval_cases.json"
    cases = json.loads(case_file.read_text(encoding="utf-8"))["cases"]
    input_rate = os.getenv("WEEK10_INPUT_COST_PER_MILLION")
    output_rate = os.getenv("WEEK10_OUTPUT_COST_PER_MILLION")
    rates = {
        "input_per_million": float(input_rate) if input_rate is not None else None,
        "output_per_million": float(output_rate) if output_rate is not None else None,
    }
    rows = []
    for case in cases:
        record: dict[str, Any] = {"case_id": case["case_id"], "query": case["query"]}
        for label, runner in (("single_agent", _run_baseline), ("multi_agent", lambda q: run_week10_team(q, rates))):
            try:
                result = runner(case["query"])
                record[label] = {
                    "answer": result.get("answer", ""),
                    "status": result.get("status", "error"),
                    "stop_reason": result.get("stop_reason", "unknown"),
                    "metrics": result.get("metrics", {}),
                    "score": score_answer(result, case),
                    "selected_specialists": result.get("selected_specialists", []),
                    "tasks": result.get("tasks", []),
                    "steps": result.get("steps", result.get("execution_steps", [])),
                }
            except Exception as exc:  # keep failures in the race rather than dropping cases
                record[label] = {
                    "answer": f"Run failed: {type(exc).__name__}",
                    "status": "error",
                    "stop_reason": type(exc).__name__,
                    "metrics": {"elapsed_ms": 0, "llm_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "estimated_cost": None},
                    "score": {"score": 0, "facts_found": 0, "facts_expected": len(case.get("expected_terms", [])), "citation_ok": False, "pass": False},
                    "selected_specialists": [], "tasks": [], "steps": [],
                }
        rows.append(record)

    def aggregate(name: str) -> dict[str, Any]:
        runs = [row[name] for row in rows]
        metrics = [run["metrics"] for run in runs]
        estimated = [m["estimated_cost"] for m in metrics if m.get("estimated_cost") is not None]
        return {
            "quality_percent": round(sum(run["score"]["score"] for run in runs) / max(1, len(runs)), 1),
            "pass_count": sum(bool(run["score"]["pass"]) for run in runs),
            "case_count": len(runs),
            "mean_latency_ms": round(sum(float(m.get("elapsed_ms", 0)) for m in metrics) / max(1, len(metrics)), 1),
            "total_tokens": sum(int(m.get("total_tokens", 0)) for m in metrics),
            "input_tokens": sum(int(m.get("input_tokens", 0)) for m in metrics),
            "output_tokens": sum(int(m.get("output_tokens", 0)) for m in metrics),
            "estimated_cost_total": round(sum(estimated), 8) if estimated else None,
            "cost_cases_counted": len(estimated),
        }

    single_metrics = aggregate("single_agent")
    team_metrics = aggregate("multi_agent")
    quality_delta = team_metrics["quality_percent"] - single_metrics["quality_percent"]
    latency_delta = team_metrics["mean_latency_ms"] - single_metrics["mean_latency_ms"]
    token_delta = team_metrics["total_tokens"] - single_metrics["total_tokens"]
    if quality_delta > 0:
        verdict = (
            f"The A2A team leads the deterministic quality proxy by {quality_delta:.1f} percentage points; "
            f"its mean latency changes by {latency_delta:+.1f} ms and total tokens by {token_delta:+d}. "
            "Treat that as a candidate to keep only after manually checking the scored answers and citations."
        )
    else:
        verdict = (
            f"The single-agent baseline is the provisional keeper: the team does not improve the deterministic "
            f"quality proxy (delta {quality_delta:+.1f} percentage points), while mean latency changes by "
            f"{latency_delta:+.1f} ms and total tokens by {token_delta:+d}. Keep multi-agent delegation for "
            "cases where specialist separation or parallel work is specifically useful."
        )
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "case_file": str(case_file.relative_to(ROOT)),
        "scoring_note": "Quality is a deterministic proxy from expected facts/refusal/citation checks; inspect raw cases before claiming semantic quality.",
        "model": os.getenv("GROQ_GENERATION_MODEL", "openai/gpt-oss-120b"),
        "rates_per_million_tokens": rates,
        "single_agent": single_metrics,
        "multi_agent": team_metrics,
        "verdict": verdict,
        "cases": rows,
    }
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    (output_dir / "week10_agent_race.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "week10_agent_race.md").write_text(_markdown(report), encoding="utf-8")
    return report


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Week 10 — Single-Agent vs Multi-Agent Race", "",
        f"Created: {report['created_at']}", "",
        "Quality is a deterministic expected-fact/refusal/citation proxy, not an LLM semantic judge. Cost is omitted unless token rates are configured.", "",
        "| System | Quality | Passed | Mean latency (ms) | Input tokens | Output tokens | Total tokens | Estimated cost |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, key in (("Single-agent MCP baseline", "single_agent"), ("A2A specialist team", "multi_agent")):
        item = report[key]
        cost = f"${item['estimated_cost_total']:.6f}" if item["estimated_cost_total"] is not None else "not configured"
        lines.append(f"| {label} | {item['quality_percent']}% | {item['pass_count']}/{item['case_count']} | {item['mean_latency_ms']} | {item['input_tokens']} | {item['output_tokens']} | {item['total_tokens']} | {cost} |")
    lines += ["", "## Provisional verdict", "", report["verdict"], ""]
    lines += ["", "## Per-case results", "", "| Case | Single quality | Team quality | Single latency ms | Team latency ms | Team specialists |", "|---|---:|---:|---:|---:|---|"]
    for case in report["cases"]:
        single, team = case["single_agent"], case["multi_agent"]
        lines.append(f"| {case['case_id']} | {single['score']['score']}% | {team['score']['score']}% | {single['metrics'].get('elapsed_ms', 0):.0f} | {team['metrics'].get('elapsed_ms', 0):.0f} | {', '.join(team.get('selected_specialists', []))} |")
    for case in report["cases"]:
        lines += ["", f"### {case['case_id']} — {case['query']}", "", "**Single-agent:**", "", str(case["single_agent"]["answer"]), "", "**A2A team:**", "", str(case["multi_agent"]["answer"]), "", "**A2A task IDs/status:**", "", "```json", json.dumps(case["multi_agent"].get("tasks", []), indent=2), "```"]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    summary = run_race()
    print(f"Week 10 race complete: single={summary['single_agent']['quality_percent']}%, team={summary['multi_agent']['quality_percent']}%")
    print("Reports: output/week10_agent_race.md and output/week10_agent_race.json")
