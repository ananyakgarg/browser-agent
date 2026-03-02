"""Judge: scores pioneer traces and picks the best one for playbook distillation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anthropic

from distill import build_trace_text

logger = logging.getLogger(__name__)

JUDGE_MODEL = "claude-sonnet-4-6"

JUDGE_SYSTEM = """\
You are a judge evaluating a browser automation agent's execution trace and output.

You will receive:
1. The original task instructions the agent was given
2. The agent's full execution trace (every tool call, result, and reasoning)
3. The agent's final output data (the fields it extracted)
4. The list of required output columns

Evaluate the agent on these 4 dimensions. For EACH dimension, write 1-2 sentences \
of concrete reasoning BEFORE giving a score (1-10).

**Correctness** (weight: 40%): Did the agent successfully complete the task as \
described in the instructions? Compare what was asked to what was delivered. \
Did it accomplish the actual goal, or did it go off track, skip steps, or \
produce results that don't satisfy the task requirements?

**Completeness** (weight: 25%): Are all required fields filled with meaningful data? \
Empty strings, "N/A", "unknown", or placeholder values count against completeness.

**Methodology** (weight: 20%): Did the agent take a reasonable path to the answer? \
Did it recover well from failures? Did it verify findings before completing? \
Excessive loops or brute-force approaches score lower.

**Evidence quality** (weight: 15%): Are the screenshots, diffs, and explanations \
substantive? Could a human reviewer verify the conclusion from the evidence provided? \
Vague explanations or missing evidence score lower.

After scoring all dimensions, write an overall assessment in `overall_reasoning` \
(2-3 sentences) and compute `overall_score` (0-100) using the weights above: \
overall_score = correctness*4 + completeness*2.5 + methodology*2 + evidence*1.5

You MUST include all 6 fields in the judgment tool call: correctness, completeness, \
methodology, evidence, overall_reasoning, and overall_score.\
"""

JUDGE_TOOL = {
    "name": "judgment",
    "description": "Submit the structured evaluation of the agent's work.",
    "input_schema": {
        "type": "object",
        "properties": {
            "correctness": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["reasoning", "score"],
            },
            "completeness": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["reasoning", "score"],
            },
            "methodology": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["reasoning", "score"],
            },
            "evidence": {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string"},
                    "score": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["reasoning", "score"],
            },
            "overall_reasoning": {"type": "string"},
            "overall_score": {"type": "integer", "minimum": 0, "maximum": 100},
        },
        "required": [
            "correctness", "completeness", "methodology",
            "evidence", "overall_reasoning", "overall_score",
        ],
    },
}


def llm_judge(
    trace_path: Path,
    audit_path: Path,
    result_data: dict[str, Any] | None,
    csv_columns: list[str],
    task_instructions: str,
) -> dict[str, Any]:
    """Score a pioneer trace using an LLM that reasons about output quality.

    Falls back to score_trace() if the LLM call fails.
    """
    if result_data is None:
        # Nothing to judge — use math scorer
        return score_trace(trace_path, audit_path, result_data, csv_columns)

    trace_text = build_trace_text(trace_path)

    user_parts = [
        f"## Task Instructions\n{task_instructions}",
        f"## Execution Trace\n{trace_text}",
        f"## Final Output Data\n{json.dumps(result_data, indent=2, default=str)}",
        f"## Required Output Columns\n{json.dumps(csv_columns)}",
    ]
    user_msg = "\n\n".join(user_parts)

    try:
        client = anthropic.Anthropic(max_retries=3)
        response = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=4096,
            system=JUDGE_SYSTEM,
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": "judgment"},
            messages=[{"role": "user", "content": user_msg}],
        )

        if response.stop_reason == "max_tokens":
            raise RuntimeError("Judge response truncated (max_tokens)")

        for block in response.content:
            if block.type == "tool_use" and block.name == "judgment":
                judgment = block.input
                logger.debug(f"Raw LLM judgment keys: {list(judgment.keys())}")
                if "overall_score" not in judgment:
                    logger.warning(
                        f"LLM judge missing 'overall_score'. Keys: {list(judgment.keys())}. "
                        f"Raw: {json.dumps(judgment, indent=2)[:500]}"
                    )
                    raise KeyError("overall_score")
                # Add metadata from audit
                audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}
                judgment["total_tokens"] = audit.get("total_tokens", 0)
                judgment["elapsed_sec"] = audit.get("elapsed_sec", 0)
                judgment["iterations"] = len(
                    json.loads(trace_path.read_text()) if trace_path.exists() else []
                )
                judgment["completed"] = True
                judgment["score"] = judgment["overall_score"]
                return judgment

        logger.warning("LLM judge returned no tool use, falling back to math scorer")
    except Exception as e:
        logger.warning(f"LLM judge failed ({e}), falling back to math scorer")

    return score_trace(trace_path, audit_path, result_data, csv_columns)


def score_trace(
    trace_path: Path,
    audit_path: Path,
    result_data: dict[str, Any] | None,
    csv_columns: list[str],
) -> dict[str, Any]:
    """Score a single pioneer trace.

    Returns a dict with individual metrics and an overall score (0-100).
    Higher is better.
    """
    score = 0
    details: dict[str, Any] = {}

    # Load trace
    trace = json.loads(trace_path.read_text()) if trace_path.exists() else []
    audit = json.loads(audit_path.read_text()) if audit_path.exists() else {}

    # 1. Completion — did it produce data? (50 points)
    completed = result_data is not None
    details["completed"] = completed
    if completed:
        score += 50

    # 2. Field coverage — what fraction of required columns are populated? (25 points)
    if completed and csv_columns:
        # Exclude internal fields like 'playbook'
        required = [c for c in csv_columns]
        filled = sum(
            1 for c in required
            if c in result_data and result_data[c] not in (None, "", "N/A", "unknown")
        )
        coverage = filled / len(required) if required else 1.0
        details["field_coverage"] = round(coverage, 2)
        details["fields_filled"] = filled
        details["fields_required"] = len(required)
        score += int(coverage * 25)
    else:
        details["field_coverage"] = 0.0

    # 3. Efficiency — fewer iterations = better (15 points)
    iterations = len(trace)
    details["iterations"] = iterations
    if iterations > 0:
        # 1 iteration = full 15 points, 30 iterations = 0 points
        efficiency = max(0.0, 1.0 - (iterations - 1) / 29.0)
        score += int(efficiency * 15)
        details["efficiency"] = round(efficiency, 2)

    # 4. No loops — clean execution without getting stuck (5 points)
    loop_nudges = sum(1 for s in trace if s.get("loop_nudge"))
    details["loop_nudges"] = loop_nudges
    if loop_nudges == 0:
        score += 5

    # 5. Has playbook — pioneer actually wrote a guide (5 points)
    has_playbook = bool(result_data and result_data.get("playbook"))
    details["has_playbook"] = has_playbook
    if has_playbook:
        score += 5

    # Token cost from audit (informational, not scored)
    details["total_tokens"] = audit.get("total_tokens", 0)
    details["elapsed_sec"] = audit.get("elapsed_sec", 0)

    details["score"] = score
    return details


def pick_winner(
    candidates: list[dict[str, Any]],
) -> int:
    """Pick the best pioneer from scored candidates.

    Each candidate should have: {"pioneer_id": int, "score": <score_dict>, "result": <data|None>}
    Returns the index of the winner.
    """
    if not candidates:
        raise ValueError("No candidates to judge")

    # Filter to only completed candidates
    completed = [(i, c) for i, c in enumerate(candidates) if c["score"]["completed"]]

    if not completed:
        # None completed — pick the one that got furthest (most iterations = most work done)
        logger.warning("No pioneers completed. Picking the one with most progress.")
        return max(range(len(candidates)), key=lambda i: candidates[i]["score"]["iterations"])

    if len(completed) == 1:
        return completed[0][0]

    # Among completed, pick highest score
    best_idx = max(completed, key=lambda x: x[1]["score"]["score"])[0]
    return best_idx


def format_judgment(candidates: list[dict[str, Any]], winner_idx: int) -> str:
    """Format a human-readable judgment summary."""
    lines = ["Pioneer Tournament Results:", "=" * 40]

    for i, c in enumerate(candidates):
        s = c["score"]
        marker = " << WINNER" if i == winner_idx else ""

        # Handle both LLM judgment and math score formats
        if "overall_reasoning" in s:
            # LLM judge format
            dims = []
            for dim in ("correctness", "completeness", "methodology", "evidence"):
                if dim in s:
                    dims.append(f"{dim}={s[dim]['score']}/10")
            lines.append(
                f"  Pioneer {c['pioneer_id']}: score={s['score']}/100 "
                f"{' '.join(dims)} iterations={s.get('iterations', '?')}"
                f"{marker}"
            )
            if marker:
                lines.append(f"    Reasoning: {s['overall_reasoning']}")
        else:
            # Math score fallback format
            lines.append(
                f"  Pioneer {c['pioneer_id']}: score={s['score']}/100 "
                f"completed={s['completed']} fields={s.get('field_coverage', 0):.0%} "
                f"iterations={s['iterations']} loops={s['loop_nudges']}"
                f"{marker}"
            )

    winner = candidates[winner_idx]
    lines.append(f"\nWinner: Pioneer {winner['pioneer_id']} (score {winner['score']['score']}/100)")
    return "\n".join(lines)
