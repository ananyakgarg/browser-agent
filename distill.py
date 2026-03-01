"""Distill: converts a raw pioneer trace into a clean, reusable playbook."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

DISTILL_MODEL = "claude-sonnet-4-6"

DISTILL_SYSTEM = """\
You are a technical writer distilling a browser automation agent's raw execution \
trace into a clean, reusable playbook.

You will receive:
1. The original task instructions
2. The agent's full action trace (every tool call and result)
3. The agent's self-written playbook (if any)

Your job: produce a CLEAN numbered step-by-step guide that strips away all \
failed attempts, backtracking, loops, and exploration. Keep ONLY the successful \
action chain — the shortest path from start to completion.

Rules:
- Use {column_name} placeholders for any sample-specific values (URLs, IDs, names)
- Remove ALL specific data from this run — only keep the METHOD
- Include exact execute_js expressions that worked
- Include exact URL patterns with {placeholders}
- Note which elements to look for and how to identify them
- Include what to wait for / verify after each action
- Note any gotchas or edge cases the agent encountered
- If the agent used search_page, include the query patterns
- Keep it concise — a follower agent should execute this mechanically
- Number every step
- Do NOT include steps that failed or were retried — only the final working approach
"""


def build_trace_text(trace_path: Path, max_steps: int = 50) -> str:
    """Build a readable text summary of the full trace for the distillation LLM."""
    if not trace_path.exists():
        return "(no trace available)"

    trace = json.loads(trace_path.read_text())
    lines = []

    for step in trace[:max_steps]:
        iteration = step.get("iteration", "?")
        lines.append(f"--- Step {iteration} ---")

        # Include reasoning (the agent's thinking)
        reasoning = step.get("reasoning", "")
        if reasoning:
            # Truncate long reasoning
            if len(reasoning) > 500:
                reasoning = reasoning[:500] + "..."
            lines.append(f"Thinking: {reasoning}")

        # Include tool calls and results
        tools = step.get("tools", [])
        for t in tools:
            tool_name = t.get("tool", "")
            tool_input = t.get("input", {})
            result = t.get("result", "")

            # Format input
            input_str = json.dumps(tool_input, default=str)
            if len(input_str) > 300:
                input_str = input_str[:300] + "..."

            # Format result
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."

            lines.append(f"Action: {tool_name}({input_str})")
            lines.append(f"Result: {result_str}")

        # Note loop/budget warnings
        if step.get("loop_nudge"):
            lines.append("[LOOP DETECTED — agent was stuck here]")
        if step.get("budget_warning"):
            lines.append("[BUDGET WARNING — running low on iterations]")

        lines.append("")

    return "\n".join(lines)


def distill_playbook(
    original_instructions: str,
    trace_path: Path,
    raw_playbook: str | None = None,
    model: str | None = None,
) -> str:
    """Distill a pioneer's trace into a clean follower playbook.

    Uses the full trace (not just the raw playbook) to understand what
    actually worked vs what was exploration/backtracking.
    """
    client = anthropic.Anthropic(max_retries=10)
    distill_model = model or DISTILL_MODEL

    trace_text = build_trace_text(trace_path)

    user_parts = [
        f"## Original Task Instructions\n{original_instructions}",
        f"## Full Action Trace\n{trace_text}",
    ]

    if raw_playbook:
        user_parts.append(f"## Agent's Self-Written Playbook\n{raw_playbook}")

    user_parts.append(
        "\nDistill the above into a clean numbered playbook. "
        "Strip all failed attempts and backtracking. "
        "Use {column_name} placeholders for sample-specific values. "
        "Include exact JS expressions and URL patterns that worked."
    )

    user_msg = "\n\n".join(user_parts)

    response = client.messages.create(
        model=distill_model,
        max_tokens=2048,
        system=DISTILL_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "text":
            return block.text

    # Fallback to raw playbook if distillation fails
    if raw_playbook:
        logger.warning("Distillation produced no text, falling back to raw playbook")
        return raw_playbook

    return "(distillation failed — no playbook available)"
