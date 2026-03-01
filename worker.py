"""LLM agent loop: drives tool providers for a single sample using Claude tool use."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import anthropic

from tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-5-20250929"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are a browser automation agent. You control a web browser to complete tasks by using the provided tools.

After each action, you will see the updated page state showing:
- Current URL and page title
- A numbered list of interactive elements (links, buttons, inputs, etc.)
- Truncated visible text from the page

To interact with elements, use their ID number (shown in square brackets like [0], [1], etc.).

Guidelines:
- Be methodical: navigate to the right page, find the right elements, extract the data.
- Prefer get_text and the page state text for data extraction — text is faster and cheaper than screenshots.
- Only take screenshots if the task instructions explicitly ask for screenshots or if an output field requires an image. Never take screenshots just to "see" a page — the text representation already shows you the content.
- If a page is loading slowly, use the wait tool.
- If you get stuck, try scrolling, going back, or finding elements by text/selector.
- When you have gathered all required information, call the complete tool with the extracted data.
- The data keys in your complete call MUST match the required output columns exactly.
"""


def build_user_prompt(instructions: str, csv_columns: list[str], row: dict[str, Any]) -> str:
    """Build the initial user message for the worker."""
    cols = ", ".join(f'"{c}"' for c in csv_columns)
    row_summary = json.dumps(row, indent=2, default=str)
    return f"""\
## Task Instructions
{instructions}

## Required Output Columns
You must extract the following fields and provide them in your `complete` tool call:
{cols}

## Input Data for This Sample
```json
{row_summary}
```

Begin by navigating to the appropriate page and completing the task. When done, call the `complete` tool with all required fields.
"""


async def run_worker(
    registry: ToolRegistry,
    instructions: str,
    csv_columns: list[str],
    row: dict[str, Any],
    output_dir: Path,
    max_iterations: int = 30,
) -> dict[str, Any]:
    """
    Run the LLM agent loop for a single sample.

    Returns the extracted data dict from the complete tool call.
    Raises on failure (timeout, max iterations, API error).
    """
    client = anthropic.Anthropic(max_retries=25)

    tools = registry.get_tool_schemas()
    user_prompt = build_user_prompt(instructions, csv_columns, row)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    # Ensure output dir exists before writing any trace/audit files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Trace log — records every step for debugging/review
    trace: list[dict[str, Any]] = []
    trace_path = output_dir / "trace.json"
    audit_path = output_dir / "audit.json"
    start_time = time.time()
    total_input_tokens = 0
    total_output_tokens = 0
    api_calls = 0

    def _save_trace():
        with open(trace_path, "w") as f:
            json.dump(trace, f, indent=2, default=str)

    def _save_audit():
        audit = {
            "model": MODEL,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_input_tokens + total_output_tokens,
            "api_calls": api_calls,
            "iterations": len(trace),
            "elapsed_sec": round(time.time() - start_time, 2),
        }
        with open(audit_path, "w") as f:
            json.dump(audit, f, indent=2)

    for iteration in range(max_iterations):
        logger.info(f"  Iteration {iteration + 1}/{max_iterations}")
        step: dict[str, Any] = {
            "iteration": iteration + 1,
            "timestamp": round(time.time() - start_time, 2),
        }

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as e:
            step["error"] = str(e)
            trace.append(step)
            _save_trace()
            _save_audit()
            raise

        api_calls += 1
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens
        step["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Extract reasoning text from the response
        reasoning = []
        for block in assistant_content:
            if block.type == "text":
                reasoning.append(block.text)
        if reasoning:
            step["reasoning"] = "\n".join(reasoning)

        tool_uses = [block for block in assistant_content if block.type == "tool_use"]

        if not tool_uses:
            step["action"] = "no_tool_call"
            step["stop_reason"] = response.stop_reason
            trace.append(step)
            _save_trace()
            _save_audit()

            if response.stop_reason == "end_turn":
                logger.warning("  Agent ended without calling complete tool. Prompting to complete.")
                messages.append({
                    "role": "user",
                    "content": "You must call the `complete` tool with the extracted data. Please do so now.",
                })
                continue
            break

        tool_results = []
        step_tools: list[dict[str, Any]] = []

        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            tool_id = tool_use.id

            logger.info(f"  Tool: {tool_name}({json.dumps(tool_input, default=str)[:120]})")

            tool_trace: dict[str, Any] = {
                "tool": tool_name,
                "input": tool_input,
            }

            # Intercept complete — handled here, not dispatched to providers
            if tool_name == "complete":
                data = tool_input.get("data", {})
                notes = tool_input.get("notes", "")
                logger.info(f"  Task complete. Notes: {notes}")

                tool_trace["result"] = "TASK COMPLETE"
                step_tools.append(tool_trace)
                step["tools"] = step_tools
                trace.append(step)
                _save_trace()
                _save_audit()

                # No auto-screenshot on complete — agent takes them only when needed
                return data

            # Dispatch to the registry
            try:
                result = await registry.execute(tool_name, tool_input)
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
                logger.error(f"  {result}")

            # Save truncated result to trace (full page state is too verbose)
            tool_trace["result"] = result[:2000]
            step_tools.append(tool_trace)

            # Build tool result content
            content: list[dict[str, Any]] = []

            # If screenshot, include image in the response (skip if >4.5MB)
            if tool_name == "screenshot":
                if "saved to" in result:
                    screenshot_path = Path(result.split("saved to ")[-1])
                    if screenshot_path.exists():
                        img_data = screenshot_path.read_bytes()
                        if len(img_data) <= 4_500_000:
                            content.append({
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64.b64encode(img_data).decode(),
                                },
                            })
                        else:
                            logger.warning(f"  Screenshot too large ({len(img_data)} bytes), sending text-only result")

            content.append({"type": "text", "text": result[:12000]})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            })

        step["tools"] = step_tools
        trace.append(step)
        _save_trace()

        messages.append({"role": "user", "content": tool_results})

    _save_trace()
    _save_audit()
    raise RuntimeError(f"Agent did not complete within {max_iterations} iterations")
