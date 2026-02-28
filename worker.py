"""LLM agent loop: drives tool providers for a single sample using Claude tool use."""

from __future__ import annotations

import base64
import json
import logging
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
- If a page is loading slowly, use the wait tool.
- Take screenshots when you need to see visual content not captured in text.
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
    client = anthropic.Anthropic()

    tools = registry.get_tool_schemas()
    user_prompt = build_user_prompt(instructions, csv_columns, row)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    for iteration in range(max_iterations):
        logger.info(f"  Iteration {iteration + 1}/{max_iterations}")

        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=tools,
                messages=messages,
            )
        except anthropic.APIError as e:
            logger.error(f"  API error: {e}")
            raise

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        tool_uses = [block for block in assistant_content if block.type == "tool_use"]

        if not tool_uses:
            if response.stop_reason == "end_turn":
                logger.warning("  Agent ended without calling complete tool. Prompting to complete.")
                messages.append({
                    "role": "user",
                    "content": "You must call the `complete` tool with the extracted data. Please do so now.",
                })
                continue
            break

        tool_results = []
        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            tool_id = tool_use.id

            logger.info(f"  Tool: {tool_name}({json.dumps(tool_input, default=str)[:120]})")

            # Intercept complete — handled here, not dispatched to providers
            if tool_name == "complete":
                data = tool_input.get("data", {})
                notes = tool_input.get("notes", "")
                logger.info(f"  Task complete. Notes: {notes}")
                # Take a final screenshot via the registry (browser provider handles it)
                try:
                    await registry.execute("screenshot", {"filename": "final.png"})
                except Exception:
                    pass
                return data

            # Dispatch to the registry
            try:
                result = await registry.execute(tool_name, tool_input)
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
                logger.error(f"  {result}")

            # Build tool result content
            content: list[dict[str, Any]] = []

            # If screenshot, include image in the response
            if tool_name == "screenshot":
                for png in output_dir.glob("*.png"):
                    if png.stat().st_mtime > 0:
                        # Find the most recently written screenshot
                        pass
                # Simpler: parse the path from the result text
                if "saved to" in result:
                    screenshot_path = Path(result.split("saved to ")[-1])
                    if screenshot_path.exists():
                        img_data = screenshot_path.read_bytes()
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64.b64encode(img_data).decode(),
                            },
                        })

            content.append({"type": "text", "text": result[:12000]})

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": content,
            })

        messages.append({"role": "user", "content": tool_results})

    raise RuntimeError(f"Agent did not complete within {max_iterations} iterations")
