"""LLM agent loop: drives tool providers for a single sample using Claude tool use."""

from __future__ import annotations

import base64
import json
import logging
import time
from pathlib import Path
from typing import Any

import anthropic

from telemetry import LearningTelemetry, hash_tool_input
from tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096

SYSTEM_PROMPT = """\
You are a browser automation agent. You complete tasks by using the provided tools.

After each action, you will see the page state: current URL, title, interactive elements \
(with IDs like [0], [1], and *[2] for NEW elements that just appeared), indented to show \
parent-child structure, and truncated visible text.

## Step Discipline
Before EVERY action, explicitly state:
1. **Evaluate**: Did my last action succeed? Check the page state or screenshot. \
If the URL didn't change, expected elements aren't visible, or data is missing, \
the action likely failed — do NOT proceed as if it worked.
2. **Memory**: What key facts have I learned so far that I need to remember?
3. **Next goal**: What specific thing am I trying to accomplish in this step?

## Tools
- Use search_page FIRST when looking for specific text on a page. It searches the \
full page text (including off-screen content) instantly. Always try this before \
scrolling or writing execute_js queries.
- Use execute_js for targeted DOM extraction. Examples:
  - document.querySelector('h1').innerText
  - [...document.querySelectorAll('table tr')].map(r => r.innerText)
  - window.scrollBy(0, 500)
- Use navigate to go to URLs, click to interact with elements by ID.
- Use http_request for quick URL checks (status codes, link verification) without \
browser navigation.
- Use run_python to process data, parse HTML/JSON, do calculations, or validate results.
- Use hover for dropdown menus and tooltips. Use press_key for keyboard shortcuts.
- Use screenshot to VERIFY important actions succeeded — screenshots are ground truth. \
If you clicked a button or submitted a form, screenshot to confirm it worked before \
moving on.

## Handling Blocked Sites
If you see a block page (403, CAPTCHA, "automated tool detected", Cloudflare challenge):
1. FIRST try: configure_browser(action="enable_stealth") then re-navigate to the same URL.
2. If still blocked: configure_browser(action="set_headers", value='{"Accept-Language":"en-US"}') \
and try again.
3. If you find a workaround that works: call save_site_workaround to remember it for future runs.
4. LAST resort: fall back to http_request or API endpoints for DATA extraction only.

NEVER do the following workaround: fetching HTML via http_request, saving it to a \
local file, and opening it with file:///. This produces fake screenshots of locally \
rendered HTML, not the real website. If you cannot load a page in the browser, \
extract the data you need via http_request/API and note that the screenshot could \
not be taken. Partial results with honest notes are better than fake screenshots.

## Avoiding Loops
- Track your STRATEGY, not just individual tool calls. Scrolling to position 85000 \
and then 92000 is the SAME strategy (scroll-and-hope). If scroll+check doesn't \
find your target in 2 attempts, the page is probably using virtual scrolling — \
the content isn't in the DOM and more scrolling won't fix it.
- When a browser approach hits a wall, switch to data access: use http_request to \
hit the site's API, fetch raw file content, or use your code analysis tools. \
APIs return ALL the data, not just what's rendered in a viewport.
- Do NOT repeat the same strategy more than twice. Changing parameters (different \
scroll offset, different selector) does NOT count as a different approach. A truly \
different approach means a different data source or method entirely.
- If you've been trying to extract data from the DOM for 3+ iterations and it's \
not working, STOP and ask yourself: is there an API endpoint, a raw URL, or a \
tool that gives me this data directly without browser rendering?

## Completing the Task
- When done, call complete with all required output fields.
- The data keys in your complete call MUST match the required output columns exactly.
- For any field requiring judgment (ratings, flags, assessments): you MUST first \
extract the raw source data that supports your conclusion. Never assess based on \
your training knowledge alone.
- BEFORE calling complete, verify:
  1. Re-read the required output columns — is every field populated?
  2. Did you actually extract each value from the page (not from memory)?
  3. If screenshots were required, do they exist?
  4. If any field is missing or uncertain, note it — partial results beat hallucinated ones.
"""

PIONEER_ADDENDUM = """

## PIONEER MODE
You are the first agent to attempt this task. After you complete it, your approach \
will be used to guide other agents on similar samples. As you work:

1. Complete the task normally — extract all required data.
2. When you call `complete`, include an extra field called `playbook` in your data. \
The playbook should be a step-by-step guide that another agent could follow on a \
DIFFERENT input to complete the same task without any exploration or trial-and-error.

Your playbook should include:
- URL patterns with {placeholders} for variable parts (e.g. "Navigate to {pr_url}")
- Exact execute_js expressions that worked for extracting data
- Which elements to look for and how to identify them
- What to wait for after each action
- Gotchas, fallbacks, and edge cases you encountered
- The order of operations that worked

Be specific and concrete — the follower agents should be able to execute your playbook \
almost mechanically.
"""

FOLLOWER_PREAMBLE = """\
A pioneer agent already completed this task on a similar sample and documented \
the following playbook. Follow these steps closely — only deviate if something \
unexpected happens or a step fails.

=== PLAYBOOK ===
{playbook}
=== END PLAYBOOK ===

"""


def _classify_error_code(text: str) -> str | None:
    lowered = text.lower()
    if "timeout" in lowered:
        return "timeout"
    if "403" in lowered or "forbidden" in lowered:
        return "http_403"
    if "429" in lowered or "too many requests" in lowered:
        return "http_429"
    if "captcha" in lowered or "cloudflare" in lowered or "automated tool" in lowered:
        return "bot_detection"
    if "not found" in lowered:
        return "not_found"
    if "unknown tool" in lowered:
        return "unknown_tool"
    if "invalid" in lowered:
        return "invalid_input"
    if "error" in lowered or "failed" in lowered:
        return "runtime_error"
    return None


def _classify_tool_result(tool_name: str, result: str) -> tuple[str, bool, str | None]:
    if tool_name == "complete":
        return "complete", True, None
    err = _classify_error_code(result)
    if err:
        if err in {"http_403", "http_429", "bot_detection"}:
            return "blocked_or_rate_limited", False, err
        return "tool_error", False, err
    return "tool_success", True, None


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

When done, call the `complete` tool with all required fields.
"""


async def run_worker(
    registry: ToolRegistry,
    instructions: str,
    csv_columns: list[str],
    row: dict[str, Any],
    output_dir: Path,
    max_iterations: int = 30,
    model_override: str | None = None,
    pioneer_mode: bool = False,
    playbook: str | None = None,
    system_prompt_override: str | None = None,
    run_id: str | None = None,
    sample_id: str | None = None,
    phase: str = "browser",
    telemetry: LearningTelemetry | None = None,
) -> dict[str, Any]:
    """
    Run the LLM agent loop for a single sample.

    Returns the extracted data dict from the complete tool call.
    If pioneer_mode=True, the data will include a 'playbook' field.
    If playbook is provided, it's prepended to instructions as a guide.
    Raises on failure (timeout, max iterations, API error).
    """
    client = anthropic.Anthropic(max_retries=25)
    model = model_override or MODEL
    resolved_sample_id = sample_id or str(row.get("sample_id") or row.get("id") or "unknown")

    # Compaction beta — Claude auto-summarizes old context when it gets large
    COMPACTION_BETA = "compact-2026-01-12"
    COMPACTION_TRIGGER_TOKENS = 50_000

    # Build system prompt — augment for pioneer mode
    system_prompt = system_prompt_override or SYSTEM_PROMPT
    if pioneer_mode:
        system_prompt += PIONEER_ADDENDUM

    # Prepend playbook to instructions for follower mode
    if playbook:
        instructions = FOLLOWER_PREAMBLE.format(playbook=playbook) + instructions

    tools = registry.get_tool_schemas()
    user_prompt = build_user_prompt(instructions, csv_columns, row)

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": user_prompt},
    ]

    # Scout phase — agent plans before acting (skip for resolve agents and followers)
    skip_scout = system_prompt_override is not None or playbook is not None
    if not skip_scout:
        tool_names = [t["name"] for t in tools]
        scout_prompt = (
            "Before taking any action, write a brief numbered plan (3-7 steps) "
            "for how you will complete this task. Consider:\n"
            "- What URL(s) to navigate to first\n"
            "- What data you need to find and how (search_page, execute_js, etc.)\n"
            "- What screenshots or extractions are required\n"
            "- The most efficient order of operations\n\n"
            f"Available tools: {', '.join(tool_names)}\n\n"
            "Output ONLY your numbered plan. Do not take any actions yet."
        )
        messages.append({"role": "user", "content": scout_prompt})

        logger.info("  Scout phase — generating plan...")
        try:
            scout_response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=messages,
            )
            # Extract the plan text
            plan_text = ""
            for block in scout_response.content:
                if hasattr(block, "type") and block.type == "text":
                    plan_text += block.text
            if plan_text:
                messages.append({"role": "assistant", "content": scout_response.content})
                messages.append({
                    "role": "user",
                    "content": "Good plan. Now execute it step by step. Start with step 1.",
                })
                logger.info(f"  Plan:\n{plan_text[:300]}")
            else:
                # Scout returned nothing useful — remove the scout prompt
                messages.pop()  # remove scout prompt
        except Exception as e:
            logger.warning(f"  Scout failed: {e}. Proceeding without plan.")
            messages.pop()  # remove scout prompt

    # Ensure output dir exists before writing any trace/audit files
    output_dir.mkdir(parents=True, exist_ok=True)

    # Loop detection — track recent action signatures to detect stuck agents
    recent_actions: list[str] = []  # rolling window of action hashes
    LOOP_WINDOW = 10
    LOOP_THRESHOLD = 3  # same action 3x in window = stuck

    def _action_signature(tool_name: str, tool_input: dict) -> str:
        """Hash an action for loop detection. Normalize inputs."""
        key_parts = [tool_name]
        if tool_name == "execute_js":
            # Normalize JS — just first 80 chars
            key_parts.append(str(tool_input.get("expression", ""))[:80])
        elif tool_name == "navigate":
            key_parts.append(tool_input.get("url", ""))
        elif tool_name == "click":
            key_parts.append(str(tool_input.get("element_id", "")))
        elif tool_name == "http_request":
            key_parts.append(tool_input.get("url", "")[:100])
        return "|".join(key_parts)

    def _check_loop() -> str | None:
        """Returns a nudge message if agent is stuck, else None."""
        if len(recent_actions) < LOOP_THRESHOLD:
            return None
        window = recent_actions[-LOOP_WINDOW:]
        from collections import Counter
        counts = Counter(window)
        most_common, count = counts.most_common(1)[0]
        if count >= LOOP_THRESHOLD:
            tool_name = most_common.split("|")[0]
            return (
                f"WARNING: You have repeated the same {tool_name} action {count} times "
                f"in the last {len(window)} steps. You are stuck in a loop. "
                f"STOP this approach immediately and try something completely different. "
                f"Consider: use a different tool, navigate to a different URL, "
                f"use http_request to fetch raw content, or use run_python to process "
                f"data you already have."
            )
        return None

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
            "model": model,
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
        iter_start = time.time()
        step: dict[str, Any] = {
            "iteration": iteration + 1,
            "timestamp": round(time.time() - start_time, 2),
        }

        try:
            response = client.beta.messages.create(
                betas=[COMPACTION_BETA],
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                tools=tools,
                messages=messages,
                context_management={
                    "edits": [{
                        "type": "compact_20260112",
                        "trigger": {"type": "input_tokens", "value": COMPACTION_TRIGGER_TOKENS},
                    }]
                },
            )
        except anthropic.APIError as e:
            step["error"] = str(e)
            trace.append(step)
            _save_trace()
            _save_audit()
            if telemetry and run_id:
                await telemetry.log_step_event(
                    run_id=run_id,
                    sample_id=resolved_sample_id,
                    phase=phase,
                    iteration=iteration + 1,
                    result_type="api_error",
                    success=False,
                    error_code="anthropic_api_error",
                    latency_ms=int((time.time() - iter_start) * 1000),
                    payload={"error": str(e)[:500]},
                )
            raise

        api_calls += 1
        # Track tokens across all iterations (including compaction)
        if hasattr(response.usage, "iterations") and response.usage.iterations:
            for it in response.usage.iterations:
                total_input_tokens += it.input_tokens
                total_output_tokens += it.output_tokens
        else:
            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens
        step["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }

        assistant_content = response.content
        messages.append({"role": "assistant", "content": assistant_content})

        # Log compaction events
        for block in assistant_content:
            if hasattr(block, "type") and block.type == "compaction":
                logger.info("  Context compacted — old messages summarized")
                step["compaction"] = True

        # Extract reasoning text from the response
        reasoning = []
        for block in assistant_content:
            if hasattr(block, "type") and block.type == "text":
                reasoning.append(block.text)
        if reasoning:
            step["reasoning"] = "\n".join(reasoning)

        tool_uses = [block for block in assistant_content if hasattr(block, "type") and block.type == "tool_use"]

        if not tool_uses:
            step["action"] = "no_tool_call"
            step["stop_reason"] = response.stop_reason
            trace.append(step)
            _save_trace()
            _save_audit()
            if telemetry and run_id:
                await telemetry.log_step_event(
                    run_id=run_id,
                    sample_id=resolved_sample_id,
                    phase=phase,
                    iteration=iteration + 1,
                    result_type="no_tool_call",
                    success=response.stop_reason in {"compaction", "end_turn"},
                    error_code=None if response.stop_reason in {"compaction", "end_turn"} else "no_tool_call",
                    latency_ms=int((time.time() - iter_start) * 1000),
                    payload={"stop_reason": response.stop_reason},
                )

            if response.stop_reason == "end_turn":
                logger.warning("  Agent ended without calling complete tool. Prompting to complete.")
                messages.append({
                    "role": "user",
                    "content": "You must call the `complete` tool with the extracted data. Please do so now.",
                })
                continue
            # Compaction stop reason — just continue the loop
            if response.stop_reason == "compaction":
                continue
            break

        tool_results = []
        step_tools: list[dict[str, Any]] = []

        for tool_use in tool_uses:
            tool_name = tool_use.name
            tool_input = tool_use.input
            tool_id = tool_use.id
            tool_start = time.time()

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
                if telemetry and run_id:
                    await telemetry.log_tool_event(
                        run_id=run_id,
                        sample_id=resolved_sample_id,
                        phase=phase,
                        iteration=iteration + 1,
                        tool_name=tool_name,
                        tool_input_hash=hash_tool_input(tool_input),
                        result_type="complete",
                        success=True,
                        error_code=None,
                        latency_ms=int((time.time() - tool_start) * 1000),
                        payload={"notes": notes[:500], "keys": list(data.keys())},
                    )
                    await telemetry.log_step_event(
                        run_id=run_id,
                        sample_id=resolved_sample_id,
                        phase=phase,
                        iteration=iteration + 1,
                        result_type="complete",
                        success=True,
                        error_code=None,
                        latency_ms=int((time.time() - iter_start) * 1000),
                        payload={"stop_reason": "complete"},
                    )

                # No auto-screenshot on complete — agent takes them only when needed
                return data

            # Track action for loop detection
            recent_actions.append(_action_signature(tool_name, tool_input))
            if len(recent_actions) > LOOP_WINDOW:
                recent_actions.pop(0)

            # Dispatch to the registry
            try:
                result = await registry.execute(tool_name, tool_input)
            except Exception as e:
                result = f"Error executing {tool_name}: {e}"
                logger.error(f"  {result}")

            result_type, success, error_code = _classify_tool_result(tool_name, result)
            if telemetry and run_id:
                await telemetry.log_tool_event(
                    run_id=run_id,
                    sample_id=resolved_sample_id,
                    phase=phase,
                    iteration=iteration + 1,
                    tool_name=tool_name,
                    tool_input_hash=hash_tool_input(tool_input),
                    result_type=result_type,
                    success=success,
                    error_code=error_code,
                    latency_ms=int((time.time() - tool_start) * 1000),
                    payload={"result_preview": result[:500]},
                )

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

        # Inject nudges: loop detection + budget warning
        nudges = []
        loop_nudge = _check_loop()
        if loop_nudge:
            nudges.append(loop_nudge)
            logger.warning(f"  Loop detected — injecting nudge")
            step["loop_nudge"] = True

        # Budget warning at 75% of max iterations
        budget_pct = (iteration + 1) / max_iterations
        if budget_pct >= 0.75 and not any(s.get("budget_warning") for s in trace):
            remaining = max_iterations - iteration - 1
            nudges.append(
                f"BUDGET WARNING: You have used {iteration + 1}/{max_iterations} iterations "
                f"({remaining} remaining). Wrap up your current approach and call complete "
                f"with whatever data you have. Partial results are better than no results."
            )
            logger.warning(f"  Budget warning — 75% iterations used")
            step["budget_warning"] = True

        if nudges:
            messages.append({
                "role": "user",
                "content": "\n\n".join(nudges),
            })

        if telemetry and run_id:
            await telemetry.log_step_event(
                run_id=run_id,
                sample_id=resolved_sample_id,
                phase=phase,
                iteration=iteration + 1,
                result_type="tool_iteration",
                success=True,
                error_code=None,
                latency_ms=int((time.time() - iter_start) * 1000),
                payload={
                    "tool_count": len(step_tools),
                    "loop_nudge": bool(step.get("loop_nudge")),
                    "budget_warning": bool(step.get("budget_warning")),
                },
            )

    _save_trace()
    _save_audit()
    if telemetry and run_id:
        await telemetry.log_step_event(
            run_id=run_id,
            sample_id=resolved_sample_id,
            phase=phase,
            iteration=max_iterations,
            result_type="max_iterations_exceeded",
            success=False,
            error_code="max_iterations",
            latency_ms=int((time.time() - start_time) * 1000),
            payload={"max_iterations": max_iterations},
        )
    raise RuntimeError(f"Agent did not complete within {max_iterations} iterations")
