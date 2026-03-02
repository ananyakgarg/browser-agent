"""Orchestrator: plan task via LLM, read CSV, dispatch workers concurrently."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

import os

import anthropic
from playwright.async_api import async_playwright

from auth import find_storage_state, validate_storage_state
from browser_tools import create_browser_provider
from code_analysis import CodeAnalysisProvider
from code_tools import CodeToolProvider
from compiler import compile_results
from config import TaskConfig, TaskSpec
from distill import distill_playbook
from http_tools import HttpToolProvider
from judge import score_trace, llm_judge, pick_winner, format_judgment
from progress import ProgressTracker, SampleStatus
from tool_registry import ToolRegistry
from validator import validate_sample, write_metadata
from skills import get_skill_registry, log_skill_selection, log_skill_outcome
from worker import run_worker

logger = logging.getLogger(__name__)

PLAN_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Planning call — replaces the JSON task spec
# ---------------------------------------------------------------------------

_PLAN_SYSTEM_BASE = """\
You are a planning assistant for a browser automation system. You split work \
into two phases:

## Phase 1: RESOLVE (optional — only http_request and run_python)
Use resolve ONLY for simple lookups that make the browser phase faster:
- Looking up a specific URL, file path, or ID via a well-known API
- Resolving a CIK number, converting an identifier

Keep resolve MINIMAL. Do NOT put core task logic here. If resolve fails, \
the browser agent must still be able to complete the task on its own.

The resolve agent calls complete with resolved key-value pairs. These become \
{resolved_key} template variables in the browser instructions.

## Phase 2: BROWSER (the main agent — full browser with all tools)
The browser agent does the actual work. It has: browser navigation, \
screenshots, clicks, forms, execute_js, search_page, http_request, and \
run_python. It can do EVERYTHING.

IMPORTANT rules for browser instructions:
- The browser agent MUST navigate to real web pages and interact with them. \
It should NOT fetch HTML via http_request and render it locally.
- Screenshots MUST be of actual live web pages, not locally-generated HTML.
- If the task says "go to a website" or "take a screenshot", that means \
the real website in the browser, not an API response rendered to HTML.
- The browser agent has a real Chrome user-agent and can access most sites.

## Your decisions
1. sample_column — which CSV column **uniquely** identifies each sample. \
Every value in this column MUST be distinct across all rows. If no single \
column has unique values, prefer a column like 'sample_id' or 'id'. \
NEVER pick a column with duplicate values (e.g. repo_url when multiple rows \
share the same repo).
2. output_columns — fields to extract (add _evidence columns for any judgments)
3. resolve_instructions — simple lookups to speed up browser work (leave \
empty if the browser can navigate directly)
4. per_sample_instructions — the main task steps using {{column_name}} and \
{{resolved_key}} templates. Include ALL task logic here.

## CRITICAL: Respect the instruction's output fields
If the instruction specifies "Output fields:" or lists required fields, you \
MUST include ALL of them in output_columns (use the exact names given). \
Do NOT rename them, do NOT drop any, do NOT substitute your own column names. \
You may ADD extra columns (like _evidence companions) but never remove any \
that the instruction explicitly lists.

## CRITICAL: Preserve ALL task logic
The per_sample_instructions MUST include EVERY step, conditional branch, and \
decision point from the original instruction. If the instruction says \
"if X then do Y, else do Z", that MUST appear in per_sample_instructions. \
Do NOT simplify, summarize, or drop any logic. The browser agent needs the \
complete task description to do its job correctly.

## Grounding rule
For any output column requiring judgment (ratings, flags, yes/no), add a \
companion _evidence column requiring verbatim extracted data from the source.
"""

_PLAN_SKILL_SECTION = """
## Skill selection
Select which skills the worker agents should use.  Available skills:
{skill_descriptions}

Guidelines:
- Select **code_analysis** when the task involves inspecting source code, tracing \
dependencies, reading git history, or assessing code changes.
- Select **browser_navigation** when the task requires interacting with web pages \
(clicking, filling forms, taking screenshots, reading page content).
- Select **data_extraction** when the primary goal is pulling structured data from \
APIs or web pages without heavy browser interaction.
- You may select MULTIPLE skills — their tools and guidance are combined.
- Set confidence 0.0–1.0 reflecting how central the skill is to the task.
- If unsure, select the skills that seem most relevant — extra tools don't hurt much, \
but missing a key skill means the worker lacks critical guidance.

Include your skill selections in the `required_skills` field of the task_plan tool call.
"""


def _build_plan_system(use_skills: bool) -> str:
    """Build the PLAN_SYSTEM prompt, optionally with skill selection."""
    base = _PLAN_SYSTEM_BASE
    if use_skills:
        registry = get_skill_registry()
        base += _PLAN_SKILL_SECTION.format(
            skill_descriptions=registry.describe_for_planner(),
        )
    base += "\nCall the task_plan tool with your decisions."
    return base

# Keep a module-level constant for backward compat (no-skills path)
PLAN_SYSTEM = _build_plan_system(use_skills=False)

PLAN_TOOL = {
    "name": "task_plan",
    "description": "Output the structured task plan based on the user's instruction and CSV.",
    "input_schema": {
        "type": "object",
        "properties": {
            "task_name": {
                "type": "string",
                "description": "Short snake_case name for the output folder (e.g. 'jira_ticket_scrape').",
            },
            "sample_column": {
                "type": "string",
                "description": "The CSV column that UNIQUELY identifies each sample. Must have distinct values for every row.",
            },
            "output_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "ALL output fields per sample. If the instruction lists 'Output fields:' "
                "use those EXACT names. Then add a column for every judgment/decision the instruction "
                "asks for, plus _evidence companion columns for any judgment columns. "
                "NEVER drop fields that the instruction explicitly requests.",
            },
            "resolve_instructions": {
                "type": "string",
                "description": "Instructions for the RESOLVE phase (http_request + run_python only, NO browser). "
                "Must call complete with resolved key-value pairs. These become {key} templates "
                "in per_sample_instructions. Use {column_name} for CSV values. "
                "Example: 'Use the GitHub API to find which file contains {code_string}. "
                "Fetch the raw file and find the line number. Call complete with "
                "file_path, line_number, and blame_url.' "
                "Leave empty ONLY if the CSV already contains direct URLs and no lookup is needed.",
            },
            "per_sample_instructions": {
                "type": "string",
                "description": "The COMPLETE task instructions for the browser agent. Use {column_name} and "
                "{resolved_key} templates. Include ALL logic from the user's instruction — every step, "
                "every conditional branch, every judgment call. The browser agent is intelligent and "
                "can reason, analyze diffs, assess materiality, and make decisions. Do NOT simplify "
                "or drop any steps from the original instruction.",
            },
            "required_skills": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                    "required": ["name", "confidence"],
                },
                "description": "Skills the worker agents should use (only when skill selection is enabled).",
            },
        },
        "required": ["task_name", "sample_column", "output_columns", "resolve_instructions", "per_sample_instructions"],
    },
}


def _csv_preview(csv_path: str, max_rows: int = 3) -> tuple[list[str], list[dict[str, str]]]:
    """Read CSV headers and first N rows for the planning prompt."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        headers = list(reader.fieldnames or [])
        rows = []
        for i, row in enumerate(reader):
            if i >= max_rows:
                break
            rows.append(row)
    return headers, rows


def _extract_output_fields(instruction: str) -> list[str]:
    """Extract explicit output field names from an instruction like 'Output fields: a, b, c'."""
    for line in instruction.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("output fields:") or stripped.lower().startswith("output_fields:"):
            fields_str = stripped.split(":", 1)[1].strip()
            return [f.strip() for f in fields_str.split(",") if f.strip()]
    return []


def plan_task(instruction: str, csv_path: str, use_skills: bool = False) -> dict[str, Any]:
    """Make one LLM call to turn a natural-language instruction into a structured plan."""
    # If instruction is a file path, read its contents
    if Path(instruction).is_file():
        instruction = Path(instruction).read_text(encoding="utf-8")
    headers, preview_rows = _csv_preview(csv_path)

    # Extract explicit output fields if the instruction lists them
    output_fields_hint = ""
    for line in instruction.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("output fields:") or stripped.lower().startswith("output_fields:"):
            fields_str = stripped.split(":", 1)[1].strip()
            output_fields_hint = (
                f"\n\n## REQUIRED output fields (from instruction)\n"
                f"The instruction explicitly lists these output fields: {fields_str}\n"
                f"You MUST include ALL of these as output_columns (use exact names). "
                f"You may add _evidence companion columns but NEVER drop or rename these fields."
            )
            break

    user_msg = (
        f"## User instruction\n{instruction}\n\n"
        f"## CSV columns\n{headers}\n\n"
        f"## First rows\n```json\n{json.dumps(preview_rows, indent=2)}\n```"
        f"{output_fields_hint}"
    )

    plan_system = _build_plan_system(use_skills=use_skills)
    client = anthropic.Anthropic(max_retries=10)
    response = client.messages.create(
        model=PLAN_MODEL,
        max_tokens=8192,
        system=plan_system,
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "task_plan"},
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "tool_use" and block.name == "task_plan":
            plan = block.input
            logger.info(f"Plan: {json.dumps(plan, indent=2)}")
            # Validate sample_column exists in CSV
            if plan["sample_column"] not in headers:
                raise ValueError(
                    f"Planning call chose sample_column='{plan['sample_column']}' "
                    f"but CSV columns are: {headers}"
                )
            # Validate sample_column has unique values
            col = plan["sample_column"]
            values = [r[col] for r in csv.DictReader(open(csv_path, encoding="utf-8-sig"))]
            if len(values) != len(set(values)):
                dupes = [v for v in set(values) if values.count(v) > 1]
                raise ValueError(
                    f"sample_column='{col}' has duplicate values: {dupes}. "
                    f"Choose a column with unique values per row. Available: {headers}"
                )
            # Ensure required keys exist (model may truncate on long resolve_instructions)
            if "per_sample_instructions" not in plan:
                raise RuntimeError(
                    "Planning call truncated — missing per_sample_instructions. "
                    "This usually means resolve_instructions was too long. "
                    f"Plan keys received: {list(plan.keys())}"
                )
            # Enforce explicit output fields from instruction
            required_fields = _extract_output_fields(instruction)
            if required_fields:
                existing = set(plan.get("output_columns", []))
                missing = [f for f in required_fields if f not in existing]
                if missing:
                    logger.warning(
                        f"Planner dropped required output fields: {missing}. Adding them."
                    )
                    plan["output_columns"] = list(plan.get("output_columns", [])) + missing
            # Append original instruction to per_sample_instructions so no logic is lost
            plan["per_sample_instructions"] += (
                "\n\n---\n## ORIGINAL TASK INSTRUCTION (for reference — follow ALL steps)\n"
                + instruction
            )
            return plan

    raise RuntimeError("Planning call did not return a task_plan tool use")


# ---------------------------------------------------------------------------
# Resolve phase — non-browser preprocessing per sample
# ---------------------------------------------------------------------------

RESOLVE_SYSTEM = """\
You are a research assistant. You resolve data using HTTP requests and code \
execution BEFORE a browser agent takes over. You do NOT have a browser.

Your tools: http_request (GET/POST any URL) and run_python (execute Python code).

Guidelines:
- Use http_request to call APIs, fetch raw files, check URLs.
- Use run_python to parse responses, search text, compute values.
- Be efficient — resolve everything in as few calls as possible.
- When done, call complete with all resolved values as key-value pairs.
- The browser agent will use your resolved values as template variables.
"""


async def resolve_sample(
    spec: TaskSpec,
    row: dict[str, Any],
    model_override: str | None = None,
) -> dict[str, Any]:
    """Run a non-browser agent to resolve data before browser work.

    Returns a dict of resolved key-value pairs to inject into browser instructions.
    """
    if not spec.resolve_instructions:
        return {}

    registry = ToolRegistry()
    registry.register(CodeToolProvider())
    registry.register(HttpToolProvider())

    instructions = spec.resolve_instructions
    for key, value in row.items():
        instructions = instructions.replace(f"{{{key}}}", str(value))

    sample_id = str(row.get(spec.sample_id_column, "unknown"))
    resolve_dir = spec.sample_dir(sample_id) / "resolve"

    try:
        data = await asyncio.wait_for(
            run_worker(
                registry=registry,
                instructions=instructions,
                csv_columns=[],  # resolve doesn't have output columns
                row=row,
                output_dir=resolve_dir,
                max_iterations=10,
                model_override=model_override,
                system_prompt_override=RESOLVE_SYSTEM,
            ),
            timeout=120,  # resolve should be fast
        )
        logger.info(f"  Resolved: {list(data.keys())}")
        return data
    except Exception as e:
        logger.warning(f"  Resolve failed: {e}. Browser agent will proceed without resolved data.")
        return {}
    finally:
        await registry.close()


# ---------------------------------------------------------------------------
# CSV reading
# ---------------------------------------------------------------------------

def read_csv_rows(csv_path: str, sample_id_column: str) -> list[dict[str, Any]]:
    """Read CSV and return list of row dicts. Validates sample_id_column exists."""
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header row: {csv_path}")
        if sample_id_column not in reader.fieldnames:
            raise ValueError(
                f"Column '{sample_id_column}' not found in CSV. "
                f"Available: {reader.fieldnames}"
            )
        return list(reader)


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

async def process_sample(
    spec: TaskSpec,
    row: dict[str, Any],
    browser,
    progress: ProgressTracker,
    semaphore: asyncio.Semaphore,
    model_override: str | None = None,
    pioneer_mode: bool = False,
    playbook: str | None = None,
    pw=None,
    browserbase: bool = False,
) -> bool | dict[str, Any]:
    """Process a single sample.

    Returns True on success, False on failure.
    If pioneer_mode=True, returns the extracted data dict (including playbook) on success.
    """
    sample_id = str(row[spec.sample_id_column])
    sample_dir = spec.sample_dir(sample_id)

    async with semaphore:
        progress.set_status(sample_id, SampleStatus.IN_PROGRESS)
        logger.info(f"Processing sample: {sample_id}" + (" [PIONEER]" if pioneer_mode else ""))

        # Resolve phase — run before browser work
        resolved = await resolve_sample(spec, row, model_override=model_override)

        # Merge resolved data into row for template rendering
        enriched_row = {**row, **resolved}

        registry = ToolRegistry()
        bb_browser = None
        bb_session_id = None
        try:
            if browserbase and pw:
                from browserbase import Browserbase
                bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
                session = bb.sessions.create(
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                )
                bb_session_id = session.id
                logger.info(f"  Browserbase session: https://browserbase.com/sessions/{session.id}")
                bb_browser = await pw.chromium.connect_over_cdp(session.connect_url)
                context = bb_browser.contexts[0]
                browser_provider = await create_browser_provider(
                    context=context,
                    output_dir=sample_dir,
                )
            else:
                browser_provider = await create_browser_provider(
                    browser=browser,
                    output_dir=sample_dir,
                    storage_state_path=spec.storage_state_path,
                )
            registry.register(browser_provider)
            registry.register(CodeToolProvider())
            registry.register(CodeAnalysisProvider())
            http_provider = HttpToolProvider()
            registry.register(http_provider)

            instructions = spec.render_instructions(enriched_row)

            # Resolve skills → tool filtering + prompt injection
            skill_prompt_fragments: str | None = None
            tool_allowlist: set[str] | None = None
            if spec.required_skills:
                skill_registry = get_skill_registry()
                resolved_skills = skill_registry.resolve_skills(spec.required_skills)
                if resolved_skills:
                    skill_prompt_fragments = skill_registry.build_prompt_fragments(resolved_skills)
                    tool_allowlist = skill_registry.build_tool_allowlist(resolved_skills)
                    log_skill_selection(spec.output_dir, sample_id, spec.required_skills, resolved_skills)

            data = await asyncio.wait_for(
                run_worker(
                    registry=registry,
                    instructions=instructions,
                    csv_columns=spec.csv_columns,
                    row=row,
                    output_dir=sample_dir,
                    max_iterations=spec.config.max_iterations,
                    model_override=model_override,
                    pioneer_mode=pioneer_mode,
                    playbook=playbook,
                    skill_prompt_fragments=skill_prompt_fragments,
                    tool_allowlist=tool_allowlist,
                ),
                timeout=spec.config.timeout_per_sample_sec,
            )

            # Log skill outcome
            if spec.required_skills and resolved_skills:
                log_skill_outcome(
                    spec.output_dir, sample_id,
                    skills_used=[s.name for s in resolved_skills],
                    success=True,
                )

            # Write metadata with playbook flag
            metadata_extra = {"playbook_used": playbook is not None}
            if bb_session_id:
                metadata_extra["browserbase_session"] = f"https://browserbase.com/sessions/{bb_session_id}"
            write_metadata(sample_dir, sample_id, data, row, extra=metadata_extra)

            errors = validate_sample(sample_dir, spec.csv_columns, data)
            if errors:
                logger.warning(f"  Validation warnings for {sample_id}: {errors}")

            progress.set_status(sample_id, SampleStatus.COMPLETED)
            logger.info(f"  Completed: {sample_id}")

            if pioneer_mode:
                return data  # includes playbook field
            return True

        except asyncio.TimeoutError:
            msg = f"Timeout after {spec.config.timeout_per_sample_sec}s"
            logger.error(f"  {msg} for {sample_id}")
            progress.set_status(sample_id, SampleStatus.FAILED, error=msg)
            return False

        except Exception as e:
            msg = str(e)[:500]
            logger.error(f"  Error for {sample_id}: {msg}")
            progress.set_status(sample_id, SampleStatus.FAILED, error=msg)
            return False

        finally:
            await registry.close()
            if bb_browser:
                await bb_browser.close()


# ---------------------------------------------------------------------------
# Pioneer tournament — run N pioneers, judge, distill best into playbook
# ---------------------------------------------------------------------------

async def run_pioneer_tournament(
    spec: TaskSpec,
    pioneer_row: dict[str, Any],
    pioneer_id: str,
    browser,
    progress: ProgressTracker,
    semaphore: asyncio.Semaphore,
    num_pioneers: int,
    model_override: str | None = None,
    pw=None,
    browserbase: bool = False,
) -> str | None:
    """Run N pioneers concurrently on the same sample, judge, distill winner.

    Returns the distilled playbook text, or None if all pioneers failed.
    """
    logger.info(f"Pioneer tournament: launching {num_pioneers} pioneers on {pioneer_id}")

    # Resolve once for all pioneers (same sample, same resolved data)
    resolved = await resolve_sample(spec, pioneer_row, model_override=model_override)
    enriched_row = {**pioneer_row, **resolved}
    instructions = spec.render_instructions(enriched_row)

    # Log skill selection for this tournament sample
    if spec.required_skills:
        skill_registry = get_skill_registry()
        tournament_resolved = skill_registry.resolve_skills(spec.required_skills)
        if tournament_resolved:
            log_skill_selection(
                spec.output_dir, pioneer_id,
                spec.required_skills, tournament_resolved,
            )

    # Each pioneer gets its own output dir
    pioneer_dirs = []
    for i in range(num_pioneers):
        d = spec.sample_dir(pioneer_id) / f"pioneer_{i}"
        d.mkdir(parents=True, exist_ok=True)
        pioneer_dirs.append(d)

    # Launch all pioneers concurrently
    async def _run_one_pioneer(idx: int) -> dict[str, Any] | None:
        """Run a single pioneer. Returns result data or None on failure."""
        p_dir = pioneer_dirs[idx]
        logger.info(f"  Pioneer {idx} starting...")

        registry = ToolRegistry()
        bb_browser = None
        try:
            if browserbase and pw:
                from browserbase import Browserbase
                bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
                session = bb.sessions.create(
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                )
                logger.info(f"  Pioneer {idx} Browserbase: https://browserbase.com/sessions/{session.id}")
                bb_browser = await pw.chromium.connect_over_cdp(session.connect_url)
                context = bb_browser.contexts[0]
                browser_provider = await create_browser_provider(
                    context=context, output_dir=p_dir,
                )
            else:
                browser_provider = await create_browser_provider(
                    browser=browser, output_dir=p_dir,
                    storage_state_path=spec.storage_state_path,
                )
            registry.register(browser_provider)
            registry.register(CodeToolProvider())
            registry.register(CodeAnalysisProvider())
            registry.register(HttpToolProvider())

            # Resolve skills for pioneer
            p_skill_fragments: str | None = None
            p_tool_allowlist: set[str] | None = None
            if spec.required_skills:
                skill_registry = get_skill_registry()
                p_resolved = skill_registry.resolve_skills(spec.required_skills)
                if p_resolved:
                    p_skill_fragments = skill_registry.build_prompt_fragments(p_resolved)
                    p_tool_allowlist = skill_registry.build_tool_allowlist(p_resolved)

            data = await asyncio.wait_for(
                run_worker(
                    registry=registry,
                    instructions=instructions,
                    csv_columns=spec.csv_columns,
                    row=pioneer_row,
                    output_dir=p_dir,
                    max_iterations=spec.config.max_iterations,
                    model_override=model_override,
                    pioneer_mode=True,
                    skill_prompt_fragments=p_skill_fragments,
                    tool_allowlist=p_tool_allowlist,
                ),
                timeout=spec.config.timeout_per_sample_sec,
            )
            logger.info(f"  Pioneer {idx} completed successfully")
            return data

        except Exception as e:
            logger.warning(f"  Pioneer {idx} failed: {str(e)[:200]}")
            return None
        finally:
            await registry.close()
            if bb_browser:
                await bb_browser.close()

    # Run all pioneers concurrently
    pioneer_tasks = [_run_one_pioneer(i) for i in range(num_pioneers)]
    results = await asyncio.gather(*pioneer_tasks, return_exceptions=True)

    # Score each pioneer with LLM judge
    candidates = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            result_data = None
        else:
            result_data = result

        trace_path = pioneer_dirs[i] / "trace.json"
        audit_path = pioneer_dirs[i] / "audit.json"
        score = llm_judge(
            trace_path, audit_path, result_data, spec.csv_columns,
            task_instructions=spec.per_sample_instructions,
        )

        # Save judgment to pioneer dir
        judgment_out = pioneer_dirs[i] / "judgment.json"
        with open(judgment_out, "w") as f:
            json.dump(score, f, indent=2)

        candidates.append({
            "pioneer_id": i,
            "score": score,
            "result": result_data,
            "dir": pioneer_dirs[i],
        })

    # Judge picks winner
    winner_idx = pick_winner(candidates)
    judgment = format_judgment(candidates, winner_idx)
    logger.info(f"\n{judgment}")

    # Save judgment to output dir
    judgment_path = spec.sample_dir(pioneer_id) / "tournament.json"
    judgment_data = {
        "num_pioneers": num_pioneers,
        "winner": winner_idx,
        "candidates": [
            {"pioneer_id": c["pioneer_id"], "score": c["score"]}
            for c in candidates
        ],
    }
    with open(judgment_path, "w") as f:
        json.dump(judgment_data, f, indent=2)

    winner = candidates[winner_idx]
    winner_result = winner["result"]

    # Log skill outcome for the tournament
    if spec.required_skills and tournament_resolved:
        log_skill_outcome(
            spec.output_dir, pioneer_id,
            skills_used=[s.name for s in tournament_resolved],
            success=winner_result is not None,
        )

    if winner_result is None:
        logger.warning("All pioneers failed — no playbook available")
        progress.set_status(pioneer_id, SampleStatus.FAILED, error="All pioneers failed")
        return None

    # Write metadata for the winning pioneer as the canonical sample result
    # Copy the winner's result data (minus playbook) as the sample output
    sample_data = {k: v for k, v in winner_result.items() if k != "playbook"}
    write_metadata(spec.sample_dir(pioneer_id), pioneer_id, sample_data, pioneer_row)
    progress.set_status(pioneer_id, SampleStatus.COMPLETED)

    # Distill the winner's trace into a clean playbook
    raw_playbook = winner_result.get("playbook")
    winner_trace_path = winner["dir"] / "trace.json"

    logger.info(f"Distilling playbook from Pioneer {winner_idx}...")
    playbook_text = distill_playbook(
        original_instructions=spec.per_sample_instructions,
        trace_path=winner_trace_path,
        raw_playbook=raw_playbook,
    )

    # Save playbook
    playbook_path = spec.output_dir / "playbook.md"
    playbook_path.write_text(playbook_text)
    logger.info(f"Playbook saved to {playbook_path}")

    return playbook_text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_orchestrator(
    instruction: str,
    csv_path: str,
    max_workers: int = 3,
    output_dir_override: str | None = None,
    session_state_path: str | None = None,
    model_override: str | None = None,
    num_pioneers: int = 1,
    max_iterations: int = 30,
    browserbase: bool = False,
    use_skills: bool = False,
):
    """Plan the task, then dispatch workers."""
    start = time.time()

    # 1. Planning call
    logger.info("Running planning call..." + (" (skills enabled)" if use_skills else ""))
    plan = plan_task(instruction, csv_path, use_skills=use_skills)

    # 2. Build TaskSpec from plan + CLI args
    task_name = plan["task_name"]
    spec = TaskSpec(
        task_name=task_name,
        per_sample_instructions=plan["per_sample_instructions"],
        input_csv=csv_path,
        sample_id_column=plan["sample_column"],
        csv_columns=plan["output_columns"],
        resolve_instructions=plan.get("resolve_instructions", ""),
        required_skills=plan.get("required_skills", []) if use_skills else [],
        config=TaskConfig(max_workers=max_workers, num_pioneers=num_pioneers, max_iterations=max_iterations),
        output_dir_override=output_dir_override,
    )

    logger.info(
        f"Task: {spec.task_name} | "
        f"Sample column: {spec.sample_id_column} | "
        f"Output columns: {spec.csv_columns} | "
        f"Workers: {spec.config.max_workers}"
    )

    # 3. Setup
    import shutil
    free_mb = shutil.disk_usage(Path.cwd()).free // (1024 * 1024)
    if free_mb < 500:
        raise RuntimeError(f"Only {free_mb}MB disk space free. Need at least 500MB to run safely.")
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.samples_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(spec.input_csv, spec.sample_id_column)
    logger.info(f"Loaded {len(rows)} rows from {spec.input_csv}")

    # 3b. Load session state — explicit path or auto-discover
    if session_state_path and Path(session_state_path).exists():
        spec.storage_state_path = session_state_path
        logger.info(f"Using session state from {session_state_path}")
    else:
        auto_path = find_storage_state()
        if auto_path:
            info = validate_storage_state(auto_path)
            if info["valid"]:
                spec.storage_state_path = str(auto_path)
                logger.info(
                    f"Auto-discovered auth state: {auto_path} "
                    f"({info['cookie_count']} cookies, "
                    f"{len(info['domains'])} domains)"
                )
                for w in info["warnings"]:
                    logger.warning(f"  Auth: {w}")

    all_ids = [str(r[spec.sample_id_column]) for r in rows]
    id_to_row = {str(r[spec.sample_id_column]): r for r in rows}

    progress = ProgressTracker(spec.output_dir)
    pending_ids = progress.get_pending_samples(all_ids)
    logger.info(f"Pending samples: {len(pending_ids)} / {len(all_ids)}")

    if not pending_ids:
        logger.info("All samples already completed. Compiling results.")
        results_path = compile_results(spec.output_dir, spec.csv_columns)
        logger.info(f"Results: {results_path}")
        return

    # 4. Dispatch workers (pioneer-follower pattern)
    semaphore = asyncio.Semaphore(spec.config.max_workers)
    playbook_text: str | None = None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        if browserbase:
            logger.info("Browserbase mode — pioneer gets a cloud browser with session replay")

        try:
            # 4a. Pioneer phase — single or tournament
            if spec.config.pioneer_enabled and len(pending_ids) > 1:
                pioneer_id = pending_ids[0]
                pioneer_row = id_to_row[pioneer_id]
                n_pioneers = spec.config.num_pioneers

                if n_pioneers > 1:
                    # Tournament mode — run N pioneers, judge, distill
                    playbook_text = await run_pioneer_tournament(
                        spec, pioneer_row, pioneer_id,
                        browser, progress, semaphore,
                        num_pioneers=n_pioneers,
                        model_override=model_override,
                        pw=pw, browserbase=browserbase,
                    )
                else:
                    # Single pioneer (original behavior)
                    logger.info(f"Pioneer phase: running {pioneer_id}")
                    pioneer_result = await process_sample(
                        spec, pioneer_row, browser, progress, semaphore,
                        model_override=model_override,
                        pioneer_mode=True,
                        pw=pw, browserbase=browserbase,
                    )

                    if isinstance(pioneer_result, dict):
                        # Judge the single pioneer
                        trace_path = spec.sample_dir(pioneer_id) / "trace.json"
                        audit_path = spec.sample_dir(pioneer_id) / "audit.json"
                        judgment = llm_judge(
                            trace_path, audit_path, pioneer_result, spec.csv_columns,
                            task_instructions=spec.per_sample_instructions,
                        )
                        judgment_out = spec.sample_dir(pioneer_id) / "judgment.json"
                        with open(judgment_out, "w") as f:
                            json.dump(judgment, f, indent=2)
                        logger.info(f"Pioneer judged: {judgment.get('overall_score', '?')}/100")
                        if "overall_reasoning" in judgment:
                            logger.info(f"  Reasoning: {judgment['overall_reasoning']}")

                        raw_playbook = pioneer_result.pop("playbook", None)
                        if raw_playbook:
                            logger.info("Pioneer succeeded — distilling playbook...")
                            playbook_text = distill_playbook(
                                original_instructions=spec.per_sample_instructions,
                                trace_path=trace_path,
                                raw_playbook=raw_playbook,
                            )
                            playbook_path = spec.output_dir / "playbook.md"
                            playbook_path.write_text(playbook_text)
                            logger.info(f"Playbook saved to {playbook_path}")
                        else:
                            logger.warning("Pioneer completed but did not include a playbook")
                    else:
                        logger.warning("Pioneer failed — followers will run without playbook")

                # Remove pioneer from pending list
                pending_ids = pending_ids[1:]

            # 4b. Follower phase — dispatch remaining samples in parallel
            if not pending_ids:
                logger.info("No remaining samples after pioneer.")
            else:
                if playbook_text:
                    logger.info(f"Dispatching {len(pending_ids)} followers with playbook")
                else:
                    logger.info(f"Dispatching {len(pending_ids)} workers (no playbook)")

                tasks = []
                for sid in pending_ids:
                    row = id_to_row[sid]

                    async def _process_with_retries(
                        sample_row=row, sample_id=sid, pb=playbook_text
                    ):
                        max_retries = spec.config.max_retries
                        for attempt in range(max_retries + 1):
                            if attempt > 0:
                                logger.info(f"  Retry {attempt}/{max_retries} for {sample_id}")
                            result = await process_sample(
                                spec, sample_row, browser, progress, semaphore,
                                model_override=model_override,
                                playbook=pb,
                            )
                            if result is True or isinstance(result, dict):
                                return True
                            if attempt < max_retries:
                                progress.set_status(sample_id, SampleStatus.PENDING)
                        return False

                    tasks.append(_process_with_retries())

                results = await asyncio.gather(*tasks, return_exceptions=True)

                succeeded = sum(1 for r in results if r is True)
                failed = len(results) - succeeded
                logger.info(f"Done: {succeeded} succeeded, {failed} failed")

        finally:
            await browser.close()

    # 5. Compile results
    results_path = compile_results(spec.output_dir, spec.csv_columns)

    elapsed = time.time() - start
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Results written to: {results_path}")
    logger.info(f"Progress: {json.dumps(progress.summary())}")
