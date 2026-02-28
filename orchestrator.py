"""Orchestrator: plan task via LLM, read CSV, dispatch workers concurrently."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from typing import Any

import anthropic
from playwright.async_api import async_playwright

from browser_tools import create_browser_provider
from compiler import compile_results
from config import TaskConfig, TaskSpec
from progress import ProgressTracker, SampleStatus
from tool_registry import ToolRegistry
from validator import validate_sample, write_metadata
from worker import run_worker

logger = logging.getLogger(__name__)

PLAN_MODEL = "claude-sonnet-4-5-20250929"

# ---------------------------------------------------------------------------
# Planning call — replaces the JSON task spec
# ---------------------------------------------------------------------------

PLAN_SYSTEM = """\
You are a planning assistant for a browser automation system. Given a user's \
natural-language instruction and a preview of their CSV data, you must decide:

1. Which CSV column identifies each sample (the one with URLs, ticket numbers, \
names, or other unique identifiers that the browser agent will act on).
2. What output fields the user wants extracted.
3. Clear, actionable per-sample instructions for a browser agent. Use \
{column_name} template variables so the agent gets the actual value from \
each CSV row at runtime.
4. A short snake_case task name for the output folder.

Call the task_plan tool with your decisions. Do NOT add output columns the \
user didn't ask for. Keep per_sample_instructions concise and actionable.\
"""

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
                "description": "The CSV column that identifies each sample.",
            },
            "output_columns": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of field names to extract per sample.",
            },
            "per_sample_instructions": {
                "type": "string",
                "description": "Instructions for the browser agent per sample. Use {column_name} templates.",
            },
        },
        "required": ["task_name", "sample_column", "output_columns", "per_sample_instructions"],
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


def plan_task(instruction: str, csv_path: str) -> dict[str, Any]:
    """Make one LLM call to turn a natural-language instruction into a structured plan."""
    headers, preview_rows = _csv_preview(csv_path)

    user_msg = (
        f"## User instruction\n{instruction}\n\n"
        f"## CSV columns\n{headers}\n\n"
        f"## First rows\n```json\n{json.dumps(preview_rows, indent=2)}\n```"
    )

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=PLAN_MODEL,
        max_tokens=1024,
        system=PLAN_SYSTEM,
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
            return plan

    raise RuntimeError("Planning call did not return a task_plan tool use")


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
) -> bool:
    """Process a single sample. Returns True on success."""
    sample_id = str(row[spec.sample_id_column])
    sample_dir = spec.sample_dir(sample_id)

    async with semaphore:
        progress.set_status(sample_id, SampleStatus.IN_PROGRESS)
        logger.info(f"Processing sample: {sample_id}")

        registry = ToolRegistry()
        try:
            browser_provider = await create_browser_provider(
                browser,
                output_dir=sample_dir,
                cookies_path=spec.cookies_path,
            )
            registry.register(browser_provider)

            instructions = spec.render_instructions(row)

            data = await asyncio.wait_for(
                run_worker(
                    registry=registry,
                    instructions=instructions,
                    csv_columns=spec.csv_columns,
                    row=row,
                    output_dir=sample_dir,
                    max_iterations=spec.config.max_iterations,
                ),
                timeout=spec.config.timeout_per_sample_sec,
            )

            write_metadata(sample_dir, sample_id, data, row)

            errors = validate_sample(sample_dir, spec.csv_columns, data)
            if errors:
                logger.warning(f"  Validation warnings for {sample_id}: {errors}")

            progress.set_status(sample_id, SampleStatus.COMPLETED)
            logger.info(f"  Completed: {sample_id}")
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


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def run_orchestrator(
    instruction: str,
    csv_path: str,
    cookies_path: str | None = None,
    max_workers: int = 3,
    output_dir_override: str | None = None,
):
    """Plan the task, then dispatch workers."""
    start = time.time()

    # 1. Planning call
    logger.info("Running planning call...")
    plan = plan_task(instruction, csv_path)

    # 2. Build TaskSpec from plan + CLI args
    task_name = plan["task_name"]
    spec = TaskSpec(
        task_name=task_name,
        per_sample_instructions=plan["per_sample_instructions"],
        input_csv=csv_path,
        sample_id_column=plan["sample_column"],
        csv_columns=plan["output_columns"],
        config=TaskConfig(max_workers=max_workers),
        cookies_path=cookies_path,
        output_dir_override=output_dir_override,
    )

    logger.info(
        f"Task: {spec.task_name} | "
        f"Sample column: {spec.sample_id_column} | "
        f"Output columns: {spec.csv_columns} | "
        f"Workers: {spec.config.max_workers}"
    )

    # 3. Setup
    spec.output_dir.mkdir(parents=True, exist_ok=True)
    spec.samples_dir.mkdir(parents=True, exist_ok=True)

    rows = read_csv_rows(spec.input_csv, spec.sample_id_column)
    logger.info(f"Loaded {len(rows)} rows from {spec.input_csv}")

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

    # 4. Dispatch workers
    semaphore = asyncio.Semaphore(spec.config.max_workers)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)

        try:
            tasks = []
            for sid in pending_ids:
                row = id_to_row[sid]

                async def _process_with_retries(
                    sample_row=row, sample_id=sid
                ):
                    max_retries = spec.config.max_retries
                    for attempt in range(max_retries + 1):
                        if attempt > 0:
                            logger.info(f"  Retry {attempt}/{max_retries} for {sample_id}")
                        success = await process_sample(
                            spec, sample_row, browser, progress, semaphore
                        )
                        if success:
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
