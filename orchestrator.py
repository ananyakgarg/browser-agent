"""Orchestrator: parse task spec, read CSV, dispatch workers concurrently."""

from __future__ import annotations

import asyncio
import csv
import json
import logging
import time
from pathlib import Path
from typing import Any

from playwright.async_api import async_playwright

from browser_tools import create_browser_provider
from compiler import compile_results
from config import TaskSpec
from progress import ProgressTracker, SampleStatus
from tool_registry import ToolRegistry
from validator import validate_sample, write_metadata
from worker import run_worker

logger = logging.getLogger(__name__)


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
            # Create browser provider and register it
            browser_provider = await create_browser_provider(
                browser,
                output_dir=sample_dir,
                cookies_path=spec.auth.cookies_path,
            )
            registry.register(browser_provider)
            # Future: registry.register(CodeExecProvider(...))
            # Future: registry.register(ApiCallProvider(...))

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


async def run_orchestrator(spec: TaskSpec):
    """Main orchestration loop."""
    start = time.time()

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

    results_path = compile_results(spec.output_dir, spec.csv_columns)

    elapsed = time.time() - start
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info(f"Results written to: {results_path}")
    logger.info(f"Progress: {json.dumps(progress.summary())}")
