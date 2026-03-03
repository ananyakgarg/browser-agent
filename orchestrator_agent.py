"""Agentic orchestrator: an LLM agent loop that plans, dispatches, inspects, and adapts.

Gated behind --agent flag. When active, replaces the fixed pipeline in orchestrator.py
with a Claude agent that reasons about what to do at each step.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from pathlib import Path
from typing import Any

import anthropic
from playwright.async_api import async_playwright

from auth import worker_copy, cleanup_worker
from browser_tools import create_browser_provider
from code_analysis import CodeAnalysisProvider
from code_tools import CodeToolProvider
import csv

from compiler import compile_results, append_result_row
from config import TaskConfig, TaskSpec
from distill import build_trace_text, distill_playbook
from http_tools import HttpToolProvider
from judge import score_trace, pick_winner, format_judgment
from observability import generate_failure_report, EventEmittingRegistry, EventCallback, WorkerEvent
from orchestrator import plan_task, read_csv_rows, resolve_sample, RESOLVE_SYSTEM
from progress import ProgressTracker, SampleStatus
from tool_registry import ToolProvider, ToolRegistry
from validator import validate_sample, write_metadata
from worker import run_worker

logger = logging.getLogger(__name__)

# Well-known websites that users might mention in instructions
_KNOWN_SITES = [
    "linkedin", "google", "amazon", "twitter", "x.com", "facebook", "instagram",
    "github", "reddit", "youtube", "craigslist", "zillow", "glassdoor", "indeed",
    "yelp", "tripadvisor", "ebay", "walmart", "target", "etsy",
]


def _extract_mentioned_sites(instruction: str) -> list[str]:
    """Extract explicitly mentioned website names from the instruction."""
    lower = instruction.lower()
    found = []
    for site in _KNOWN_SITES:
        if site in lower:
            # Capitalize nicely for display
            found.append(site.capitalize() if site != "x.com" else "X.com")
    return found


MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 4096
MAX_ITERATIONS = 50
COMPACTION_TRIGGER_TOKENS = 80_000

ORCHESTRATOR_SYSTEM = """\
You are an orchestrator agent for a browser automation system. Your job is to accomplish \
the user's instruction by deploying browser worker agents however you see fit.

## CRITICAL: Follow the instruction literally

When the user names a specific website (LinkedIn, Google, Amazon, etc.), that website is your \
PRIMARY and ONLY target. Send your agents DIRECTLY there. Do NOT detour to other sites first \
to gather lists, find URLs, or do preliminary research.

Example: "Find all YC W25 founders on LinkedIn" → agents go DIRECTLY to linkedin.com, search \
for "YC W25", and extract data from LinkedIn. Do NOT go to ycombinator.com first. The user said \
LinkedIn, so LinkedIn is where you go.

## How it works

You are the brain. You decompose work into parallel sub-agents. You ALWAYS use the multi-agent \
pipeline — never try to do everything in a single worker. The flow is:

1. **Discover** (if no input CSV): Use `run_discovery` to send a browser agent to find the list \
of items to process. This creates a working CSV.
2. **Plan**: Use `plan_task` to create a structured plan from the instruction + CSV. This defines \
what each per-sample worker will do.
3. **Dispatch**: Use `dispatch_samples` to launch N workers in parallel, one per CSV row. You \
decide how many to run at once (via `adjust_config` for max_workers). Start with a small pilot \
batch (3-5 samples), verify results, then scale up.
4. **Monitor & adapt**: Use `check_progress`, `read_sample_trace`, `read_sample_result` to \
inspect results. If workers are failing, use `update_instructions` to fix the approach, then \
re-dispatch failed samples.
5. **Compile & finish**: Use `compile_results` to generate the final CSV, then `finish`.

## Your tools

**Deploy agents:**
- `run_discovery` — Deploy one agent to find a list of items on the web, then auto-create a CSV \
for further processing. Use this when no input CSV is provided or when you need to build a list \
before processing each item.
- `dispatch_samples` — Deploy N agents in parallel, one per CSV row. Requires `plan_task` first. \
Each agent processes one sample independently.

**Plan and manage:**
- `plan_task` — Create a structured plan from instruction + CSV. Required before `dispatch_samples`.
- `check_progress` — See status of dispatched samples.
- `read_sample_trace` / `read_sample_result` — Inspect a sample's execution for debugging.
- `update_instructions` — Change the per-sample instructions to fix issues.
- `adjust_config` — Tune workers, model, timeout, max iterations.
- `compile_results` — Generate the final results.csv from completed samples.
- `finish` — Complete the orchestration. Call this when you're done.

## Principles

- **The user's instruction is law.** If they say "go to LinkedIn", you go to LinkedIn. Period. \
Do not go to other sites first. Do not "optimize" by finding data elsewhere.
- **Always use the multi-agent pipeline.** Discover → Plan → Dispatch → Monitor → Compile. \
Never try to cram everything into a single worker.
- **You decide concurrency.** Use `adjust_config` to set max_workers based on the task. Simple \
tasks can run 5-10 workers in parallel. Complex/auth-gated tasks may need fewer.
- **Start small, verify, scale up.** Pilot with 3-5 samples first. Check results. Then dispatch the rest.
- **Adapt on failure.** Read traces of failed samples, understand why, change instructions or approach.
- **Critically evaluate discovery results.** After any discovery step, ask yourself: "Does this \
seem complete?" A YC batch has 150+ companies with 2-3 founders each. If you only found 50 \
founders, that's clearly incomplete. Try different search queries, vary search terms, paginate \
further. Call `run_discovery` multiple times with different strategies until you have comprehensive \
coverage. Don't just accept the first result as final.
- **Use multiple search strategies.** One search query rarely captures everything. Try variations: \
different keywords, company-by-company searches, different phrasings. Cast a wide net.
- `dispatch_samples` blocks until all workers complete — this lets you inspect between batches.
- Results are written to results.csv incrementally as workers complete.
"""


class OrchestratorToolProvider(ToolProvider):
    """Provides orchestrator-level tools for the agent loop."""

    def __init__(
        self,
        instruction: str,
        csv_path: str | None = None,
        max_workers: int = 3,
        output_dir_override: str | None = None,
        session_state_path: str | None = None,
        model_override: str | None = None,
        num_pioneers: int = 1,
        max_iterations: int = 30,
        browserbase: bool = False,
        dashboard: bool = False,
        profile_dir: str | None = None,
        direct_profile: bool = False,
    ):
        super().__init__()

        # Mutable state
        self.instruction = instruction
        self.csv_path = csv_path
        self.spec: TaskSpec | None = None
        self.progress: ProgressTracker | None = None
        self.rows: list[dict[str, Any]] = []
        self.id_to_row: dict[str, dict[str, Any]] = {}
        self.all_ids: list[str] = []
        self.playbook: str | None = None
        self.model_override = model_override
        self.session_state_path = session_state_path
        self.browserbase = browserbase
        self.profile_dir = profile_dir
        self.direct_profile = direct_profile

        # Dashboard
        self._dashboard_enabled = dashboard
        self._dashboard_server = None
        self._event_callback: EventCallback | None = None

        # Defaults from CLI
        self._initial_max_workers = max_workers
        self._initial_max_iterations = max_iterations
        self._initial_num_pioneers = num_pioneers
        self._output_dir_override = output_dir_override

        # Playwright state (initialized on first dispatch)
        self._pw = None
        self._browser = None
        self._semaphore: asyncio.Semaphore | None = None
        self._worker_counter = itertools.count()

        # Register all tools
        self.register_tool({
            "name": "plan_task",
            "description": "Create a structured task plan from the instruction and CSV. Must be called first.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, self._handle_plan_task)

        self.register_tool({
            "name": "run_discovery",
            "description": (
                "Dispatch a browser worker to discover items from the web and create the working CSV. "
                "Use this when no input CSV was provided. The worker browses a website, extracts a list "
                "of items, and the tool writes them as a CSV. After this, call plan_task."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "task_name": {
                        "type": "string",
                        "description": "Short snake_case name for this task (e.g., 'yc_w25_founders').",
                    },
                    "instructions": {
                        "type": "string",
                        "description": (
                            "Instructions for the discovery worker. Tell it what website to visit, "
                            "what items to find, and to call complete with a JSON array of items in "
                            "the 'items' field. Each item should be an object with consistent keys. "
                            "Example: 'Go to example.com/list, extract all company names and URLs. "
                            "Call complete with data: {items: \"[{name: ..., url: ...}, ...]\"}.'"
                        ),
                    },
                    "sample_column": {
                        "type": "string",
                        "description": "Which key in each discovered item to use as the sample ID (e.g., 'company_name').",
                    },
                },
                "required": ["task_name", "instructions", "sample_column"],
            },
        }, self._handle_run_discovery)

        self.register_tool({
            "name": "dispatch_samples",
            "description": (
                "Launch browser workers on a batch of sample IDs. Blocks until all complete. "
                "The first sample runs in pioneer mode to generate a playbook; remaining "
                "samples follow that playbook. Pass sample IDs from the CSV."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of sample IDs to process in this batch.",
                    },
                },
                "required": ["sample_ids"],
            },
        }, self._handle_dispatch_samples)

        self.register_tool({
            "name": "check_progress",
            "description": "Return the current status of all samples: counts by status and per-sample details.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, self._handle_check_progress)

        self.register_tool({
            "name": "read_sample_trace",
            "description": "Read a sample's execution trace (step-by-step agent actions). Useful for diagnosing failures.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "The sample ID to read the trace for.",
                    },
                },
                "required": ["sample_id"],
            },
        }, self._handle_read_sample_trace)

        self.register_tool({
            "name": "read_sample_result",
            "description": "Read a sample's metadata.json (extracted data, validation, etc.).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "sample_id": {
                        "type": "string",
                        "description": "The sample ID to read results for.",
                    },
                },
                "required": ["sample_id"],
            },
        }, self._handle_read_sample_result)

        self.register_tool({
            "name": "update_instructions",
            "description": (
                "Update the per-sample instructions that worker agents receive. "
                "Use this to fix issues discovered from reading failed traces."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "new_instructions": {
                        "type": "string",
                        "description": "The updated per-sample instructions. Use {column_name} templates for CSV values.",
                    },
                },
                "required": ["new_instructions"],
            },
        }, self._handle_update_instructions)

        self.register_tool({
            "name": "adjust_config",
            "description": "Change runtime configuration: workers, model, timeout, or max iterations.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "max_workers": {
                        "type": "integer",
                        "description": "Max concurrent browser workers.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Model override for worker agents.",
                    },
                    "timeout_per_sample_sec": {
                        "type": "integer",
                        "description": "Timeout per sample in seconds.",
                    },
                    "max_iterations": {
                        "type": "integer",
                        "description": "Max agent iterations per sample.",
                    },
                },
                "required": [],
            },
        }, self._handle_adjust_config)

        self.register_tool({
            "name": "compile_results",
            "description": "Generate results.csv from all completed samples' metadata.json files.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        }, self._handle_compile_results)

        # finish is registered but intercepted by the agent loop (like complete in worker)
        self.register_tool({
            "name": "finish",
            "description": "Signal that orchestration is complete. Provide a summary of what was accomplished.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of the orchestration run.",
                    },
                },
                "required": ["summary"],
            },
        }, self._handle_finish)

    async def _ensure_browser(self):
        """Lazily initialize Playwright browser on first dispatch."""
        if self._pw is None:
            self._pw = await async_playwright().start()
            # When using profile_dir, each worker launches its own Chrome
            # via launch_persistent_context — no shared browser needed.
            if not self.profile_dir:
                self._browser = await self._pw.chromium.launch(headless=True)

    async def _handle_plan_task(self, tool_input: dict[str, Any]) -> str:
        if not self.csv_path:
            return "Error: no CSV available. Call run_discovery first to create one."

        plan = plan_task(self.instruction, self.csv_path)

        self.spec = TaskSpec(
            task_name=plan["task_name"],
            per_sample_instructions=plan["per_sample_instructions"],
            input_csv=self.csv_path,
            sample_id_column=plan["sample_column"],
            csv_columns=plan["output_columns"],
            resolve_instructions=plan.get("resolve_instructions", ""),
            config=TaskConfig(
                max_workers=self._initial_max_workers,
                num_pioneers=self._initial_num_pioneers,
                max_iterations=self._initial_max_iterations,
            ),
            output_dir_override=self._output_dir_override,
            storage_state_path=self.session_state_path,
            profile_dir=self.profile_dir,
            direct_profile=self.direct_profile,
        )

        self.spec.output_dir.mkdir(parents=True, exist_ok=True)
        self.spec.samples_dir.mkdir(parents=True, exist_ok=True)

        self.rows = read_csv_rows(self.csv_path, self.spec.sample_id_column)
        self.all_ids = [str(r[self.spec.sample_id_column]) for r in self.rows]
        self.id_to_row = {str(r[self.spec.sample_id_column]): r for r in self.rows}

        self.progress = ProgressTracker(self.spec.output_dir)
        self._semaphore = asyncio.Semaphore(self.spec.config.max_workers)

        pending = self.progress.get_pending_samples(self.all_ids)

        # Start dashboard now that we know task_name and total_samples
        if self._dashboard_enabled and self._dashboard_server is None:
            try:
                from dashboard import DashboardServer
                self._dashboard_server = DashboardServer(
                    task_name=self.spec.task_name,
                    output_dir=self.spec.output_dir,
                    total_samples=len(self.all_ids),
                )
                await self._dashboard_server.start()
                self._event_callback = self._dashboard_server.broadcast_event
                # Register all samples as pending so dashboard shows the full list
                for sid in self.all_ids:
                    self._event_callback(WorkerEvent(
                        event_type="status_change", sample_id=sid, status="pending",
                    ))
            except ImportError:
                logger.warning("Dashboard dependencies not installed (fastapi/uvicorn). Skipping.")
            except Exception as e:
                logger.warning(f"Failed to start dashboard: {e}. Continuing without it.")

        return json.dumps({
            "task_name": self.spec.task_name,
            "sample_column": self.spec.sample_id_column,
            "output_columns": self.spec.csv_columns,
            "total_samples": len(self.all_ids),
            "pending_samples": len(pending),
            "pending_ids": pending,
            "resolve_instructions": self.spec.resolve_instructions or "(none)",
            "per_sample_instructions": self.spec.per_sample_instructions[:500],
        }, indent=2)

    async def _handle_run_discovery(self, tool_input: dict[str, Any]) -> str:
        """Dispatch a browser worker to discover items, then create the working CSV."""
        task_name = tool_input.get("task_name", "discovery")
        instructions = tool_input.get("instructions", "")
        sample_column = tool_input.get("sample_column", "id")

        if not instructions:
            return "Error: instructions cannot be empty"

        await self._ensure_browser()

        # Set up output dir for the discovery worker
        output_dir = Path(self._output_dir_override) if self._output_dir_override else Path("output") / task_name
        output_dir.mkdir(parents=True, exist_ok=True)
        discovery_dir = output_dir / "discovery"
        discovery_dir.mkdir(parents=True, exist_ok=True)

        # Discovery system prompt — tells the worker to return structured items
        discovery_system = (
            "You are a discovery agent. Follow the instructions EXACTLY as given.\n\n"
            "CRITICAL:\n"
            "- If the instructions mention a specific website, that is your target. Do NOT go to unrelated sites.\n"
            "- If the target site requires login and you can't access it, call `complete` with "
            'data.items set to "[]" and data.error explaining what happened (e.g., "LinkedIn requires '
            'login — session expired"). Do NOT fall back to other sites.\n'
            "- Try MULTIPLE search queries to maximize coverage. One query is rarely enough. "
            "Vary your keywords, try synonyms, search page by page.\n"
            "- DO NOT reverse-engineer or call backend APIs (Algolia, GraphQL, internal endpoints, etc.). "
            "You are a BROWSER agent — interact with the page as a user would: scroll, click, "
            "use execute_js to read the rendered DOM. If a page doesn't render content, try "
            "enabling stealth mode, waiting for dynamic content, or scrolling to trigger lazy loading.\n\n"
            "When you have found all items, call the `complete` tool with:\n"
            '- data.items: a JSON string containing an array of objects, e.g., '
            '\'[{"name": "foo", "url": "..."}, {"name": "bar", "url": "..."}]\'\n\n'
            "Each object should have consistent keys across all items.\n"
            "Be thorough — extract ALL items, not just the first few. "
            "Scroll, paginate, or navigate as needed to find everything.\n"
            "Use execute_js to extract structured data efficiently from the DOM.\n"
            "Deduplicate results before returning."
        )

        registry = ToolRegistry()
        discovery_worker_id = None
        bb_browser = None
        try:
            if self.browserbase and self._pw:
                import os
                from browserbase import Browserbase
                bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
                session = bb.sessions.create(
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                )
                logger.info(f"  Browserbase discovery session: https://browserbase.com/sessions/{session.id}")
                bb_browser = await self._pw.chromium.connect_over_cdp(session.connect_url)
                bb_ctx = bb_browser.contexts[0]
                if self.session_state_path:
                    import json
                    state = json.loads(Path(self.session_state_path).read_text())
                    if state.get("cookies"):
                        await bb_ctx.add_cookies(state["cookies"])
                        logger.info(f"  Injected {len(state['cookies'])} cookies into Browserbase")
                browser_provider = await create_browser_provider(
                    context=bb_ctx,
                    output_dir=discovery_dir,
                )
            elif self.direct_profile and self.profile_dir and self._pw:
                browser_provider = await create_browser_provider(
                    profile_dir=self.profile_dir,
                    playwright=self._pw,
                    output_dir=discovery_dir,
                )
            elif self.profile_dir and self._pw:
                discovery_worker_id = next(self._worker_counter)
                copy_dir = worker_copy(discovery_worker_id)
                browser_provider = await create_browser_provider(
                    profile_dir=str(copy_dir),
                    playwright=self._pw,
                    output_dir=discovery_dir,
                )
            else:
                browser_provider = await create_browser_provider(
                    browser=self._browser,
                    output_dir=discovery_dir,
                    storage_state_path=self.session_state_path,
                )
            registry.register(browser_provider)
            registry.register(CodeToolProvider())
            registry.register(HttpToolProvider())

            data = await asyncio.wait_for(
                run_worker(
                    registry=registry,
                    instructions=instructions,
                    csv_columns=["items"],
                    row={},
                    output_dir=discovery_dir,
                    max_iterations=40,
                    model_override=self.model_override,
                    system_prompt_override=discovery_system,
                ),
                timeout=600,  # 10 min for thorough discovery
            )

            # Parse discovered items
            items_raw = data.get("items", "[]")
            if isinstance(items_raw, list):
                items = items_raw
            elif isinstance(items_raw, str):
                try:
                    items = json.loads(items_raw)
                except json.JSONDecodeError:
                    return f"Error: discovery worker returned unparseable items: {items_raw[:500]}"
            else:
                return f"Error: unexpected items type: {type(items_raw)}"

            if not items or not isinstance(items, list):
                return "Error: discovery worker returned no items"

            # Validate sample_column exists in items
            if isinstance(items[0], dict) and sample_column not in items[0]:
                available = list(items[0].keys())
                return f"Error: sample_column '{sample_column}' not found in items. Available keys: {available}"

            # Write discovered items as CSV
            csv_path = output_dir / "discovered.csv"
            if isinstance(items[0], dict):
                fieldnames = list(items[0].keys())
            else:
                # Items are simple strings — wrap in dicts
                fieldnames = [sample_column]
                items = [{sample_column: str(item)} for item in items]

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for item in items:
                    writer.writerow({k: str(v) for k, v in item.items()})

            # Set the csv_path so plan_task can use it
            self.csv_path = str(csv_path)

            logger.info(f"Discovery complete: {len(items)} items → {csv_path}")

            return json.dumps({
                "status": "success",
                "items_found": len(items),
                "csv_path": str(csv_path),
                "columns": fieldnames,
                "sample_column": sample_column,
                "first_items": items[:5],
            }, indent=2, default=str)

        except asyncio.TimeoutError:
            return "Error: discovery worker timed out after 300s"
        except Exception as e:
            return f"Error during discovery: {str(e)[:500]}"
        finally:
            await registry.close()
            if bb_browser:
                await bb_browser.close()
            if discovery_worker_id is not None:
                cleanup_worker(discovery_worker_id)

    async def _handle_dispatch_samples(self, tool_input: dict[str, Any]) -> str:
        if not self.spec or not self.progress:
            return "Error: must call plan_task first"

        sample_ids = tool_input.get("sample_ids", [])
        if not sample_ids:
            return "Error: no sample_ids provided"

        # Validate IDs
        invalid = [sid for sid in sample_ids if sid not in self.id_to_row]
        if invalid:
            return f"Error: unknown sample IDs: {invalid}"

        await self._ensure_browser()

        # Pioneer-follower within this batch
        pioneer_id = sample_ids[0]
        follower_ids = sample_ids[1:]
        results_summary: list[dict[str, Any]] = []

        # Run pioneer
        pioneer_row = self.id_to_row[pioneer_id]
        logger.info(f"Agent dispatching pioneer: {pioneer_id}")

        pioneer_result = await self._process_one(
            pioneer_id, pioneer_row, pioneer_mode=True
        )

        if isinstance(pioneer_result, dict):
            results_summary.append({"sample_id": pioneer_id, "status": "completed"})
            # Distill playbook from pioneer
            raw_playbook = pioneer_result.pop("playbook", None)
            if raw_playbook:
                trace_path = self.spec.sample_dir(pioneer_id) / "trace.json"
                self.playbook = distill_playbook(
                    original_instructions=self.spec.per_sample_instructions,
                    trace_path=trace_path,
                    raw_playbook=raw_playbook,
                )
                playbook_path = self.spec.output_dir / "playbook.md"
                playbook_path.write_text(self.playbook)
        else:
            results_summary.append({
                "sample_id": pioneer_id,
                "status": "failed",
                "error": self.progress._data.get(pioneer_id, {}).get("error", "unknown"),
            })

        # Run followers in parallel
        if follower_ids:
            async def _run_follower(sid: str) -> dict[str, Any]:
                row = self.id_to_row[sid]
                success = await self._process_one(
                    sid, row, pioneer_mode=False, playbook=self.playbook
                )
                if success is True or isinstance(success, dict):
                    return {"sample_id": sid, "status": "completed"}
                return {
                    "sample_id": sid,
                    "status": "failed",
                    "error": self.progress._data.get(sid, {}).get("error", "unknown"),
                }

            follower_results = await asyncio.gather(
                *[_run_follower(sid) for sid in follower_ids],
                return_exceptions=True,
            )
            for i, r in enumerate(follower_results):
                if isinstance(r, Exception):
                    results_summary.append({
                        "sample_id": follower_ids[i],
                        "status": "failed",
                        "error": str(r)[:200],
                    })
                else:
                    results_summary.append(r)

        succeeded = sum(1 for r in results_summary if r["status"] == "completed")
        failed = len(results_summary) - succeeded

        return json.dumps({
            "dispatched": len(sample_ids),
            "succeeded": succeeded,
            "failed": failed,
            "results": results_summary,
        }, indent=2)

    async def _process_one(
        self,
        sample_id: str,
        row: dict[str, Any],
        pioneer_mode: bool = False,
        playbook: str | None = None,
    ) -> bool | dict[str, Any]:
        """Process a single sample. Mirrors orchestrator.process_sample logic."""
        sample_dir = self.spec.sample_dir(sample_id)
        self.progress.set_status(sample_id, SampleStatus.IN_PROGRESS)
        if self._event_callback:
            self._event_callback(WorkerEvent(
                event_type="status_change", sample_id=sample_id, status="in_progress",
            ))

        # Resolve phase
        resolved = await resolve_sample(self.spec, row, model_override=self.model_override)
        enriched_row = {**row, **resolved}

        if self._event_callback:
            registry = EventEmittingRegistry(sample_id, self._event_callback)
        else:
            registry = ToolRegistry()
        worker_id = None
        bb_browser = None
        try:
            if self.browserbase and self._pw:
                import os
                from browserbase import Browserbase
                bb = Browserbase(api_key=os.environ["BROWSERBASE_API_KEY"])
                session = bb.sessions.create(
                    project_id=os.environ["BROWSERBASE_PROJECT_ID"],
                )
                logger.info(f"  Browserbase session: https://browserbase.com/sessions/{session.id}")
                bb_browser = await self._pw.chromium.connect_over_cdp(session.connect_url)
                bb_ctx = bb_browser.contexts[0]
                # Inject cookies from storage state if available
                if self.spec.storage_state_path:
                    import json
                    state = json.loads(Path(self.spec.storage_state_path).read_text())
                    if state.get("cookies"):
                        await bb_ctx.add_cookies(state["cookies"])
                        logger.info(f"  Injected {len(state['cookies'])} cookies into Browserbase")
                browser_provider = await create_browser_provider(
                    context=bb_ctx,
                    output_dir=sample_dir,
                )
            elif self.direct_profile and self.spec.profile_dir and self._pw:
                browser_provider = await create_browser_provider(
                    profile_dir=self.spec.profile_dir,
                    playwright=self._pw,
                    output_dir=sample_dir,
                )
            elif self.spec.profile_dir and self._pw:
                worker_id = next(self._worker_counter)
                copy_dir = worker_copy(worker_id)
                browser_provider = await create_browser_provider(
                    profile_dir=str(copy_dir),
                    playwright=self._pw,
                    output_dir=sample_dir,
                )
            else:
                browser_provider = await create_browser_provider(
                    browser=self._browser,
                    output_dir=sample_dir,
                    storage_state_path=self.spec.storage_state_path,
                )
            registry.register(browser_provider)
            registry.register(CodeToolProvider())
            registry.register(CodeAnalysisProvider())
            registry.register(HttpToolProvider())

            instructions = self.spec.render_instructions(enriched_row)

            data = await asyncio.wait_for(
                run_worker(
                    registry=registry,
                    instructions=instructions,
                    csv_columns=self.spec.csv_columns,
                    row=row,
                    output_dir=sample_dir,
                    max_iterations=self.spec.config.max_iterations,
                    model_override=self.model_override,
                    pioneer_mode=pioneer_mode,
                    playbook=playbook,
                ),
                timeout=self.spec.config.timeout_per_sample_sec,
            )

            write_metadata(sample_dir, sample_id, data, row, extra={"playbook_used": playbook is not None})
            validate_sample(sample_dir, self.spec.csv_columns, data)
            self.progress.set_status(sample_id, SampleStatus.COMPLETED)
            if self._event_callback:
                self._event_callback(WorkerEvent(
                    event_type="status_change", sample_id=sample_id, status="completed",
                ))

            # Incremental CSV write — append this result immediately
            try:
                append_result_row(self.spec.output_dir, self.spec.csv_columns, sample_id, data)
            except Exception as e:
                logger.warning(f"Failed to append result row for {sample_id}: {e}")

            if pioneer_mode:
                return data
            return True

        except asyncio.TimeoutError:
            msg = f"Timeout after {self.spec.config.timeout_per_sample_sec}s"
            self.progress.set_status(sample_id, SampleStatus.FAILED, error=msg)
            if self._event_callback:
                self._event_callback(WorkerEvent(
                    event_type="status_change", sample_id=sample_id, status="failed", error=msg,
                ))
            return False

        except Exception as e:
            msg = str(e)[:500]
            self.progress.set_status(sample_id, SampleStatus.FAILED, error=msg)
            if self._event_callback:
                self._event_callback(WorkerEvent(
                    event_type="status_change", sample_id=sample_id, status="failed", error=msg,
                ))
            return False

        finally:
            await registry.close()
            if bb_browser:
                await bb_browser.close()
            if worker_id is not None:
                cleanup_worker(worker_id)

    async def _handle_check_progress(self, tool_input: dict[str, Any]) -> str:
        if not self.progress:
            return "Error: must call plan_task first"

        summary = self.progress.summary()

        # Per-status ID lists
        by_status: dict[str, list[str]] = {}
        for sid, entry in self.progress._data.items():
            status = entry.get("status", "unknown")
            by_status.setdefault(status, []).append(sid)

        # Include errors for failed
        failed_details = []
        for sid in by_status.get("failed", []):
            error = self.progress._data[sid].get("error", "unknown")
            failed_details.append({"sample_id": sid, "error": error})

        return json.dumps({
            "summary": summary,
            "by_status": by_status,
            "failed_details": failed_details,
        }, indent=2)

    async def _handle_read_sample_trace(self, tool_input: dict[str, Any]) -> str:
        if not self.spec:
            return "Error: must call plan_task first"

        sample_id = tool_input.get("sample_id", "")
        trace_path = self.spec.sample_dir(sample_id) / "trace.json"
        return build_trace_text(trace_path, max_steps=30)

    async def _handle_read_sample_result(self, tool_input: dict[str, Any]) -> str:
        if not self.spec:
            return "Error: must call plan_task first"

        sample_id = tool_input.get("sample_id", "")
        metadata_path = self.spec.sample_dir(sample_id) / "metadata.json"

        if not metadata_path.exists():
            return f"No metadata.json found for sample {sample_id}"

        try:
            meta = json.loads(metadata_path.read_text())
            return json.dumps(meta, indent=2, default=str)
        except (json.JSONDecodeError, OSError) as e:
            return f"Error reading metadata: {e}"

    async def _handle_update_instructions(self, tool_input: dict[str, Any]) -> str:
        if not self.spec:
            return "Error: must call plan_task first"

        new_instructions = tool_input.get("new_instructions", "")
        if not new_instructions:
            return "Error: new_instructions cannot be empty"

        old = self.spec.per_sample_instructions[:200]
        self.spec.per_sample_instructions = new_instructions
        # Clear playbook since instructions changed
        self.playbook = None

        return json.dumps({
            "status": "updated",
            "old_instructions_preview": old + "...",
            "new_instructions_preview": new_instructions[:200] + "...",
            "note": "Playbook cleared — next batch will run a new pioneer.",
        })

    async def _handle_adjust_config(self, tool_input: dict[str, Any]) -> str:
        if not self.spec:
            return "Error: must call plan_task first"

        changes: dict[str, Any] = {}
        if "max_workers" in tool_input:
            self.spec.config.max_workers = tool_input["max_workers"]
            self._semaphore = asyncio.Semaphore(tool_input["max_workers"])
            changes["max_workers"] = tool_input["max_workers"]

        if "model" in tool_input:
            self.model_override = tool_input["model"]
            changes["model"] = tool_input["model"]

        if "timeout_per_sample_sec" in tool_input:
            self.spec.config.timeout_per_sample_sec = tool_input["timeout_per_sample_sec"]
            changes["timeout_per_sample_sec"] = tool_input["timeout_per_sample_sec"]

        if "max_iterations" in tool_input:
            self.spec.config.max_iterations = tool_input["max_iterations"]
            changes["max_iterations"] = tool_input["max_iterations"]

        if not changes:
            return "No config changes specified."

        return json.dumps({"updated": changes})

    async def _handle_compile_results(self, tool_input: dict[str, Any]) -> str:
        if not self.spec:
            return "Error: must call plan_task first"

        results_path = compile_results(self.spec.output_dir, self.spec.csv_columns)
        return f"Results compiled to {results_path}"

    async def _handle_finish(self, tool_input: dict[str, Any]) -> str:
        # This is intercepted by the agent loop before reaching here
        return tool_input.get("summary", "Orchestration complete.")

    async def close(self) -> None:
        if self._dashboard_server:
            try:
                await self._dashboard_server.stop()
            except Exception:
                pass
        if self._browser:
            try:
                await self._browser.close()
            except Exception:
                pass
        if self._pw:
            await self._pw.stop()


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

async def run_orchestrator_agent(
    instruction: str,
    csv_path: str | None = None,
    max_workers: int = 3,
    output_dir_override: str | None = None,
    session_state_path: str | None = None,
    model_override: str | None = None,
    num_pioneers: int = 1,
    max_iterations: int = 30,
    browserbase: bool = False,
    dashboard: bool = False,
    profile_dir: str | None = None,
    direct_profile: bool = False,
):
    """Run the orchestrator as an LLM agent loop.

    Mirrors worker.py's run_worker() structurally but at the orchestrator level.
    The agent reasons about what to do, calls orchestrator tools, inspects results,
    and adapts its strategy.
    """
    start_time = time.time()
    client = anthropic.Anthropic(max_retries=25)
    model = model_override or MODEL

    COMPACTION_BETA = "compact-2026-01-12"

    # Initialize the tool provider
    provider = OrchestratorToolProvider(
        instruction=instruction,
        csv_path=csv_path,
        max_workers=max_workers,
        output_dir_override=output_dir_override,
        session_state_path=session_state_path,
        model_override=model_override,
        num_pioneers=num_pioneers,
        max_iterations=max_iterations,
        browserbase=browserbase,
        dashboard=dashboard,
        profile_dir=profile_dir,
        direct_profile=direct_profile,
    )

    registry = ToolRegistry()
    registry.register(provider)

    tools = registry.get_tool_schemas()
    # Remove the default 'complete' tool — orchestrator uses 'finish' instead
    tools = [t for t in tools if t["name"] != "complete"]

    # Analyze the instruction to extract any explicitly mentioned websites
    mentioned_sites = _extract_mentioned_sites(instruction)
    site_guidance = ""
    if mentioned_sites:
        sites_str = ", ".join(mentioned_sites)
        site_guidance = (
            f"\n\n**IMPORTANT: The instruction explicitly mentions these websites: {sites_str}. "
            f"Your agents MUST go directly to {sites_str}. Do NOT go to other websites first "
            f"to find lists or do preliminary research. Go DIRECTLY to {sites_str}.**"
        )

    if csv_path:
        initial_msg = (
            f"## Instruction\n{instruction}\n\n"
            f"## Input CSV\n{csv_path}\n\n"
            f"Analyze this task and decide how to accomplish it.{site_guidance}"
        )
    else:
        initial_msg = (
            f"## Instruction\n{instruction}\n\n"
            f"No input CSV provided. Analyze this task and decide the best approach.{site_guidance}"
        )

    messages: list[dict[str, Any]] = [
        {"role": "user", "content": initial_msg},
    ]

    # Trace for debugging the orchestrator itself
    trace: list[dict[str, Any]] = []
    total_input_tokens = 0
    total_output_tokens = 0
    api_calls = 0

    output_dir = Path(output_dir_override) if output_dir_override else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for iteration in range(MAX_ITERATIONS):
            logger.info(f"Orchestrator iteration {iteration + 1}/{MAX_ITERATIONS}")
            step: dict[str, Any] = {
                "iteration": iteration + 1,
                "timestamp": round(time.time() - start_time, 2),
            }

            try:
                response = client.beta.messages.create(
                    betas=[COMPACTION_BETA],
                    model=model,
                    max_tokens=MAX_TOKENS,
                    system=ORCHESTRATOR_SYSTEM,
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
                raise

            api_calls += 1
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

            # Extract reasoning
            reasoning = []
            for block in assistant_content:
                if hasattr(block, "type") and block.type == "text":
                    reasoning.append(block.text)
            if reasoning:
                step["reasoning"] = "\n".join(reasoning)

            tool_uses = [
                block for block in assistant_content
                if hasattr(block, "type") and block.type == "tool_use"
            ]

            if not tool_uses:
                step["action"] = "no_tool_call"
                step["stop_reason"] = response.stop_reason
                trace.append(step)

                if response.stop_reason == "end_turn":
                    messages.append({
                        "role": "user",
                        "content": "You must call a tool. Use `check_progress` to see status, or `finish` if done.",
                    })
                    continue
                if response.stop_reason == "compaction":
                    continue
                break

            tool_results = []
            step_tools: list[dict[str, Any]] = []

            for tool_use in tool_uses:
                tool_name = tool_use.name
                tool_input = tool_use.input
                tool_id = tool_use.id

                logger.info(f"  Orchestrator tool: {tool_name}({json.dumps(tool_input, default=str)[:120]})")

                tool_trace: dict[str, Any] = {
                    "tool": tool_name,
                    "input": tool_input,
                }

                # Intercept finish — same pattern as complete in worker
                if tool_name == "finish":
                    summary = tool_input.get("summary", "")
                    logger.info(f"Orchestrator finished. Summary: {summary}")

                    tool_trace["result"] = "ORCHESTRATION COMPLETE"
                    step_tools.append(tool_trace)
                    step["tools"] = step_tools
                    trace.append(step)

                    # Generate failure report if there are failures
                    if provider.spec and provider.progress:
                        failure_summary = provider.progress.summary()
                        if failure_summary.get("failed", 0) > 0:
                            generate_failure_report(
                                provider.spec.output_dir,
                                provider.progress,
                                task_instructions=provider.spec.per_sample_instructions,
                            )

                    _save_orchestrator_trace(trace, provider, start_time, api_calls,
                                            total_input_tokens, total_output_tokens, model)
                    return

                # Dispatch to registry
                try:
                    result = await registry.execute(tool_name, tool_input)
                except Exception as e:
                    result = f"Error executing {tool_name}: {e}"
                    logger.error(f"  {result}")

                tool_trace["result"] = result[:3000]
                step_tools.append(tool_trace)

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tool_id,
                    "content": [{"type": "text", "text": result[:12000]}],
                })

            step["tools"] = step_tools
            trace.append(step)
            messages.append({"role": "user", "content": tool_results})

        # Max iterations reached
        logger.warning(f"Orchestrator did not finish within {MAX_ITERATIONS} iterations")

    finally:
        _save_orchestrator_trace(trace, provider, start_time, api_calls,
                                total_input_tokens, total_output_tokens, model)
        await provider.close()


def _save_orchestrator_trace(
    trace: list[dict[str, Any]],
    provider: OrchestratorToolProvider,
    start_time: float,
    api_calls: int,
    total_input_tokens: int,
    total_output_tokens: int,
    model: str,
):
    """Save orchestrator trace and audit to the output directory."""
    output_dir = provider.spec.output_dir if provider.spec else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_path = output_dir / "orchestrator_trace.json"
    with open(trace_path, "w") as f:
        json.dump(trace, f, indent=2, default=str)

    audit_path = output_dir / "orchestrator_audit.json"
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

    logger.info(f"Orchestrator trace saved to {trace_path}")
