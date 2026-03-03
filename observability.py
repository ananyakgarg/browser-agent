"""Observability: failure reports and event-emitting registry for live dashboard."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from distill import build_trace_text
from progress import ProgressTracker, SampleStatus
from tool_registry import ToolRegistry

logger = logging.getLogger(__name__)

REPORT_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Layer 1: Failure report
# ---------------------------------------------------------------------------

FAILURE_REPORT_SYSTEM = """\
You are a failure analyst for a browser automation system. You receive traces \
from failed agent runs and produce structured diagnoses.

For each failed sample:
1. Identify the ROOT CAUSE — what went wrong and why
2. Classify the failure type (e.g., site blocked, element not found, timeout, \
wrong page, data extraction error, login required, CAPTCHA, API error)
3. Suggest a specific fix or workaround

After analyzing individual failures, identify CROSS-FAILURE PATTERNS:
- Are multiple samples failing for the same reason?
- Is there a systemic issue (e.g., site blocking all requests)?
- Would a change in instructions fix most failures?

Format your output as markdown with clear headers.\
"""


def generate_failure_report(
    output_dir: Path,
    progress: ProgressTracker,
    task_instructions: str | None = None,
    model: str | None = None,
) -> Path | None:
    """Generate a failure report for all failed samples in a task run.

    Reads each failed sample's trace.json, batches them for LLM analysis,
    and writes failure_report.md to the output directory.

    Returns the path to the report, or None if no failures.
    """
    samples_dir = output_dir / "samples"
    if not samples_dir.exists():
        return None

    # Collect failed sample IDs from progress
    failed_ids = []
    for sample_id, entry in progress._data.items():
        if entry.get("status") == SampleStatus.FAILED.value:
            failed_ids.append(sample_id)

    if not failed_ids:
        logger.info("No failures to report.")
        return None

    logger.info(f"Generating failure report for {len(failed_ids)} failed samples...")

    # Read traces for failed samples
    failure_traces: list[dict[str, Any]] = []
    for sample_id in failed_ids:
        safe_id = sample_id.replace("/", "_").replace("\\", "_")
        sample_dir = samples_dir / safe_id
        trace_path = sample_dir / "trace.json"
        error = progress._data.get(sample_id, {}).get("error", "unknown")

        trace_text = build_trace_text(trace_path, max_steps=30)
        failure_traces.append({
            "sample_id": sample_id,
            "error": error,
            "trace": trace_text,
        })

    # Batch traces for LLM calls (max 5 per call to stay within context)
    client = anthropic.Anthropic(max_retries=10)
    report_model = model or REPORT_MODEL
    report_sections: list[str] = []

    BATCH_SIZE = 5
    for i in range(0, len(failure_traces), BATCH_SIZE):
        batch = failure_traces[i:i + BATCH_SIZE]

        batch_text_parts = []
        for ft in batch:
            batch_text_parts.append(
                f"## Sample: {ft['sample_id']}\n"
                f"Error: {ft['error']}\n\n"
                f"### Trace\n{ft['trace']}\n"
            )
        batch_text = "\n---\n\n".join(batch_text_parts)

        user_msg = f"Analyze these {len(batch)} failed browser automation runs:\n\n{batch_text}"
        if task_instructions:
            user_msg = f"## Task Instructions\n{task_instructions}\n\n{user_msg}"

        if i + BATCH_SIZE >= len(failure_traces):
            # Last batch — also request cross-failure pattern analysis
            user_msg += (
                "\n\nAfter diagnosing each failure individually, provide a "
                "## Cross-Failure Patterns section analyzing commonalities "
                "and systemic issues across ALL failures in this run."
            )

        response = client.messages.create(
            model=report_model,
            max_tokens=4096,
            system=FAILURE_REPORT_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )

        for block in response.content:
            if block.type == "text":
                report_sections.append(block.text)

    # Assemble final report
    summary = progress.summary()
    report = (
        f"# Failure Report\n\n"
        f"**Total samples**: {sum(summary.values())}  \n"
        f"**Completed**: {summary.get('completed', 0)}  \n"
        f"**Failed**: {summary.get('failed', 0)}  \n\n"
        f"---\n\n"
    )
    report += "\n\n---\n\n".join(report_sections)

    report_path = output_dir / "failure_report.md"
    report_path.write_text(report)
    logger.info(f"Failure report written to {report_path}")
    return report_path


# ---------------------------------------------------------------------------
# Layer 2: Event system for live dashboard
# ---------------------------------------------------------------------------

@dataclass
class WorkerEvent:
    """Event emitted during worker execution for live dashboard updates."""
    event_type: str  # "tool_start", "tool_end", "status_change", "error"
    sample_id: str
    timestamp: float = field(default_factory=time.time)
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_result: str | None = None
    iteration: int | None = None
    status: str | None = None
    error: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "event_type": self.event_type,
            "sample_id": self.sample_id,
            "timestamp": self.timestamp,
        }
        if self.tool_name is not None:
            d["tool_name"] = self.tool_name
        if self.tool_input is not None:
            # Truncate input for transport
            input_str = json.dumps(self.tool_input, default=str)
            d["tool_input"] = input_str[:500]
        if self.tool_result is not None:
            d["tool_result"] = self.tool_result[:1000]
        if self.iteration is not None:
            d["iteration"] = self.iteration
        if self.status is not None:
            d["status"] = self.status
        if self.error is not None:
            d["error"] = self.error
        if self.extra:
            d["extra"] = self.extra
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


# Type alias for event callbacks
EventCallback = Callable[[WorkerEvent], None]


class EventEmittingRegistry(ToolRegistry):
    """Wraps ToolRegistry to emit WorkerEvents on every tool execution.

    The worker loop is unchanged — this intercepts at the registry level.
    Events are broadcast to all connected dashboard clients via the callback.
    """

    def __init__(self, sample_id: str, event_callback: EventCallback | None = None):
        super().__init__()
        self.sample_id = sample_id
        self._event_callback = event_callback
        self._iteration = 0

    def set_iteration(self, iteration: int) -> None:
        self._iteration = iteration

    def _emit(self, event: WorkerEvent) -> None:
        if self._event_callback:
            try:
                self._event_callback(event)
            except Exception:
                pass  # Never let event emission break the worker

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        # Emit tool_start
        self._emit(WorkerEvent(
            event_type="tool_start",
            sample_id=self.sample_id,
            tool_name=tool_name,
            tool_input=tool_input,
            iteration=self._iteration,
        ))

        # Dispatch to parent
        result = await super().execute(tool_name, tool_input)

        # Emit tool_end
        self._emit(WorkerEvent(
            event_type="tool_end",
            sample_id=self.sample_id,
            tool_name=tool_name,
            tool_result=result,
            iteration=self._iteration,
        ))

        return result
