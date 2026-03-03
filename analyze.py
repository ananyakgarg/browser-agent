"""Post-mortem CLI: diagnose failures from completed runs.

Usage:
    python analyze.py output/task_name              # all failures + pattern analysis
    python analyze.py output/task_name/samples/X    # single sample diagnosis
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import anthropic

from distill import build_trace_text

logger = logging.getLogger(__name__)

ANALYZE_MODEL = "claude-sonnet-4-6"

DIAGNOSIS_SYSTEM = """\
You are a failure analyst for a browser automation system. You receive the \
execution trace of an agent that attempted to complete a browser automation task.

Provide a clear, structured diagnosis:
1. **What happened**: Brief summary of the agent's actions
2. **Root cause**: Why it failed (be specific)
3. **Failure type**: Classify (site blocked, element not found, timeout, \
wrong navigation, data extraction error, login required, CAPTCHA, API error, etc.)
4. **Suggested fix**: Concrete actionable fix (e.g., "add stealth mode before \
navigating", "use execute_js instead of click for this element", "use API endpoint \
/api/v2/... instead of browser")

Keep it concise — focus on actionable insights.\
"""

TASK_ANALYSIS_SYSTEM = """\
You are a failure analyst for a browser automation system. You receive traces \
from multiple failed runs AND a sample of successful runs.

Provide:
1. **Per-failure diagnosis** — root cause and fix for each failed sample
2. **Cross-failure patterns** — common themes across failures
3. **Quality check** — any concerns about the successful samples (incomplete data, \
suspicious patterns, potential false positives)
4. **Recommendations** — changes to instructions, config, or approach that would \
improve the overall success rate

Be specific and actionable.\
"""


def diagnose_sample(sample_dir: Path, model: str | None = None) -> str:
    """Diagnose a single sample from its trace."""
    trace_path = sample_dir / "trace.json"
    metadata_path = sample_dir / "metadata.json"

    if not trace_path.exists():
        return f"No trace.json found in {sample_dir}"

    trace_text = build_trace_text(trace_path)

    # Include metadata if available
    context_parts = [f"## Execution Trace\n{trace_text}"]

    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            context_parts.insert(0, f"## Metadata\n```json\n{json.dumps(meta, indent=2)}\n```")
        except (json.JSONDecodeError, OSError):
            pass

    # Check for error in progress
    progress_path = sample_dir.parent.parent / "progress.json"
    if progress_path.exists():
        try:
            progress = json.loads(progress_path.read_text())
            sample_id = sample_dir.name
            entry = progress.get(sample_id, {})
            if entry.get("error"):
                context_parts.insert(0, f"## Error\n{entry['error']}")
            if entry.get("status"):
                context_parts.insert(0, f"**Status**: {entry['status']}")
        except (json.JSONDecodeError, OSError):
            pass

    user_msg = "Diagnose this browser automation run:\n\n" + "\n\n".join(context_parts)

    client = anthropic.Anthropic(max_retries=10)
    response = client.messages.create(
        model=model or ANALYZE_MODEL,
        max_tokens=2048,
        system=DIAGNOSIS_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "text":
            return block.text
    return "(analysis produced no output)"


def diagnose_task(task_dir: Path, model: str | None = None) -> str:
    """Diagnose all failures in a task run + quality-check successes."""
    progress_path = task_dir / "progress.json"
    samples_dir = task_dir / "samples"

    if not progress_path.exists():
        return f"No progress.json found in {task_dir}"
    if not samples_dir.exists():
        return f"No samples directory found in {task_dir}"

    progress = json.loads(progress_path.read_text())

    failed_ids = [sid for sid, entry in progress.items() if entry.get("status") == "failed"]
    completed_ids = [sid for sid, entry in progress.items() if entry.get("status") == "completed"]

    if not failed_ids and not completed_ids:
        return "No completed or failed samples found in progress.json"

    parts: list[str] = []

    # Collect failed traces
    for sid in failed_ids:
        safe_id = sid.replace("/", "_").replace("\\", "_")
        sample_dir = samples_dir / safe_id
        trace_path = sample_dir / "trace.json"
        error = progress.get(sid, {}).get("error", "unknown")
        trace_text = build_trace_text(trace_path, max_steps=25)

        parts.append(
            f"## FAILED: {sid}\n"
            f"Error: {error}\n\n"
            f"### Trace\n{trace_text}"
        )

    # Sample a few successful traces for quality check (max 3)
    for sid in completed_ids[:3]:
        safe_id = sid.replace("/", "_").replace("\\", "_")
        sample_dir = samples_dir / safe_id
        trace_path = sample_dir / "trace.json"
        metadata_path = sample_dir / "metadata.json"

        trace_text = build_trace_text(trace_path, max_steps=15)
        meta_text = ""
        if metadata_path.exists():
            try:
                meta = json.loads(metadata_path.read_text())
                data = meta.get("extracted_data", {})
                meta_text = f"\nExtracted data: {json.dumps(data, indent=2)}"
            except (json.JSONDecodeError, OSError):
                pass

        parts.append(
            f"## SUCCEEDED: {sid}{meta_text}\n\n"
            f"### Trace (abbreviated)\n{trace_text}"
        )

    user_msg = (
        f"Analyze this browser automation task run.\n"
        f"**{len(failed_ids)} failed**, **{len(completed_ids)} succeeded**.\n\n"
        + "\n\n---\n\n".join(parts)
    )

    client = anthropic.Anthropic(max_retries=10)
    response = client.messages.create(
        model=model or ANALYZE_MODEL,
        max_tokens=4096,
        system=TASK_ANALYSIS_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )

    for block in response.content:
        if block.type == "text":
            return block.text
    return "(analysis produced no output)"


def main():
    parser = argparse.ArgumentParser(
        description="Post-mortem analysis of browser automation runs"
    )
    parser.add_argument(
        "path",
        help="Path to task output dir (has progress.json) or sample dir (has trace.json)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help=f"Model for analysis (default: {ANALYZE_MODEL})",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    target = Path(args.path)
    if not target.exists():
        print(f"Path not found: {target}", file=sys.stderr)
        sys.exit(1)

    # Detect: task dir (has progress.json) vs sample dir (has trace.json)
    if (target / "progress.json").exists():
        print(f"Analyzing task: {target}\n")
        result = diagnose_task(target, model=args.model)
    elif (target / "trace.json").exists():
        print(f"Analyzing sample: {target}\n")
        result = diagnose_sample(target, model=args.model)
    else:
        print(
            f"Cannot determine type of {target}.\n"
            f"Expected either progress.json (task dir) or trace.json (sample dir).",
            file=sys.stderr,
        )
        sys.exit(1)

    print(result)


if __name__ == "__main__":
    main()
