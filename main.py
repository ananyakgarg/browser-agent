"""CLI entry point: python main.py --task path/to/spec.json"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from config import load_task_spec
from orchestrator import run_orchestrator


def main():
    parser = argparse.ArgumentParser(
        description="Browser automation agent — processes CSV items via LLM-controlled browser"
    )
    parser.add_argument(
        "--task",
        required=True,
        help="Path to the task specification JSON file",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        spec = load_task_spec(args.task)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading task spec: {e}", file=sys.stderr)
        sys.exit(1)

    logging.getLogger(__name__).info(
        f"Task: {spec.task_name} | "
        f"Workers: {spec.config.max_workers} | "
        f"Retries: {spec.config.max_retries} | "
        f"Timeout: {spec.config.timeout_per_sample_sec}s"
    )

    asyncio.run(run_orchestrator(spec))


if __name__ == "__main__":
    main()
