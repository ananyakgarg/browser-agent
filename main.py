"""CLI entry point for natural-language browser automation.

Usage:
    python main.py \
        --instruction "Go to each Jira ticket URL, take a screenshot, ..." \
        --csv tickets.csv \
        --workers 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from dotenv import load_dotenv

from orchestrator import run_orchestrator

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Browser automation agent — give it an instruction and a CSV"
    )
    parser.add_argument(
        "--instruction", "-i",
        required=True,
        help="Natural language instruction describing what to do per CSV row",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Max concurrent browser workers (default: 3)",
    )
    parser.add_argument(
        "--session", "-s",
        default=None,
        help="Path to session_state.json from login.py (for authenticated sites)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: auto-generated from task name)",
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

    asyncio.run(run_orchestrator(
        instruction=args.instruction,
        csv_path=args.csv,
        max_workers=args.workers,
        output_dir_override=args.output_dir,
        session_state_path=args.session,
    ))


if __name__ == "__main__":
    main()
