"""CLI entry point for natural-language browser automation.

Usage:
    python main.py \
        --instruction "Go to each Jira ticket URL, take a screenshot, ..." \
        --csv tickets.csv \
        --workers 1

    python main.py --auth-status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

from dotenv import load_dotenv

from orchestrator import run_orchestrator

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Browser automation agent — give it an instruction and a CSV"
    )
    parser.add_argument(
        "--instruction", "-i",
        default=None,
        help="Natural language instruction describing what to do per CSV row",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to the input CSV file",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=3,
        help="Max concurrent browser workers (default: 3)",
    )
    parser.add_argument(
        "--auth-state",
        default=None,
        help="Path to storage state JSON (from extension or login.py). "
        "Auto-discovered from ~/.agent-auth/state.json if not set.",
    )
    # Keep --session as hidden alias for backwards compat
    parser.add_argument(
        "--session", "-s",
        default=None,
        dest="session_legacy",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--auth-status",
        action="store_true",
        help="Print auth state info and exit",
    )
    parser.add_argument(
        "--auth-check",
        nargs="*",
        metavar="URL",
        help="Health-check auth sessions against URLs and exit "
        "(e.g. --auth-check https://github.com)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Output directory (default: auto-generated from task name)",
    )
    parser.add_argument(
        "--model", "-m",
        default=None,
        help="Model for the worker agent (default: claude-sonnet-4-5-20250929)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=30,
        help="Max iterations per sample (default: 30)",
    )
    parser.add_argument(
        "--pioneers", "-p",
        type=int,
        default=1,
        help="Number of pioneer agents for tournament mode (default: 1, no tournament)",
    )
    parser.add_argument(
        "--browserbase", "-bb",
        action="store_true",
        help="Use Browserbase cloud browsers with session replay",
    )
    parser.add_argument(
        "--skills",
        action="store_true",
        help="Enable SkillRegistry: planner selects skills that filter tools and inject guidance into worker prompts",
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

    # --- Auth status command ---
    if args.auth_status:
        from auth import print_auth_status
        sys.exit(0 if print_auth_status() else 1)

    # --- Auth health check command ---
    if args.auth_check is not None:
        from auth import find_storage_state, health_check_with_state
        from pathlib import Path

        state_path = Path(args.auth_state) if args.auth_state else find_storage_state()
        if not state_path or not state_path.exists():
            print("No storage state found. Export from the Agent Auth extension first.")
            sys.exit(1)

        if not args.auth_check:
            print("Provide URLs to check: --auth-check https://github.com")
            sys.exit(1)

        async def _check():
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                results = await health_check_with_state(
                    p, state_path, args.auth_check
                )
            for url, result in results.items():
                sym = {"ok": "+", "expired": "!", "error": "x"}[result["status"]]
                print(f"  [{sym}] {url} -> {result['status']} ({result['details']})")
            return all(r["status"] == "ok" for r in results.values())

        sys.exit(0 if asyncio.run(_check()) else 1)

    # --- Main run command ---
    if not args.instruction or not args.csv:
        parser.error("--instruction and --csv are required (unless using --auth-status or --auth-check)")

    # Resolve auth state: explicit --auth-state > legacy --session > auto-discover
    auth_state = args.auth_state or args.session_legacy

    asyncio.run(run_orchestrator(
        instruction=args.instruction,
        csv_path=args.csv,
        max_workers=args.workers,
        output_dir_override=args.output_dir,
        session_state_path=auth_state,
        model_override=args.model,
        num_pioneers=args.pioneers,
        max_iterations=args.max_iterations,
        browserbase=args.browserbase,
        use_skills=args.skills,
    ))


if __name__ == "__main__":
    main()
