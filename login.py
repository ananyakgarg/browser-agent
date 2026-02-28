"""Interactive login via real Chrome (bypasses Google's automation detection).

Playwright-launched browsers get blocked by Google SSO. This script launches
your actual Chrome with --remote-debugging-port, you log in normally, then
we connect via CDP to save the session state for the agent workers.

Usage:
    1. Quit Chrome if it's running
    2. python login.py --url https://ananyakgarg25.atlassian.net -o session_state.json
    3. Log in via Google SSO as usual, press ENTER when done
"""

from __future__ import annotations

import argparse
import asyncio
import shutil
import signal
import subprocess
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CDP_PORT = 9222


async def do_login(url: str, output: str):
    tmp_dir = tempfile.mkdtemp(prefix="chrome_login_")

    print(f"\nLaunching Chrome to: {url}")
    print("Log in normally (Google SSO works here).")
    print("Press ENTER here once you're logged in.\n")

    proc = subprocess.Popen(
        [
            CHROME_PATH,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1280,900",
            url,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    input(">>> Press ENTER after you're logged in... ")

    print("Saving session...")
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
        context = browser.contexts[0]
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=output)
        print(f"Session saved to {output}")

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("Done.\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive login — save browser session")
    parser.add_argument("--url", required=True, help="URL to open for login")
    parser.add_argument("--output", "-o", default="session_state.json", help="Where to save session state")
    args = parser.parse_args()
    asyncio.run(do_login(args.url, args.output))


if __name__ == "__main__":
    main()
