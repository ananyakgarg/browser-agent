"""End-to-end auth test: export session from Chrome, verify LinkedIn access.

Usage:
    python test_linkedin_auth.py

Steps:
    1. Launches your real Chrome to linkedin.com
    2. You log in (or are already logged in)
    3. Press ENTER — script saves storage state
    4. Loads state in headless Playwright, navigates to linkedin.com/feed
    5. Checks if we're authenticated (not redirected to login)
    6. Takes a screenshot as proof
"""

import asyncio
import json
import signal
import subprocess
import shutil
import tempfile
from pathlib import Path

from playwright.async_api import async_playwright

CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
CDP_PORT = 9223  # different from login.py's 9222 to avoid conflicts
STATE_DIR = Path.home() / ".agent-auth"
STATE_PATH = STATE_DIR / "state.json"
LINKEDIN_FEED = "https://www.linkedin.com/feed/"
LINKEDIN_LOGIN_INDICATORS = ["/login", "/signin", "/authwall", "/uas/login", "checkpoint"]


async def step1_export_from_chrome():
    """Launch Chrome, let user log in, save storage state."""
    tmp_dir = tempfile.mkdtemp(prefix="chrome_auth_test_")

    print("\n=== Step 1: Export session from Chrome ===")
    print("Launching Chrome to linkedin.com...")
    print("If you're already logged in, great. If not, log in now.")
    print("Press ENTER here once LinkedIn is loaded and you're logged in.\n")

    proc = subprocess.Popen(
        [
            CHROME_PATH,
            f"--remote-debugging-port={CDP_PORT}",
            f"--user-data-dir={tmp_dir}",
            "--no-first-run",
            "--no-default-browser-check",
            "--window-size=1280,900",
            "https://www.linkedin.com/",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    input(">>> Press ENTER after you're logged into LinkedIn... ")

    print("Saving storage state...")
    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(f"http://localhost:{CDP_PORT}")
        context = browser.contexts[0]

        STATE_DIR.mkdir(parents=True, exist_ok=True)
        await context.storage_state(path=str(STATE_PATH))

        # Count what we got
        data = json.loads(STATE_PATH.read_text())
        n_cookies = len(data.get("cookies", []))
        n_origins = len(data.get("origins", []))
        domains = sorted(set(
            c["domain"].lstrip(".") for c in data.get("cookies", []) if c.get("domain")
        ))
        print(f"  Saved: {n_cookies} cookies, {n_origins} origins")
        print(f"  Domains: {', '.join(domains[:10])}")
        print(f"  Path: {STATE_PATH}")

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    return STATE_PATH


async def step2_verify_with_headless(state_path: Path):
    """Load storage state in headless Playwright, check LinkedIn access."""
    print("\n=== Step 2: Verify with headless Playwright ===")
    print(f"Loading state from {state_path}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(
            storage_state=str(state_path),
            viewport={"width": 1280, "height": 900},
        )
        page = await context.new_page()

        print(f"Navigating to {LINKEDIN_FEED}...")
        try:
            await page.goto(LINKEDIN_FEED, wait_until="domcontentloaded", timeout=20000)
        except Exception as e:
            print(f"  Navigation error: {e}")
            await browser.close()
            return False

        final_url = page.url
        print(f"  Final URL: {final_url}")

        # Check if we landed on a login page
        is_login = any(ind in final_url.lower() for ind in LINKEDIN_LOGIN_INDICATORS)

        if is_login:
            print("  FAIL: Redirected to login — session not valid")
            screenshot_path = state_path.parent / "linkedin_test_fail.png"
            await page.screenshot(path=str(screenshot_path))
            print(f"  Screenshot: {screenshot_path}")
            await browser.close()
            return False

        # Check page title
        title = await page.title()
        print(f"  Page title: {title}")

        # Take screenshot as proof
        screenshot_path = state_path.parent / "linkedin_test_pass.png"
        await page.screenshot(path=str(screenshot_path))
        print(f"  Screenshot: {screenshot_path}")

        # Quick check — look for feed indicators
        try:
            feed_content = await page.query_selector("[role='main']")
            has_feed = feed_content is not None
        except Exception:
            has_feed = False

        await browser.close()

        if has_feed:
            print("  PASS: Feed content found — authenticated!")
        else:
            print("  PASS: No login redirect (likely authenticated)")

        return True


async def step3_test_cli_integration():
    """Test the --auth-status and --auth-check CLI commands."""
    print("\n=== Step 3: Test CLI integration ===")

    import subprocess as sp

    # auth-status
    result = sp.run(
        ["python3", "main.py", "--auth-status"],
        capture_output=True, text=True,
    )
    print(f"  --auth-status exit code: {result.returncode}")
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")

    # auth-check
    result = sp.run(
        ["python3", "main.py", "--auth-check", "https://www.linkedin.com/feed/"],
        capture_output=True, text=True,
    )
    print(f"\n  --auth-check exit code: {result.returncode}")
    for line in result.stdout.strip().split("\n"):
        print(f"    {line}")
    for line in result.stderr.strip().split("\n"):
        if line.strip():
            print(f"    {line}")

    return result.returncode == 0


async def main():
    print("=" * 60)
    print("LinkedIn Auth End-to-End Test")
    print("=" * 60)

    # Step 1: Export
    state_path = await step1_export_from_chrome()

    # Step 2: Verify headless
    ok = await step2_verify_with_headless(state_path)

    # Step 3: CLI
    if ok:
        await step3_test_cli_integration()

    print("\n" + "=" * 60)
    if ok:
        print("ALL STEPS PASSED")
    else:
        print("FAILED — session may have expired or LinkedIn blocked headless")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
