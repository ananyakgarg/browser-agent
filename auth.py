"""Authentication via Chrome extension storage state export.

The Agent Auth Bridge extension exports cookies + localStorage as a Playwright
storage state JSON file. The agent reads this file and passes it to
browser.new_context(storage_state=...).

No Chrome profile copying, no persistent contexts, no channel="chrome".
Standard Playwright contexts with injected state.
"""

import json
import logging
import os
import stat
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Primary location — extension auto-exports here or user moves the file
AGENT_STATE_PATH = Path.home() / ".agent-auth" / "state.json"
# Fallback — where the extension downloads by default
DOWNLOADS_STATE_PATH = Path.home() / "Downloads" / "agent-auth-state.json"

# Max age before we warn (seconds) — 24 hours
MAX_STATE_AGE_SEC = 86400


def find_storage_state() -> Path | None:
    """Find the most recent storage state file.

    Checks ~/.agent-auth/state.json first, then ~/Downloads/agent-auth-state.json.
    Returns the most recently modified valid file, or None.
    """
    candidates = [AGENT_STATE_PATH, DOWNLOADS_STATE_PATH]
    valid = [p for p in candidates if p.exists()]

    if not valid:
        return None

    # Return the most recently modified one
    best = max(valid, key=lambda p: p.stat().st_mtime)

    # Restrict permissions on the file (contains session tokens)
    _secure_file(best)

    return best


def _secure_file(path: Path) -> None:
    """Set file permissions to owner-only (chmod 600)."""
    try:
        current = path.stat().st_mode
        if current & (stat.S_IRGRP | stat.S_IROTH):
            os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
            logger.debug(f"Secured {path} (chmod 600)")
    except OSError:
        pass


def validate_storage_state(path: Path) -> dict[str, Any]:
    """Validate a storage state file and return summary info.

    Returns:
        {
            "valid": bool,
            "path": str,
            "age_seconds": float,
            "cookie_count": int,
            "origin_count": int,
            "domains": list[str],
            "warnings": list[str],
        }
    """
    warnings: list[str] = []

    if not path.exists():
        return {"valid": False, "path": str(path), "warnings": ["File not found"]}

    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        return {"valid": False, "path": str(path), "warnings": [f"Invalid JSON: {e}"]}

    cookies = data.get("cookies", [])
    origins = data.get("origins", [])

    # Check age
    age = time.time() - path.stat().st_mtime
    if age > MAX_STATE_AGE_SEC:
        hours = age / 3600
        warnings.append(
            f"Storage state is {hours:.1f} hours old. "
            f"Re-export for fresh sessions."
        )

    # Check for expired cookies
    now = time.time()
    expired = sum(1 for c in cookies if 0 < c.get("expires", -1) < now)
    if expired > 0:
        warnings.append(f"{expired}/{len(cookies)} cookies are expired")

    # Unique domains
    domains = sorted(set(
        c["domain"].lstrip(".") for c in cookies if c.get("domain")
    ))

    return {
        "valid": True,
        "path": str(path),
        "age_seconds": age,
        "cookie_count": len(cookies),
        "origin_count": len(origins),
        "domains": domains,
        "warnings": warnings,
    }


def print_auth_status() -> bool:
    """Print auth status to stdout. Returns True if valid state found."""
    path = find_storage_state()
    if not path:
        print("No storage state found.")
        print("  1. Install the Agent Auth extension in Chrome")
        print("  2. Click the extension icon and Export to File")
        print(f"  3. Move the file to {AGENT_STATE_PATH}")
        return False

    info = validate_storage_state(path)
    if not info["valid"]:
        print(f"Invalid storage state: {info['path']}")
        for w in info["warnings"]:
            print(f"  {w}")
        return False

    age_min = info["age_seconds"] / 60
    print(f"Storage state: {info['path']}")
    print(f"  Age: {age_min:.0f} min")
    print(f"  Cookies: {info['cookie_count']}")
    print(f"  Origins: {info['origin_count']}")
    print(f"  Domains: {', '.join(info['domains'][:10])}")
    if len(info["domains"]) > 10:
        print(f"  ... and {len(info['domains']) - 10} more")
    if info["warnings"]:
        for w in info["warnings"]:
            print(f"  Warning: {w}")
    return True


async def health_check_with_state(
    playwright,
    storage_state_path: Path,
    urls: list[str],
    timeout_ms: int = 15000,
) -> dict[str, dict]:
    """Validate sessions by loading storage state and checking each URL.

    Launches a headless browser, loads storage state, navigates to each URL,
    and checks whether we land on a login page.

    Returns: {url: {"status": "ok"|"expired"|"error", "final_url": str, "details": str}}
    """
    results = {}
    browser = await playwright.chromium.launch(headless=True)

    try:
        context = await browser.new_context(
            storage_state=str(storage_state_path),
        )
        page = await context.new_page()

        login_indicators = [
            "/login", "/signin", "/sign-in", "/auth",
            "/sso", "accounts.google.com", "/oauth",
            "login.microsoftonline.com",
        ]

        for url in urls:
            try:
                response = await page.goto(
                    url, wait_until="domcontentloaded", timeout=timeout_ms
                )
                final_url = page.url

                is_login = any(
                    ind in final_url.lower() for ind in login_indicators
                )

                if is_login:
                    results[url] = {
                        "status": "expired",
                        "final_url": final_url,
                        "details": "Redirected to login page",
                    }
                elif response and response.status >= 400:
                    results[url] = {
                        "status": "error",
                        "final_url": final_url,
                        "details": f"HTTP {response.status}",
                    }
                else:
                    results[url] = {
                        "status": "ok",
                        "final_url": final_url,
                        "details": f"HTTP {response.status if response else '?'}",
                    }
            except Exception as e:
                results[url] = {
                    "status": "error",
                    "final_url": url,
                    "details": str(e),
                }

        await context.close()
    finally:
        await browser.close()

    return results
