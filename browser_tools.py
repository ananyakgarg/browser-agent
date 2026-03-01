"""Playwright browser tool provider with element extraction and page state.

Each tool is registered as a (schema, handler) pair in __init__.
To add a new browser tool: define the schema dict, write the async handler,
call self.register_tool(schema, handler).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Download,
    ElementHandle,
    Page,
    Playwright,
)

from site_memory import SiteMemory
from tool_registry import ToolProvider

logger = logging.getLogger(__name__)

_site_memory = SiteMemory()

TEXT_TRUNCATE_LIMIT = 4000
FULL_TEXT_LIMIT = 8000

INTERACTIVE_SELECTOR = (
    "a[href], button, input, textarea, select, "
    "[role='button'], [role='link'], [role='tab'], [role='menuitem'], "
    "[role='checkbox'], [role='radio'], [role='switch'], "
    "[onclick], [tabindex]:not([tabindex='-1']), "
    "summary, details"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ElementInfo:
    id: int
    tag: str
    role: str
    text: str
    attributes: dict[str, str]
    handle: ElementHandle
    depth: int = 0  # DOM nesting depth for indentation


@dataclass
class PageState:
    url: str
    title: str
    elements: list[dict[str, Any]]
    visible_text: str
    tab_count: int

    def to_text(self) -> str:
        lines = [
            f"URL: {self.url}",
            f"Title: {self.title}",
            f"Tabs open: {self.tab_count}",
            "",
            "=== Interactive Elements ===",
        ]

        # Normalize depths: find min depth and subtract so shallowest = 0
        depths = [el.get("depth", 0) for el in self.elements]
        min_depth = min(depths) if depths else 0

        for el in self.elements:
            indent_level = max(0, el.get("depth", 0) - min_depth)
            indent = "\t" * min(indent_level, 4)  # cap at 4 levels

            # *[ prefix for new elements
            is_new = el.get("is_new", False)
            id_prefix = f"*[{el['id']}]" if is_new else f"[{el['id']}]"

            parts = [f"{indent}{id_prefix}", f"<{el['tag']}>"]
            if el.get("role"):
                parts.append(f"role={el['role']}")
            if el.get("type"):
                parts.append(f"type={el['type']}")
            if el.get("name"):
                parts.append(f"name=\"{el['name']}\"")
            if el.get("placeholder"):
                parts.append(f"placeholder=\"{el['placeholder']}\"")
            if el.get("value"):
                parts.append(f"value=\"{el['value'][:60]}\"")
            if el.get("href"):
                parts.append(f"href=\"{el['href'][:80]}\"")
            if el.get("text"):
                parts.append(f"\"{el['text'][:80]}\"")
            if el.get("aria_label"):
                parts.append(f"aria-label=\"{el['aria_label']}\"")
            if el.get("checked") is not None:
                parts.append(f"checked={el['checked']}")
            if el.get("disabled"):
                parts.append("DISABLED")
            lines.append(" ".join(parts))

        if not self.elements:
            lines.append("(no interactive elements found)")

        # Count new elements for quick summary
        new_count = sum(1 for el in self.elements if el.get("is_new"))
        if new_count > 0:
            lines.append(f"\n({new_count} new elements marked with *)")

        lines.extend(["", "=== Page Text (truncated) ===", self.visible_text])
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool schemas — co-located with their handlers below
# ---------------------------------------------------------------------------

_NAVIGATE_SCHEMA = {
    "name": "navigate",
    "description": "Navigate the browser to a URL.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "The URL to navigate to."},
        },
        "required": ["url"],
    },
}

_CLICK_SCHEMA = {
    "name": "click",
    "description": "Click on an interactive element by its ID number from the page state.",
    "input_schema": {
        "type": "object",
        "properties": {
            "element_id": {"type": "integer", "description": "The ID number of the element to click."},
        },
        "required": ["element_id"],
    },
}

_TYPE_TEXT_SCHEMA = {
    "name": "type_text",
    "description": "Type text into an input element. Clears existing content first.",
    "input_schema": {
        "type": "object",
        "properties": {
            "element_id": {"type": "integer", "description": "The ID number of the input element."},
            "text": {"type": "string", "description": "The text to type."},
            "press_enter": {
                "type": "boolean",
                "description": "Whether to press Enter after typing. Defaults to false.",
                "default": False,
            },
        },
        "required": ["element_id", "text"],
    },
}

_SCREENSHOT_SCHEMA = {
    "name": "screenshot",
    "description": "Take a screenshot of the current viewport. Only use when the task instructions explicitly require a screenshot or an output field requires an image. Use get_text for all other data extraction.",
    "input_schema": {
        "type": "object",
        "properties": {
            "filename": {
                "type": "string",
                "description": "Filename for the screenshot (without path). Defaults to 'screenshot.png'.",
                "default": "screenshot.png",
            },
        },
        "required": [],
    },
}

_SCROLL_SCHEMA = {
    "name": "scroll",
    "description": "Scroll the page up or down.",
    "input_schema": {
        "type": "object",
        "properties": {
            "direction": {"type": "string", "enum": ["up", "down"], "description": "Direction to scroll."},
            "amount": {"type": "integer", "description": "Pixels to scroll. Defaults to 500.", "default": 500},
        },
        "required": ["direction"],
    },
}

_GO_BACK_SCHEMA = {
    "name": "go_back",
    "description": "Navigate back to the previous page.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

_GET_TEXT_SCHEMA = {
    "name": "get_text",
    "description": "Get the full visible text content of the current page (up to ~8000 chars). Use when the truncated text in page state isn't enough.",
    "input_schema": {"type": "object", "properties": {}, "required": []},
}

_SELECT_OPTION_SCHEMA = {
    "name": "select_option",
    "description": "Select an option from a <select> dropdown by its value or visible text.",
    "input_schema": {
        "type": "object",
        "properties": {
            "element_id": {"type": "integer", "description": "The ID number of the select element."},
            "value": {"type": "string", "description": "The value or visible text of the option to select."},
        },
        "required": ["element_id", "value"],
    },
}

_DOWNLOAD_FILE_SCHEMA = {
    "name": "download_file",
    "description": "Click an element and wait for a file download. Returns the download path.",
    "input_schema": {
        "type": "object",
        "properties": {
            "element_id": {"type": "integer", "description": "The ID number of the element that triggers the download."},
        },
        "required": ["element_id"],
    },
}

_SWITCH_TAB_SCHEMA = {
    "name": "switch_tab",
    "description": "Switch to a different browser tab by index (0-based).",
    "input_schema": {
        "type": "object",
        "properties": {
            "tab_index": {"type": "integer", "description": "The 0-based index of the tab to switch to."},
        },
        "required": ["tab_index"],
    },
}

_FIND_ELEMENT_SCHEMA = {
    "name": "find_element",
    "description": "Search for elements matching a CSS selector or text content. Returns matching elements with their IDs.",
    "input_schema": {
        "type": "object",
        "properties": {
            "selector": {"type": "string", "description": "CSS selector to search for."},
            "text": {"type": "string", "description": "Text content to search for (partial match)."},
        },
        "required": [],
    },
}

_WAIT_SCHEMA = {
    "name": "wait",
    "description": "Wait for a specified duration or for an element to appear.",
    "input_schema": {
        "type": "object",
        "properties": {
            "seconds": {"type": "number", "description": "Seconds to wait. Defaults to 2.", "default": 2},
            "selector": {"type": "string", "description": "Optional CSS selector to wait for."},
        },
        "required": [],
    },
}

_EXECUTE_JS_SCHEMA = {
    "name": "execute_js",
    "description": (
        "Execute a JavaScript expression in the browser and return the result. "
        "Use for targeted data extraction from the DOM — e.g. "
        "'document.querySelector(\"h1\").innerText', "
        "'[...document.querySelectorAll(\"table tr\")].map(r => r.innerText)', "
        "'document.querySelector(\".status\").getAttribute(\"data-value\")'. "
        "The result is JSON-serialized. Much more precise than get_text."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "JavaScript expression to evaluate. Should return a serializable value.",
            },
        },
        "required": ["expression"],
    },
}

_HOVER_SCHEMA = {
    "name": "hover",
    "description": "Hover over an interactive element by its ID number. Use for dropdown menus, tooltips, and hover-triggered UI.",
    "input_schema": {
        "type": "object",
        "properties": {
            "element_id": {
                "type": "integer",
                "description": "The ID number of the element to hover over.",
            },
        },
        "required": ["element_id"],
    },
}

_SEARCH_PAGE_SCHEMA = {
    "name": "search_page",
    "description": (
        "Search the FULL page text for a string or regex pattern. Returns all matches "
        "with surrounding context (±2 lines) and approximate character positions. "
        "Free and instant — use this BEFORE scrolling or execute_js when looking for "
        "specific text, code, dates, IDs, or error messages on a page. "
        "Works even on content not visible in the viewport."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text or regex pattern to search for (case-insensitive).",
            },
            "max_results": {
                "type": "integer",
                "description": "Max matches to return. Defaults to 10.",
                "default": 10,
            },
        },
        "required": ["query"],
    },
}

_PRESS_KEY_SCHEMA = {
    "name": "press_key",
    "description": (
        "Press a keyboard key or key combination. "
        "Key names: Enter, Escape, Tab, Backspace, Delete, Space, "
        "ArrowDown, ArrowUp, ArrowLeft, ArrowRight, Home, End, PageUp, PageDown. "
        "Modifier combos with '+': Control+a, Meta+c, Shift+Tab, Alt+F4. "
        "Use Meta for Cmd on Mac."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "key": {
                "type": "string",
                "description": "Key or combo to press (e.g. 'Escape', 'Control+a', 'Shift+Tab').",
            },
        },
        "required": ["key"],
    },
}

_CONFIGURE_BROWSER_SCHEMA = {
    "name": "configure_browser",
    "description": (
        "Modify browser configuration at runtime. Use this when you detect "
        "that a site is blocking you (bot detection, CAPTCHA, 403 errors, "
        "'automated tool' messages). You can change user-agent, enable stealth "
        "mode, inject custom scripts, set extra headers, or clear cookies. "
        "Changes apply to all future page loads in this session."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["set_user_agent", "enable_stealth", "inject_script", "set_headers", "clear_cookies", "restart_context"],
                "description": (
                    "Action to take: "
                    "set_user_agent — change the browser's user-agent string. "
                    "enable_stealth — inject anti-detection scripts (webdriver=false, fake plugins, etc). "
                    "inject_script — run custom JS on every page load. "
                    "set_headers — add extra HTTP headers to all requests. "
                    "clear_cookies — clear all cookies. "
                    "restart_context — close and reopen browser context with clean state."
                ),
            },
            "value": {
                "type": "string",
                "description": (
                    "The value for the action. "
                    "For set_user_agent: the user-agent string. "
                    "For inject_script: JavaScript code to run on every page load. "
                    "For set_headers: JSON object of header name-value pairs. "
                    "For enable_stealth/clear_cookies/restart_context: not needed."
                ),
                "default": "",
            },
        },
        "required": ["action"],
    },
}

_SAVE_SITE_WORKAROUND_SCHEMA = {
    "name": "save_site_workaround",
    "description": (
        "Save a workaround you discovered for a website so future agents "
        "automatically apply it. Call this after you successfully work around "
        "a site's bot detection, rate limiting, or other blocking. "
        "The workaround is saved permanently and loaded for any future visits "
        "to this domain."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "description": "The domain (e.g. 'sec.gov', 'linkedin.com')",
            },
            "workaround": {
                "type": "object",
                "description": (
                    "Key-value pairs describing the workaround. Common keys: "
                    "blocks_headless (bool), needs_stealth (bool), needs_headers (object), "
                    "rate_limited (bool), needs_auth (bool), workaround (string description)"
                ),
            },
        },
        "required": ["domain", "workaround"],
    },
}


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class BrowserToolProvider(ToolProvider):
    """Playwright-based browser tools.

    Each tool is a (schema, handler) pair registered in __init__.
    The handler receives the raw tool_input dict and returns a string.
    """

    def __init__(self, context: BrowserContext, output_dir: Path):
        super().__init__()
        self.context = context
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._elements: dict[int, ElementInfo] = {}
        self._page: Page | None = None
        self._screenshot_count = 0
        self._prev_element_sigs: set[str] = set()  # for new-element detection

        # ----- register browser tools -----
        # Core interaction
        self.register_tool(_NAVIGATE_SCHEMA, self._handle_navigate)
        self.register_tool(_CLICK_SCHEMA, self._handle_click)
        self.register_tool(_TYPE_TEXT_SCHEMA, self._handle_type_text)
        self.register_tool(_SCREENSHOT_SCHEMA, self._handle_screenshot)
        self.register_tool(_HOVER_SCHEMA, self._handle_hover)
        self.register_tool(_PRESS_KEY_SCHEMA, self._handle_press_key)
        self.register_tool(_WAIT_SCHEMA, self._handle_wait)
        # Data extraction & scripting
        self.register_tool(_EXECUTE_JS_SCHEMA, self._handle_execute_js)
        self.register_tool(_SEARCH_PAGE_SCHEMA, self._handle_search_page)
        # File & tab management
        self.register_tool(_DOWNLOAD_FILE_SCHEMA, self._handle_download_file)
        self.register_tool(_SWITCH_TAB_SCHEMA, self._handle_switch_tab)
        # Self-healing: browser reconfiguration & site memory
        self.register_tool(_CONFIGURE_BROWSER_SCHEMA, self._handle_configure_browser)
        self.register_tool(_SAVE_SITE_WORKAROUND_SCHEMA, self._handle_save_site_workaround)

    async def close(self) -> None:
        try:
            await self.context.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Page / element helpers (not tools — internal only)
    # ------------------------------------------------------------------

    @property
    def page(self) -> Page:
        if self._page is None:
            raise RuntimeError("No page open. Call navigate first.")
        return self._page

    async def _ensure_page(self) -> Page:
        if self._page is None or self._page.is_closed():
            pages = self.context.pages
            if pages:
                self._page = pages[-1]
            else:
                self._page = await self.context.new_page()
        return self._page

    async def _extract_elements(self) -> list[ElementInfo]:
        page = await self._ensure_page()
        self._elements.clear()

        try:
            handles = await page.query_selector_all(INTERACTIVE_SELECTOR)
        except Exception as e:
            logger.warning(f"Element extraction failed: {e}")
            return []

        elements: list[ElementInfo] = []
        eid = 0

        for handle in handles:
            try:
                if not await handle.is_visible():
                    continue

                tag = await handle.evaluate("el => el.tagName.toLowerCase()")
                role = await handle.get_attribute("role") or ""
                text_content = await handle.evaluate(
                    "el => (el.innerText || el.textContent || '').trim().substring(0, 100)"
                )
                # DOM depth for indentation (count parents up to body)
                depth = await handle.evaluate(
                    "el => { let d=0, n=el; while(n.parentElement && n.parentElement !== document.body) { d++; n=n.parentElement; } return d; }"
                )
                attrs: dict[str, str] = {}

                for attr in ("type", "name", "placeholder", "value", "href", "aria-label", "id", "class"):
                    val = await handle.get_attribute(attr)
                    if val:
                        attrs[attr] = val

                if tag in ("input", "select", "textarea"):
                    val = await handle.evaluate("el => el.value || ''")
                    if val:
                        attrs["value"] = val

                if tag == "input" and attrs.get("type") in ("checkbox", "radio"):
                    checked = await handle.evaluate("el => el.checked")
                    attrs["checked"] = str(checked).lower()

                disabled = await handle.evaluate("el => el.disabled || false")
                if disabled:
                    attrs["disabled"] = "true"

                info = ElementInfo(id=eid, tag=tag, role=role, text=text_content, attributes=attrs, handle=handle, depth=depth)
                self._elements[eid] = info
                elements.append(info)
                eid += 1
            except Exception:
                continue

        return elements

    async def get_page_state(self) -> PageState:
        page = await self._ensure_page()

        try:
            url = page.url
            title = await page.title()
        except Exception:
            url = "about:blank"
            title = ""

        elements = await self._extract_elements()

        try:
            visible_text = await page.evaluate(
                "() => document.body ? document.body.innerText.substring(0, %d) : ''" % TEXT_TRUNCATE_LIMIT
            )
        except Exception:
            visible_text = ""

        element_dicts = []
        for el in elements:
            d: dict[str, Any] = {"id": el.id, "tag": el.tag, "depth": el.depth}
            if el.role:
                d["role"] = el.role
            if el.text:
                d["text"] = el.text
            for key in ("type", "name", "placeholder", "value", "href", "aria-label"):
                if key in el.attributes:
                    d[key.replace("-", "_")] = el.attributes[key]
            if "checked" in el.attributes:
                d["checked"] = el.attributes["checked"] == "true"
            if el.attributes.get("disabled") == "true":
                d["disabled"] = True
            element_dicts.append(d)

        # Detect new elements by comparing signatures with previous state
        current_sigs: set[str] = set()
        for el, d in zip(elements, element_dicts):
            sig = f"{el.tag}|{el.role}|{el.text[:40]}|{el.attributes.get('href', '')}|{el.attributes.get('name', '')}"
            current_sigs.add(sig)
            d["is_new"] = sig not in self._prev_element_sigs
        self._prev_element_sigs = current_sigs

        return PageState(
            url=url,
            title=title,
            elements=element_dicts,
            visible_text=visible_text,
            tab_count=len(self.context.pages),
        )

    def _get_element(self, element_id: int) -> ElementInfo:
        if element_id not in self._elements:
            raise ValueError(
                f"Element ID {element_id} not found. Available IDs: {sorted(self._elements.keys())[:20]}"
            )
        return self._elements[element_id]

    # ------------------------------------------------------------------
    # Tool handlers — each takes tool_input dict, returns str
    # ------------------------------------------------------------------

    async def _handle_navigate(self, tool_input: dict[str, Any]) -> str:
        url = tool_input["url"]

        # Check site memory and auto-apply known workarounds before navigating
        hint = _site_memory.get_hint_for_url(url)
        auto_applied = await self._auto_apply_workarounds(url)

        page = await self._ensure_page()
        try:
            response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(500)
        except Exception as e:
            return f"Navigation error: {e}"

        state = await self.get_page_state()
        result = state.to_text()

        # Detect block pages by checking visible text for common patterns
        block_warning = self._detect_block(state, response)

        parts = []
        if auto_applied:
            parts.append(f"ℹ Auto-applied workarounds: {', '.join(auto_applied)}")
        if hint:
            parts.append(f"⚠ SITE MEMORY:\n{hint}\n")
        if block_warning:
            parts.append(block_warning)
        parts.append(result)
        return "\n".join(parts)

    async def _auto_apply_workarounds(self, url: str) -> list[str]:
        """Auto-apply known workarounds from site memory before navigation."""
        applied = []
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            for d in [domain, ".".join(domain.split(".")[-2:])]:
                mem = _site_memory.get(d)
                if not mem:
                    continue
                if mem.get("needs_stealth"):
                    await self.context.add_init_script(_STEALTH_SCRIPT)
                    applied.append("stealth mode")
                if mem.get("needs_headers"):
                    headers = mem["needs_headers"]
                    if isinstance(headers, dict):
                        await self.context.set_extra_http_headers(headers)
                        applied.append(f"custom headers ({list(headers.keys())})")
                if applied:
                    break  # applied from first matching domain
        except Exception as e:
            logger.debug(f"Auto-apply workarounds error: {e}")
        return applied

    def _detect_block(self, state: PageState, response: Any) -> str | None:
        """Check if the page looks like a block/bot-detection page."""
        signals = []

        # Check HTTP status
        if response and hasattr(response, "status"):
            if response.status == 403:
                signals.append("HTTP 403 Forbidden")
            elif response.status == 429:
                signals.append("HTTP 429 Too Many Requests")

        # Check page text for block patterns
        text_lower = state.visible_text.lower()
        title_lower = state.title.lower()
        for pattern in _BLOCK_PATTERNS:
            if pattern in text_lower or pattern in title_lower:
                signals.append(f"Page contains '{pattern}'")
                break  # one match is enough

        # Very few interactive elements + short text = likely a block page
        if len(state.elements) < 3 and len(state.visible_text) < 500:
            if any(s for s in signals):  # only flag if we also have another signal
                signals.append("Minimal page content (likely not the real page)")

        if signals:
            return (
                "⚠ POSSIBLE BLOCK DETECTED:\n"
                + "\n".join(f"  - {s}" for s in signals)
                + "\n\nTry: configure_browser(action='enable_stealth') then retry, "
                "or save_site_workaround() if you find a fix.\n"
            )
        return None

    async def _handle_click(self, tool_input: dict[str, Any]) -> str:
        el = self._get_element(tool_input["element_id"])
        try:
            await el.handle.scroll_into_view_if_needed(timeout=3000)
            await el.handle.click(timeout=5000)
            await self.page.wait_for_timeout(800)
        except Exception as e:
            return f"Click error on element {tool_input['element_id']}: {e}"
        return (await self.get_page_state()).to_text()

    async def _handle_type_text(self, tool_input: dict[str, Any]) -> str:
        el = self._get_element(tool_input["element_id"])
        try:
            await el.handle.scroll_into_view_if_needed(timeout=3000)
            await el.handle.click(timeout=3000)
            await el.handle.evaluate("el => el.value = ''")
            await el.handle.type(tool_input["text"], delay=30)
            if tool_input.get("press_enter", False):
                await self.page.keyboard.press("Enter")
                await self.page.wait_for_timeout(800)
        except Exception as e:
            return f"Type error on element {tool_input['element_id']}: {e}"
        return (await self.get_page_state()).to_text()

    async def _handle_screenshot(self, tool_input: dict[str, Any]) -> str:
        self._screenshot_count += 1
        filename = tool_input.get("filename", "screenshot.png")
        if filename == "screenshot.png":
            filename = f"screenshot_{self._screenshot_count}.png"
        path = self.output_dir / filename
        try:
            page = await self._ensure_page()
            await page.screenshot(path=str(path))
            return f"Screenshot saved to {path}"
        except Exception as e:
            return f"Screenshot error: {e}"

    async def _handle_scroll(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        amount = tool_input.get("amount", 500)
        delta = amount if tool_input["direction"] == "down" else -amount
        try:
            await page.evaluate(f"window.scrollBy(0, {delta})")
            await page.wait_for_timeout(300)
        except Exception as e:
            return f"Scroll error: {e}"
        return (await self.get_page_state()).to_text()

    async def _handle_go_back(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        try:
            await page.go_back(wait_until="domcontentloaded", timeout=15000)
            await page.wait_for_timeout(500)
        except Exception as e:
            return f"Go back error: {e}"
        return (await self.get_page_state()).to_text()

    async def _handle_get_text(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        try:
            return await page.evaluate(
                "() => document.body ? document.body.innerText.substring(0, %d) : ''" % FULL_TEXT_LIMIT
            )
        except Exception as e:
            return f"Get text error: {e}"

    async def _handle_select_option(self, tool_input: dict[str, Any]) -> str:
        el = self._get_element(tool_input["element_id"])
        value = tool_input["value"]
        try:
            await el.handle.select_option(value=value, timeout=3000)
        except Exception:
            try:
                await el.handle.select_option(label=value, timeout=3000)
            except Exception as e:
                return f"Select error on element {tool_input['element_id']}: {e}"
        await self.page.wait_for_timeout(300)
        return (await self.get_page_state()).to_text()

    async def _handle_download_file(self, tool_input: dict[str, Any]) -> str:
        el = self._get_element(tool_input["element_id"])
        page = await self._ensure_page()
        try:
            async with page.expect_download(timeout=30000) as download_info:
                await el.handle.click(timeout=5000)
            download: Download = await download_info.value
            dest = self.output_dir / (download.suggested_filename or "download")
            await download.save_as(str(dest))
            return f"Downloaded file to {dest}"
        except Exception as e:
            return f"Download error: {e}"

    async def _handle_switch_tab(self, tool_input: dict[str, Any]) -> str:
        tab_index = tool_input["tab_index"]
        pages = self.context.pages
        if tab_index < 0 or tab_index >= len(pages):
            return f"Tab index {tab_index} out of range. {len(pages)} tabs open."
        self._page = pages[tab_index]
        try:
            await self._page.bring_to_front()
        except Exception:
            pass
        return (await self.get_page_state()).to_text()

    async def _handle_find_element(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        results = []
        selector = tool_input.get("selector")
        text = tool_input.get("text")

        if selector:
            try:
                handles = await page.query_selector_all(selector)
                for h in handles[:20]:
                    txt = await h.evaluate(
                        "el => (el.innerText || el.textContent || '').trim().substring(0, 100)"
                    )
                    tag = await h.evaluate("el => el.tagName.toLowerCase()")
                    matched_id = None
                    for eid, info in self._elements.items():
                        try:
                            same = await page.evaluate("(a, b) => a === b", [info.handle, h])
                            if same:
                                matched_id = eid
                                break
                        except Exception:
                            continue
                    results.append({"tag": tag, "text": txt[:80], "element_id": matched_id})
            except Exception as e:
                return f"Selector search error: {e}"

        if text:
            for eid, info in self._elements.items():
                if text.lower() in info.text.lower():
                    results.append({"element_id": eid, "tag": info.tag, "text": info.text[:80]})

        if not results:
            return "No matching elements found."
        return json.dumps(results[:20], indent=2)

    async def _handle_wait(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        seconds = tool_input.get("seconds", 2)
        selector = tool_input.get("selector")
        try:
            if selector:
                await page.wait_for_selector(selector, timeout=seconds * 1000)
                return f"Element matching '{selector}' appeared."
            else:
                await page.wait_for_timeout(seconds * 1000)
                return f"Waited {seconds} seconds."
        except Exception as e:
            return f"Wait error: {e}"

    async def _handle_execute_js(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        expression = tool_input["expression"]
        try:
            result = await page.evaluate(expression)
        except Exception as e:
            return f"JS execution error: {e}"
        try:
            serialized = json.dumps(result, indent=2, default=str)
        except (TypeError, ValueError):
            serialized = str(result)
        if len(serialized) > FULL_TEXT_LIMIT:
            serialized = serialized[:FULL_TEXT_LIMIT] + "\n... (truncated)"
        return serialized

    async def _handle_search_page(self, tool_input: dict[str, Any]) -> str:
        """Search full page text for a pattern. Returns matches with context."""
        import re
        page = await self._ensure_page()
        query = tool_input["query"]
        max_results = tool_input.get("max_results", 10)

        try:
            full_text = await page.evaluate("document.body.innerText")
        except Exception as e:
            return f"Search error: {e}"

        if not full_text:
            return "Page has no text content."

        lines = full_text.split("\n")
        matches = []
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            # Fall back to literal search
            pattern = re.compile(re.escape(query), re.IGNORECASE)

        for i, line in enumerate(lines):
            if pattern.search(line):
                # ±2 lines of context
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                context = "\n".join(
                    f"{'>>>' if j == i else '   '} L{j+1}: {lines[j]}"
                    for j in range(start, end)
                )
                matches.append(context)
                if len(matches) >= max_results:
                    break

        if not matches:
            return f"No matches found for '{query}' on this page."

        header = f"Found {len(matches)} match(es) for '{query}':\n\n"
        return header + "\n---\n".join(matches)

    async def _handle_hover(self, tool_input: dict[str, Any]) -> str:
        el = self._get_element(tool_input["element_id"])
        try:
            await el.handle.scroll_into_view_if_needed(timeout=3000)
            await el.handle.hover(timeout=5000)
            await self.page.wait_for_timeout(500)
        except Exception as e:
            return f"Hover error on element {tool_input['element_id']}: {e}"
        return (await self.get_page_state()).to_text()

    async def _handle_press_key(self, tool_input: dict[str, Any]) -> str:
        page = await self._ensure_page()
        key = tool_input["key"]
        try:
            await page.keyboard.press(key)
            await page.wait_for_timeout(500)
        except Exception as e:
            return f"Key press error: {e}"
        return (await self.get_page_state()).to_text()

    # ------------------------------------------------------------------
    # Self-healing: configure_browser & save_site_workaround
    # ------------------------------------------------------------------

    async def _handle_configure_browser(self, tool_input: dict[str, Any]) -> str:
        action = tool_input["action"]
        value = tool_input.get("value", "")

        if action == "enable_stealth":
            try:
                await self.context.add_init_script(_STEALTH_SCRIPT)
                # Also inject into current page immediately
                page = await self._ensure_page()
                await page.evaluate(_STEALTH_SCRIPT)
                return "Stealth mode enabled. Anti-detection scripts injected for all future page loads and current page."
            except Exception as e:
                return f"Stealth injection error: {e}"

        elif action == "set_user_agent":
            if not value:
                return "Error: value required — provide the user-agent string."
            # Can't change UA on existing context, but we can set it via init script
            try:
                await self.context.add_init_script(
                    f"Object.defineProperty(navigator, 'userAgent', {{get: () => '{value}'}});"
                )
                return f"User-agent override set to: {value}. Takes effect on next navigation."
            except Exception as e:
                return f"Set user-agent error: {e}"

        elif action == "inject_script":
            if not value:
                return "Error: value required — provide the JavaScript code."
            try:
                await self.context.add_init_script(value)
                page = await self._ensure_page()
                await page.evaluate(value)
                return "Custom script injected for all future page loads and current page."
            except Exception as e:
                return f"Script injection error: {e}"

        elif action == "set_headers":
            if not value:
                return "Error: value required — provide JSON object of headers."
            try:
                headers = json.loads(value)
                await self.context.set_extra_http_headers(headers)
                return f"Extra HTTP headers set: {list(headers.keys())}"
            except json.JSONDecodeError:
                return "Error: value must be valid JSON object (e.g. {\"Accept-Language\": \"en-US\"})"
            except Exception as e:
                return f"Set headers error: {e}"

        elif action == "clear_cookies":
            try:
                await self.context.clear_cookies()
                return "All cookies cleared."
            except Exception as e:
                return f"Clear cookies error: {e}"

        elif action == "restart_context":
            # Can't truly restart without the browser reference, but we can
            # clear state and reload
            try:
                await self.context.clear_cookies()
                page = await self._ensure_page()
                await page.evaluate("window.localStorage.clear(); window.sessionStorage.clear();")
                await page.reload(wait_until="domcontentloaded", timeout=15000)
                self._elements.clear()
                self._prev_element_sigs.clear()
                return "Context reset: cookies cleared, storage cleared, page reloaded."
            except Exception as e:
                return f"Restart context error: {e}"

        return f"Unknown action: {action}"

    async def _handle_save_site_workaround(self, tool_input: dict[str, Any]) -> str:
        domain = tool_input["domain"]
        workaround = tool_input["workaround"]
        try:
            _site_memory.save_workaround(domain, workaround)
            return f"Workaround saved for {domain}. Future agents will auto-apply this."
        except Exception as e:
            return f"Failed to save workaround: {e}"


# ---------------------------------------------------------------------------
# Common block-page patterns for failure detection
# ---------------------------------------------------------------------------

_BLOCK_PATTERNS = [
    "access denied",
    "automated tool",
    "are you a robot",
    "captcha",
    "please verify you are a human",
    "blocked",
    "403 forbidden",
    "bot detected",
    "unusual traffic",
    "rate limit",
    "too many requests",
    "please enable javascript",
    "browser not supported",
    "cloudflare",
    "just a moment",  # Cloudflare waiting page
    "checking your browser",
    "attention required",
    "pardon our interruption",
]

_STEALTH_SCRIPT = """
    Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
    Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3, 4, 5] });
    Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    window.chrome = { runtime: {} };
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) =>
        parameters.name === 'notifications'
            ? Promise.resolve({ state: Notification.permission })
            : originalQuery(parameters);
"""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

async def create_browser_provider(
    browser: Browser | None = None,
    output_dir: Path = Path("."),
    storage_state_path: str | None = None,
    context: Any | None = None,
) -> BrowserToolProvider:
    """Create a BrowserToolProvider.

    Either pass a *browser* (local Chromium — a new context is created) or a
    pre-existing *context* (e.g. from Browserbase CDP connection).
    """
    if context is not None:
        return BrowserToolProvider(context=context, output_dir=output_dir)

    if browser is None:
        raise ValueError("Either browser or context must be provided")

    ctx_kwargs: dict[str, Any] = {
        "viewport": {"width": 1280, "height": 900},
        "accept_downloads": True,
        "user_agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
    }

    # Load full session state (cookies + localStorage) saved from interactive login
    if storage_state_path and Path(storage_state_path).exists():
        ctx_kwargs["storage_state"] = storage_state_path
        logger.info(f"Loading session state from {storage_state_path}")

    ctx = await browser.new_context(**ctx_kwargs)

    # Stealth: mask headless/automation fingerprints
    await ctx.add_init_script(_STEALTH_SCRIPT)

    return BrowserToolProvider(context=ctx, output_dir=output_dir)
