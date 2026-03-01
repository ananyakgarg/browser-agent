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

from tool_registry import ToolProvider

logger = logging.getLogger(__name__)

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
        # Deprecated: scroll, go_back, get_text, select_option, find_element
        # All subsumed by execute_js — these caused degenerate loops
        # (find_element->find_element: 84x, scroll->scroll: 25x in traces)

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
        page = await self._ensure_page()
        try:
            await page.goto(tool_input["url"], wait_until="domcontentloaded", timeout=30000)
            await page.wait_for_timeout(500)
        except Exception as e:
            return f"Navigation error: {e}"
        return (await self.get_page_state()).to_text()

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
    await ctx.add_init_script("""
        // Hide webdriver flag
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });

        // Realistic plugins array
        Object.defineProperty(navigator, 'plugins', {
            get: () => [1, 2, 3, 4, 5],
        });

        // Realistic languages
        Object.defineProperty(navigator, 'languages', {
            get: () => ['en-US', 'en'],
        });

        // Hide automation-related Chrome properties
        window.chrome = { runtime: {} };

        // Patch permissions query
        const originalQuery = window.navigator.permissions.query;
        window.navigator.permissions.query = (parameters) =>
            parameters.name === 'notifications'
                ? Promise.resolve({ state: Notification.permission })
                : originalQuery(parameters);
    """)

    return BrowserToolProvider(context=ctx, output_dir=output_dir)
