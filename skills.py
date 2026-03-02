"""Declarative skill modules for guided tool usage.

Each Skill bundles:
- A tool allowlist (which tools Claude sees)
- A prompt fragment (injected into the system prompt)
- Step-by-step recipes and fallbacks

The SkillRegistry resolves planner-selected skills into concrete tool sets
and prompt additions for the worker agent.

Gated behind the --skills CLI flag — when disabled, nothing changes.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Skill dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Skill:
    name: str
    tool_allowlist: tuple[str, ...]  # frozen ⇒ use tuple, not list
    prompt_fragment: str
    recipes: tuple[str, ...]
    fallbacks: tuple[str, ...]
    version: str = "1.0"
    priority: int = 0  # higher = preferred strategy when skills conflict


# ---------------------------------------------------------------------------
# Concrete skill definitions
# ---------------------------------------------------------------------------

CODE_ANALYSIS_SKILL = Skill(
    name="code_analysis",
    version="1.0",
    priority=5,
    tool_allowlist=(
        "analyze_dependencies",
        "get_function_source",
        "get_recent_changes",
        "find_references",
        "get_file_structure",
        "get_commit_diff",
        "http_request",
        "run_python",
    ),
    prompt_fragment="""\
## Code Analysis Skill
You have code analysis tools. **Always trace dependencies before assessing blame.**

Tool usage guidance:
- `analyze_dependencies` reveals what a function calls and what calls it — use this
  BEFORE deciding whether a change is cosmetic or behavioral.
- `get_function_source` retrieves exact source code — use this to read implementations,
  not just signatures.
- `get_recent_changes` shows git history for a file or function — use after identifying
  relevant files/functions via dependency analysis.
- `get_commit_diff` shows the full diff for a commit — use to classify changes as
  cosmetic (formatting, comments, renames) vs behavioral (logic, control flow, APIs).
- `find_references` locates all call sites — use to assess blast radius of a change.
- `get_file_structure` gives the file tree — use first to orient yourself in a repo.
""",
    recipes=(
        """\
Recipe: Blame / change-impact analysis
1. get_file_structure → understand repo layout
2. get_function_source(target_function) → read the implementation
3. analyze_dependencies(target_function) → find callees and callers
4. For EACH dependency:
   a. get_recent_changes(dep_file) → find relevant commits
   b. get_commit_diff(commit_hash) → classify: cosmetic vs behavioral
5. Synthesize: which changes actually affect the target's behavior?""",
        """\
Recipe: Understanding a function's role
1. get_function_source(function) → read implementation
2. analyze_dependencies(function) → callees (what it uses) + callers (who uses it)
3. find_references(function) → all usage sites for full context""",
    ),
    fallbacks=(
        "If an API tool fails (rate limit, auth, timeout), do NOT retry it.  The same "
        "data is almost always visible in the browser — navigate to the web UI and "
        "extract it from the DOM using execute_js.",
        "If code analysis tools fail, use run_python to parse files locally.",
    ),
)

BROWSER_NAVIGATION_SKILL = Skill(
    name="browser_navigation",
    version="1.0",
    priority=10,
    tool_allowlist=(
        "navigate",
        "click",
        "type_text",
        "select_option",
        "press_key",
        "scroll",
        "hover",
        "screenshot",
        "get_page_state",
        "search_page",
        "execute_js",
        "configure_browser",
        "save_site_workaround",
    ),
    prompt_fragment="""\
## Browser Navigation Skill
Efficiency tips for browser interaction:
- Use `search_page` FIRST when looking for specific text — it searches the full page
  text (including off-screen content) instantly.  Faster than scrolling.
- Use `execute_js` for bulk DOM extraction — e.g. extracting all rows from a table,
  collecting all links, or reading structured data.
- Use `screenshot` to VERIFY that important actions succeeded — it is ground truth.
- Prefer targeted selectors over scrolling: if you know the element structure, use
  execute_js to extract data directly.
- When you can SEE data on a page but don't know how to extract it, **explore the DOM
  first** — use execute_js to inspect the element's parent, siblings, class names, and
  data attributes.  This is the equivalent of right-click → Inspect Element.  Do NOT
  abandon the browser for an API when the data is already rendered in front of you.
""",
    recipes=(
        """\
Recipe: Navigate and extract data from a web page
1. navigate(url) → load the page
2. search_page(query) → locate the target content
3. execute_js(selector) → extract structured data from the DOM
4. screenshot() → verify the extraction visually
5. complete(data) → return results""",
        """\
Recipe: DOM discovery — extracting data you can see but don't know the markup for
1. Find your target element: execute_js to locate it by text content, data attribute,
   or position (e.g. document.querySelector('[data-line-number="123"]'))
2. Inspect its context: read the element's parentElement, className, siblings, and
   nearby data attributes — execute_js('el.parentElement.outerHTML.slice(0, 500)')
3. From the discovered structure, write a targeted extraction query
4. If the target isn't in the DOM (virtual scrolling), scroll to approximate position:
   execute_js('window.scrollTo(0, (targetPos / totalPos) * document.body.scrollHeight)')
   then re-query — the element should now be rendered
5. NEVER abandon the browser for an API call when the data is already on-screen.
   The DOM always has the answer if the page is rendering it visually.""",
    ),
    fallbacks=(
        "If the page blocks browser access, try configure_browser(action='enable_stealth') "
        "then re-navigate.",
        "If still blocked, fall back to http_request for data extraction.",
    ),
)

DATA_EXTRACTION_SKILL = Skill(
    name="data_extraction",
    version="1.0",
    tool_allowlist=(
        "http_request",
        "run_python",
        "execute_js",
        "search_page",
    ),
    prompt_fragment="""\
## Data Extraction Skill
Strategy hierarchy (prefer higher):
1. **API** — use `http_request` to call REST/GraphQL endpoints directly.  Fastest and
   most reliable.
2. **execute_js** — extract structured data from a loaded DOM.  Good when an API isn't
   available but the page is loaded.
3. **search_page** — find specific text on the page.  Good for verification and locating
   content before extraction.
4. **run_python** — parse HTML/JSON, compute values, transform data.  Use as the
   processing layer after fetching raw content.
""",
    recipes=(
        """\
Recipe: API-first data extraction
1. Identify the API endpoint (check network tab patterns, common API paths)
2. http_request(url) → fetch raw JSON/XML
3. run_python → parse and transform the response
4. complete(data) → return results""",
    ),
    fallbacks=(
        "If the API requires authentication, check if the page embeds the data in a "
        "script tag — use execute_js to extract window.__DATA__ or similar globals.",
    ),
)

# ---------------------------------------------------------------------------
# SkillRegistry
# ---------------------------------------------------------------------------

class SkillRegistry:
    """Registry of available skills.  Resolves planner selections into
    concrete tool allowlists and prompt fragments."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def get_all(self) -> list[Skill]:
        return list(self._skills.values())

    def get_names(self) -> list[str]:
        return list(self._skills.keys())

    # -- Resolution ---------------------------------------------------------

    def resolve_skills(self, required: list[dict[str, Any]]) -> list[Skill]:
        """Resolve a list of {name, confidence} dicts into Skill objects.

        Unknown names are silently skipped (forward-compatible).
        """
        resolved: list[Skill] = []
        for entry in required:
            skill = self._skills.get(entry.get("name", ""))
            if skill is not None:
                resolved.append(skill)
        return resolved

    def build_tool_allowlist(self, skills: list[Skill]) -> set[str]:
        """Union of all allowlists + 'complete'.  Empty set when no skills."""
        if not skills:
            return set()
        allowlist: set[str] = {"complete"}
        for s in skills:
            allowlist.update(s.tool_allowlist)
        return allowlist

    _COMPOSITION_PREAMBLE = """\
## Strategy Priority
Multiple skill guides are active below, listed in priority order (highest first).
When they suggest different approaches for the same sub-task:
1. Prefer the higher-priority skill's approach
2. Fall back to lower-priority skills only when the preferred approach fails
3. Never retry a failed approach — move to the next strategy instead"""

    def build_prompt_fragments(self, skills: list[Skill]) -> str:
        """Concatenate prompt fragments + recipes, sorted by priority (desc)."""
        if not skills:
            return ""
        parts: list[str] = []
        if len(skills) > 1:
            parts.append(self._COMPOSITION_PREAMBLE)
        for s in sorted(skills, key=lambda sk: -sk.priority):
            parts.append(s.prompt_fragment)
            for recipe in s.recipes:
                parts.append(recipe)
            if s.fallbacks:
                parts.append("Fallbacks:")
                for fb in s.fallbacks:
                    parts.append(f"- {fb}")
        return "\n\n".join(parts)

    # -- Planner description ------------------------------------------------

    def describe_for_planner(self) -> str:
        """Generate a description of all skills for the PLAN_SYSTEM prompt."""
        lines: list[str] = []
        for s in sorted(self._skills.values(), key=lambda sk: -sk.priority):
            tools_str = ", ".join(s.tool_allowlist)
            lines.append(
                f"- **{s.name}** (v{s.version}, priority {s.priority}): "
                f"Tools: [{tools_str}]. "
                f"{s.prompt_fragment.split(chr(10))[1].strip()}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_DEFAULT_REGISTRY: SkillRegistry | None = None


def get_skill_registry() -> SkillRegistry:
    """Return the singleton SkillRegistry with all built-in skills."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = SkillRegistry()
        _DEFAULT_REGISTRY.register(CODE_ANALYSIS_SKILL)
        _DEFAULT_REGISTRY.register(BROWSER_NAVIGATION_SKILL)
        _DEFAULT_REGISTRY.register(DATA_EXTRACTION_SKILL)
    return _DEFAULT_REGISTRY


# ---------------------------------------------------------------------------
# JSONL logging
# ---------------------------------------------------------------------------

def log_skill_selection(
    output_dir: Path,
    sample_id: str,
    required_skills: list[dict[str, Any]],
    resolved_skills: list[Skill],
) -> None:
    """Append a skill-selection event to skill_log.jsonl."""
    entry = {
        "event": "skill_selection",
        "timestamp": time.time(),
        "sample_id": sample_id,
        "required": required_skills,
        "resolved": [s.name for s in resolved_skills],
    }
    _append_jsonl(output_dir / "skill_log.jsonl", entry)


def log_skill_outcome(
    output_dir: Path,
    sample_id: str,
    skills_used: list[str],
    success: bool,
    tool_calls: list[str] | None = None,
) -> None:
    """Append a skill-outcome event to skill_log.jsonl."""
    entry = {
        "event": "skill_outcome",
        "timestamp": time.time(),
        "sample_id": sample_id,
        "skills_used": skills_used,
        "success": success,
    }
    if tool_calls is not None:
        entry["tool_calls"] = tool_calls
    _append_jsonl(output_dir / "skill_log.jsonl", entry)


def _append_jsonl(path: Path, data: dict[str, Any]) -> None:
    """Append one JSON line to a file (create if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, default=str) + "\n")
