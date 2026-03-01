"""Code analysis tool provider — gives agents deep code understanding.

Tools for analyzing function dependencies, extracting source code,
checking git history, and finding references across a repository.
All tools work with GitHub repos via raw file fetching + AST parsing.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from typing import Any
from urllib.parse import quote

import aiohttp

from tool_registry import ToolProvider

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": "BrowserAgent/1.0 (research@example.com)",
    "Accept": "application/json",
}

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

_ANALYZE_DEPS_SCHEMA = {
    "name": "analyze_dependencies",
    "description": (
        "Analyze a Python function's dependencies using AST parsing. "
        "Given a GitHub repo URL, file path, and function name, returns: "
        "functions called, methods invoked, imports used, global variables "
        "referenced, and classes instantiated. Use this to understand what "
        "code a function depends on before checking blame/history."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repo URL (e.g. https://github.com/numpy/numpy)",
            },
            "file_path": {
                "type": "string",
                "description": "Path to file within repo (e.g. numpy/core/function_base.py)",
            },
            "function_name": {
                "type": "string",
                "description": "Name of the function to analyze (e.g. linspace)",
            },
            "branch": {
                "type": "string",
                "description": "Branch or tag (default: main)",
                "default": "main",
            },
        },
        "required": ["repo_url", "file_path", "function_name"],
    },
}

_GET_FUNCTION_SOURCE_SCHEMA = {
    "name": "get_function_source",
    "description": (
        "Extract a specific function's full source code with line numbers. "
        "Returns the function definition, body, decorators, start/end lines. "
        "Use this to read a function before analyzing its blame or changes."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repo URL",
            },
            "file_path": {
                "type": "string",
                "description": "Path to file within repo",
            },
            "function_name": {
                "type": "string",
                "description": "Name of the function to extract",
            },
            "branch": {
                "type": "string",
                "description": "Branch or tag (default: main)",
                "default": "main",
            },
        },
        "required": ["repo_url", "file_path", "function_name"],
    },
}

_GET_RECENT_CHANGES_SCHEMA = {
    "name": "get_recent_changes",
    "description": (
        "Get recent commits that modified a specific file, optionally filtered "
        "to a line range. Returns commit hashes, authors, dates, and messages. "
        "Use this to check if a function or its surrounding code was modified recently."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repo URL",
            },
            "file_path": {
                "type": "string",
                "description": "Path to file within repo",
            },
            "since": {
                "type": "string",
                "description": "ISO date string to filter commits after (e.g. 2025-03-01)",
                "default": "",
            },
            "max_results": {
                "type": "integer",
                "description": "Max commits to return (default: 20)",
                "default": 20,
            },
        },
        "required": ["repo_url", "file_path"],
    },
}

_FIND_REFERENCES_SCHEMA = {
    "name": "find_references",
    "description": (
        "Search a GitHub repo for all references to a function or variable name. "
        "Returns files and line snippets where the name appears. "
        "Use this to find callers, usages, or downstream dependencies."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repo URL",
            },
            "symbol_name": {
                "type": "string",
                "description": "Function, class, or variable name to search for",
            },
            "file_filter": {
                "type": "string",
                "description": "Optional file extension filter (e.g. '.py')",
                "default": ".py",
            },
        },
        "required": ["repo_url", "symbol_name"],
    },
}

_GET_FILE_STRUCTURE_SCHEMA = {
    "name": "get_file_structure",
    "description": (
        "Get the structure of a Python file: all classes, functions, and methods "
        "with their line numbers. Use this to understand a file's layout before "
        "diving into specific functions."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repo URL",
            },
            "file_path": {
                "type": "string",
                "description": "Path to file within repo",
            },
            "branch": {
                "type": "string",
                "description": "Branch or tag (default: main)",
                "default": "main",
            },
        },
        "required": ["repo_url", "file_path"],
    },
}

_GET_COMMIT_DIFF_SCHEMA = {
    "name": "get_commit_diff",
    "description": (
        "Get the diff for a specific commit, optionally filtered to a single file. "
        "Returns the actual code changes (added/removed lines). "
        "Use this to understand what a commit changed and assess material impact."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "repo_url": {
                "type": "string",
                "description": "GitHub repo URL",
            },
            "commit_hash": {
                "type": "string",
                "description": "Full or abbreviated commit hash",
            },
            "file_path": {
                "type": "string",
                "description": "Optional: filter diff to this file only",
                "default": "",
            },
        },
        "required": ["repo_url", "commit_hash"],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_repo(repo_url: str) -> tuple[str, str]:
    """Extract (owner, repo) from a GitHub URL."""
    # Handle https://github.com/owner/repo or github.com/owner/repo
    parts = repo_url.rstrip("/").split("/")
    return parts[-2], parts[-1]


async def _fetch_raw_file(
    session: aiohttp.ClientSession,
    owner: str,
    repo: str,
    file_path: str,
    branch: str = "main",
) -> str | None:
    """Fetch raw file content from GitHub."""
    url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
    try:
        async with session.get(url) as resp:
            if resp.status == 200:
                return await resp.text()
            # Try master if main fails
            if branch == "main":
                url2 = f"https://raw.githubusercontent.com/{owner}/{repo}/master/{file_path}"
                async with session.get(url2) as resp2:
                    if resp2.status == 200:
                        return await resp2.text()
            return None
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None


def _find_function_node(tree: ast.Module, function_name: str) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find a function definition in an AST, including methods inside classes."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == function_name:
                return node
    return None


def _extract_dependencies(node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, list[str]]:
    """Extract all dependencies from a function's AST node."""
    calls: list[str] = []
    attributes: list[str] = []
    names: list[str] = []
    subscripts: list[str] = []

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            func = child.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                # e.g. self.method(), np.array(), obj.func()
                parts = []
                current = func
                while isinstance(current, ast.Attribute):
                    parts.append(current.attr)
                    current = current.value
                if isinstance(current, ast.Name):
                    parts.append(current.id)
                parts.reverse()
                calls.append(".".join(parts))
        elif isinstance(child, ast.Name) and not isinstance(child.ctx, ast.Store):
            names.append(child.id)
        elif isinstance(child, ast.Attribute):
            if isinstance(child.value, ast.Name):
                attributes.append(f"{child.value.id}.{child.attr}")

    # Deduplicate
    return {
        "function_calls": sorted(set(calls)),
        "attribute_accesses": sorted(set(attributes)),
        "names_referenced": sorted(set(names) - set(calls)),
    }


def _get_function_source(source: str, node: ast.FunctionDef | ast.AsyncFunctionDef) -> dict[str, Any]:
    """Extract source code and metadata for a function node."""
    lines = source.splitlines()

    # Include decorators
    start_line = node.lineno
    if node.decorator_list:
        start_line = node.decorator_list[0].lineno

    end_line = node.end_lineno or node.lineno

    # Extract source with line numbers
    func_lines = []
    for i in range(start_line - 1, min(end_line, len(lines))):
        func_lines.append(f"{i + 1:>5} | {lines[i]}")

    return {
        "start_line": start_line,
        "end_line": end_line,
        "line_count": end_line - start_line + 1,
        "source": "\n".join(func_lines),
    }


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

class CodeAnalysisProvider(ToolProvider):
    """Provides code understanding tools for analyzing repos."""

    def __init__(self):
        super().__init__()
        self._session: aiohttp.ClientSession | None = None

        self.register_tool(_ANALYZE_DEPS_SCHEMA, self._handle_analyze_deps)
        self.register_tool(_GET_FUNCTION_SOURCE_SCHEMA, self._handle_get_function_source)
        self.register_tool(_GET_RECENT_CHANGES_SCHEMA, self._handle_get_recent_changes)
        self.register_tool(_FIND_REFERENCES_SCHEMA, self._handle_find_references)
        self.register_tool(_GET_FILE_STRUCTURE_SCHEMA, self._handle_get_file_structure)
        self.register_tool(_GET_COMMIT_DIFF_SCHEMA, self._handle_get_commit_diff)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers=DEFAULT_HEADERS,
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # --- analyze_dependencies ---

    async def _handle_analyze_deps(self, tool_input: dict[str, Any]) -> str:
        owner, repo = _parse_repo(tool_input["repo_url"])
        file_path = tool_input["file_path"]
        function_name = tool_input["function_name"]
        branch = tool_input.get("branch", "main")

        session = await self._get_session()
        source = await _fetch_raw_file(session, owner, repo, file_path, branch)
        if source is None:
            return f"Could not fetch {file_path} from {owner}/{repo} ({branch})"

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error parsing {file_path}: {e}"

        node = _find_function_node(tree, function_name)
        if node is None:
            return f"Function '{function_name}' not found in {file_path}"

        deps = _extract_dependencies(node)

        # Also extract file-level imports
        imports = []
        for imp in ast.walk(tree):
            if isinstance(imp, ast.Import):
                for alias in imp.names:
                    imports.append(alias.name)
            elif isinstance(imp, ast.ImportFrom):
                module = imp.module or ""
                for alias in imp.names:
                    imports.append(f"{module}.{alias.name}")

        result = {
            "function": function_name,
            "file": file_path,
            "line_range": f"{node.lineno}-{node.end_lineno}",
            "dependencies": deps,
            "file_imports": sorted(set(imports)),
        }

        return json.dumps(result, indent=2)

    # --- get_function_source ---

    async def _handle_get_function_source(self, tool_input: dict[str, Any]) -> str:
        owner, repo = _parse_repo(tool_input["repo_url"])
        file_path = tool_input["file_path"]
        function_name = tool_input["function_name"]
        branch = tool_input.get("branch", "main")

        session = await self._get_session()
        source = await _fetch_raw_file(session, owner, repo, file_path, branch)
        if source is None:
            return f"Could not fetch {file_path} from {owner}/{repo} ({branch})"

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error parsing {file_path}: {e}"

        node = _find_function_node(tree, function_name)
        if node is None:
            # List available functions as hint
            available = []
            for n in ast.walk(tree):
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    available.append(f"  {n.name} (line {n.lineno})")
            hint = "\n".join(available[:20]) if available else "(none found)"
            return f"Function '{function_name}' not found in {file_path}.\n\nAvailable functions:\n{hint}"

        info = _get_function_source(source, node)

        return (
            f"Function: {function_name}\n"
            f"File: {file_path}\n"
            f"Lines: {info['start_line']}-{info['end_line']} ({info['line_count']} lines)\n\n"
            f"{info['source']}"
        )

    # --- get_recent_changes ---

    async def _handle_get_recent_changes(self, tool_input: dict[str, Any]) -> str:
        owner, repo = _parse_repo(tool_input["repo_url"])
        file_path = tool_input["file_path"]
        since = tool_input.get("since", "")
        max_results = tool_input.get("max_results", 20)

        session = await self._get_session()
        url = f"https://api.github.com/repos/{owner}/{repo}/commits?path={quote(file_path)}&per_page={max_results}"
        if since:
            url += f"&since={since}T00:00:00Z"

        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return f"GitHub API error {resp.status}: {body[:500]}"
                commits = await resp.json()
        except Exception as e:
            return f"Error fetching commits: {e}"

        if not commits:
            since_msg = f" since {since}" if since else ""
            return f"No commits found for {file_path}{since_msg}"

        lines = [f"Recent commits for {file_path}" + (f" (since {since})" if since else "") + ":\n"]
        for c in commits[:max_results]:
            sha = c["sha"][:10]
            author = c.get("commit", {}).get("author", {})
            name = author.get("name", "unknown")
            date = author.get("date", "")[:10]
            message = c.get("commit", {}).get("message", "").split("\n")[0][:100]
            lines.append(f"  {sha}  {date}  {name:20s}  {message}")

        return "\n".join(lines)

    # --- find_references ---

    async def _handle_find_references(self, tool_input: dict[str, Any]) -> str:
        owner, repo = _parse_repo(tool_input["repo_url"])
        symbol_name = tool_input["symbol_name"]
        file_filter = tool_input.get("file_filter", ".py")

        session = await self._get_session()

        # Use GitHub code search API
        query = quote(f"{symbol_name} repo:{owner}/{repo}")
        url = f"https://api.github.com/search/code?q={query}&per_page=20"

        try:
            headers = {"Accept": "application/vnd.github.v3.text-match+json"}
            async with session.get(url, headers=headers) as resp:
                if resp.status == 403:
                    return (
                        "GitHub code search API rate limited. "
                        "Try using search_page on the GitHub website instead, or "
                        "use http_request to fetch specific files and run_python to grep them."
                    )
                if resp.status != 200:
                    body = await resp.text()
                    return f"GitHub API error {resp.status}: {body[:500]}"
                data = await resp.json()
        except Exception as e:
            return f"Error searching: {e}"

        items = data.get("items", [])
        if not items:
            return f"No references to '{symbol_name}' found in {owner}/{repo}"

        # Filter by extension
        if file_filter:
            items = [i for i in items if i["name"].endswith(file_filter)]

        lines = [f"References to '{symbol_name}' in {owner}/{repo} ({len(items)} files):\n"]
        for item in items[:20]:
            path = item["path"]
            # Include text match snippets if available
            matches = item.get("text_matches", [])
            if matches:
                for m in matches[:2]:
                    fragment = m.get("fragment", "").strip().replace("\n", " ")[:120]
                    lines.append(f"  {path}: ...{fragment}...")
            else:
                lines.append(f"  {path}")

        return "\n".join(lines)

    # --- get_file_structure ---

    async def _handle_get_file_structure(self, tool_input: dict[str, Any]) -> str:
        owner, repo = _parse_repo(tool_input["repo_url"])
        file_path = tool_input["file_path"]
        branch = tool_input.get("branch", "main")

        session = await self._get_session()
        source = await _fetch_raw_file(session, owner, repo, file_path, branch)
        if source is None:
            return f"Could not fetch {file_path} from {owner}/{repo} ({branch})"

        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            return f"Syntax error parsing {file_path}: {e}"

        lines = [f"Structure of {file_path}:\n"]

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                end = node.end_lineno or node.lineno
                lines.append(f"  class {node.name} (lines {node.lineno}-{end})")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        iend = item.end_lineno or item.lineno
                        prefix = "async " if isinstance(item, ast.AsyncFunctionDef) else ""
                        lines.append(f"    {prefix}def {item.name}() (lines {item.lineno}-{iend})")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                end = node.end_lineno or node.lineno
                prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
                lines.append(f"  {prefix}def {node.name}() (lines {node.lineno}-{end})")

        # Also show imports summary
        import_count = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))
        total_lines = len(source.splitlines())
        lines.append(f"\n  Total: {total_lines} lines, {import_count} imports")

        return "\n".join(lines)

    # --- get_commit_diff ---

    async def _handle_get_commit_diff(self, tool_input: dict[str, Any]) -> str:
        owner, repo = _parse_repo(tool_input["repo_url"])
        commit_hash = tool_input["commit_hash"]
        file_path = tool_input.get("file_path", "")

        session = await self._get_session()

        # Fetch the commit diff
        url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_hash}"
        try:
            async with session.get(url) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    return f"GitHub API error {resp.status}: {body[:500]}"
                data = await resp.json()
        except Exception as e:
            return f"Error fetching commit: {e}"

        # Header
        author = data.get("commit", {}).get("author", {})
        message = data.get("commit", {}).get("message", "")
        lines = [
            f"Commit: {commit_hash[:10]}",
            f"Author: {author.get('name', 'unknown')}",
            f"Date: {author.get('date', '')[:10]}",
            f"Message: {message.split(chr(10))[0]}",
            "",
        ]

        # Stats
        stats = data.get("stats", {})
        lines.append(f"Changes: +{stats.get('additions', 0)} -{stats.get('deletions', 0)}")

        # File diffs
        files = data.get("files", [])
        if file_path:
            files = [f for f in files if f["filename"] == file_path]
            if not files:
                return f"File {file_path} not modified in commit {commit_hash[:10]}"

        for f in files[:5]:  # Limit to 5 files
            fname = f["filename"]
            patch = f.get("patch", "(binary or too large)")
            lines.append(f"\n--- {fname} (+{f.get('additions', 0)} -{f.get('deletions', 0)}) ---")
            # Truncate large patches
            if len(patch) > 3000:
                patch = patch[:3000] + "\n... (truncated)"
            lines.append(patch)

        if len(data.get("files", [])) > 5:
            lines.append(f"\n... and {len(data['files']) - 5} more files")

        return "\n".join(lines)
