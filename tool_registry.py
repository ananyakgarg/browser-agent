"""Tool provider base class and registry.

To add a new tool category (browser, code execution, API calls, etc.):
1. Subclass ToolProvider
2. In __init__, call self.register_tool(schema, handler) for each tool
3. Register the provider instance with a ToolRegistry

The worker loop calls registry.get_tool_schemas() for the Claude API
and registry.execute() to dispatch tool calls.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Awaitable


# The 'complete' tool is universal — not owned by any provider.
# The worker loop intercepts it before dispatch.
COMPLETE_TOOL_SCHEMA = {
    "name": "complete",
    "description": (
        "Signal that the task is complete and provide the extracted data. "
        "Call this when you have gathered all required information."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "data": {
                "type": "object",
                "description": "The extracted data as key-value pairs matching the required output columns.",
            },
            "notes": {
                "type": "string",
                "description": "Optional notes about the extraction process.",
            },
        },
        "required": ["data"],
    },
}


class ToolProvider:
    """Base class for tool providers.

    Subclass this to create a new tool category. Each tool is a
    (schema, handler) pair registered in __init__:

        class MyProvider(ToolProvider):
            def __init__(self, ...):
                super().__init__()
                self.register_tool(
                    {"name": "my_tool", "description": "...", "input_schema": {...}},
                    self._handle_my_tool,
                )

            async def _handle_my_tool(self, tool_input: dict) -> str:
                ...
    """

    def __init__(self):
        self._tools: dict[str, _RegisteredTool] = {}

    def register_tool(
        self,
        schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[str]],
    ) -> None:
        name = schema["name"]
        if name in self._tools:
            raise ValueError(f"Tool '{name}' already registered in this provider")
        self._tools[name] = _RegisteredTool(schema=schema, handler=handler)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [t.schema for t in self._tools.values()]

    def handles(self, tool_name: str) -> bool:
        return tool_name in self._tools

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        entry = self._tools.get(tool_name)
        if entry is None:
            raise KeyError(f"Tool '{tool_name}' not found in {self.__class__.__name__}")
        return await entry.handler(tool_input)

    async def close(self) -> None:
        """Override to clean up resources (browser contexts, subprocesses, etc.)."""
        pass


class _RegisteredTool:
    __slots__ = ("schema", "handler")

    def __init__(
        self,
        schema: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[str]],
    ):
        self.schema = schema
        self.handler = handler


class ToolRegistry:
    """Aggregates multiple ToolProviders into one dispatch surface.

    Usage:
        registry = ToolRegistry()
        registry.register(BrowserToolProvider(context, output_dir))
        # future: registry.register(CodeExecProvider(...))

        schemas = registry.get_tool_schemas()   # pass to Claude API
        result  = await registry.execute(name, input)  # dispatch
    """

    def __init__(self):
        self._providers: list[ToolProvider] = []

    def register(self, provider: ToolProvider) -> None:
        self._providers.append(provider)

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for p in self._providers:
            schemas.extend(p.get_tool_schemas())
        schemas.append(COMPLETE_TOOL_SCHEMA)
        return schemas

    async def execute(self, tool_name: str, tool_input: dict[str, Any]) -> str:
        for p in self._providers:
            if p.handles(tool_name):
                return await p.execute(tool_name, tool_input)
        if tool_name == "complete":
            return json.dumps(tool_input.get("data", {}))
        return f"Unknown tool: {tool_name}"

    async def close(self) -> None:
        for p in self._providers:
            await p.close()
