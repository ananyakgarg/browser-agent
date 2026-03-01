"""Code execution tool provider — runs Python in a subprocess."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from tool_registry import ToolProvider

logger = logging.getLogger(__name__)

_RUN_PYTHON_SCHEMA = {
    "name": "run_python",
    "description": (
        "Execute Python code and return stdout/stderr. "
        "The code runs in a fresh subprocess with a 30-second timeout. "
        "Use for data processing, parsing, calculations, string manipulation, "
        "JSON/HTML parsing, validation, or any computation. "
        "Print results to stdout to see them."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Python code to execute.",
            },
        },
        "required": ["code"],
    },
}

OUTPUT_LIMIT = 8000


class CodeToolProvider(ToolProvider):
    """Runs Python code in a subprocess."""

    def __init__(self):
        super().__init__()
        self.register_tool(_RUN_PYTHON_SCHEMA, self._handle_run_python)

    async def _handle_run_python(self, tool_input: dict[str, Any]) -> str:
        code = tool_input["code"]
        try:
            proc = await asyncio.create_subprocess_exec(
                "python3", "-c", code,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                return "Error: code execution timed out after 30 seconds"

            out = stdout.decode(errors="replace")
            err = stderr.decode(errors="replace")

            parts = []
            if out:
                parts.append(out[:OUTPUT_LIMIT])
            if err:
                parts.append(f"STDERR:\n{err[:OUTPUT_LIMIT]}")
            if not parts:
                return "(no output)"

            result = "\n".join(parts)
            if len(result) > OUTPUT_LIMIT:
                result = result[:OUTPUT_LIMIT] + "\n... (truncated)"
            return result

        except Exception as e:
            return f"Code execution error: {e}"
