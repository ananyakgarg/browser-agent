"""HTTP request tool provider — direct HTTP calls without browser."""

from __future__ import annotations

import logging
from typing import Any

import aiohttp

from tool_registry import ToolProvider

logger = logging.getLogger(__name__)

_HTTP_REQUEST_SCHEMA = {
    "name": "http_request",
    "description": (
        "Make a direct HTTP request and return the response. "
        "Faster than browser navigation — use for checking URLs (status codes), "
        "calling REST APIs, fetching JSON/XML, or verifying links. "
        "Supports GET, POST, HEAD, PUT, DELETE."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to request.",
            },
            "method": {
                "type": "string",
                "enum": ["GET", "POST", "HEAD", "PUT", "DELETE"],
                "description": "HTTP method. Defaults to GET.",
                "default": "GET",
            },
            "headers": {
                "type": "object",
                "description": "Optional request headers as key-value pairs.",
            },
            "body": {
                "type": "string",
                "description": "Optional request body (for POST/PUT).",
            },
        },
        "required": ["url"],
    },
}

BODY_LIMIT = 8000
HEADER_LIMIT = 1000


class HttpToolProvider(ToolProvider):
    """Makes direct HTTP requests."""

    def __init__(self):
        super().__init__()
        self._session: aiohttp.ClientSession | None = None
        self.register_tool(_HTTP_REQUEST_SCHEMA, self._handle_http_request)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def _handle_http_request(self, tool_input: dict[str, Any]) -> str:
        url = tool_input["url"]
        method = tool_input.get("method", "GET").upper()
        headers = tool_input.get("headers")
        body = tool_input.get("body")

        try:
            session = await self._get_session()
            kwargs: dict[str, Any] = {}
            if headers:
                kwargs["headers"] = headers
            if body and method in ("POST", "PUT"):
                kwargs["data"] = body

            async with session.request(method, url, **kwargs) as resp:
                status = resp.status
                resp_headers = dict(resp.headers)
                resp_body = await resp.text(errors="replace")

                # Format response
                header_str = "\n".join(f"  {k}: {v}" for k, v in resp_headers.items())
                if len(header_str) > HEADER_LIMIT:
                    header_str = header_str[:HEADER_LIMIT] + "\n  ... (truncated)"

                body_str = resp_body[:BODY_LIMIT]
                if len(resp_body) > BODY_LIMIT:
                    body_str += "\n... (truncated)"

                return (
                    f"HTTP {status} {resp.reason}\n\n"
                    f"Headers:\n{header_str}\n\n"
                    f"Body:\n{body_str}"
                )

        except aiohttp.ClientError as e:
            return f"HTTP error: {e}"
        except Exception as e:
            return f"Request error: {e}"

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
