"""Site memory: persists domain-specific workarounds learned by agents.

When an agent discovers a workaround for a site (e.g., needs stealth mode,
requires specific headers, blocks headless browsers), it saves it here.
Future agents check this before navigating to any domain and auto-apply
known workarounds.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_MEMORY_PATH = Path("site_memory.json")


class SiteMemory:
    """Persistent store of domain-specific workarounds."""

    def __init__(self, path: Path = DEFAULT_MEMORY_PATH):
        self.path = path
        self._data: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self):
        if self.path.exists():
            try:
                self._data = json.loads(self.path.read_text())
            except Exception as e:
                logger.warning(f"Failed to load site memory: {e}")
                self._data = {}

    def _save(self):
        self.path.write_text(json.dumps(self._data, indent=2))

    def get(self, domain: str) -> dict[str, Any] | None:
        """Get workarounds for a domain. Returns None if no memory exists."""
        return self._data.get(domain)

    def save_workaround(self, domain: str, workaround: dict[str, Any]):
        """Save a workaround for a domain. Merges with existing."""
        if domain not in self._data:
            self._data[domain] = {}
        self._data[domain].update(workaround)
        self._save()
        logger.info(f"Site memory: saved workaround for {domain}")

    def get_all(self) -> dict[str, dict[str, Any]]:
        """Get all known domain workarounds."""
        return dict(self._data)

    def get_hint_for_url(self, url: str) -> str | None:
        """Extract domain from URL and return a human-readable hint if we have memory."""
        try:
            from urllib.parse import urlparse
            domain = urlparse(url).netloc
            # Check exact domain and parent domain
            for d in [domain, ".".join(domain.split(".")[-2:])]:
                mem = self.get(d)
                if mem:
                    parts = [f"Known issue with {d}:"]
                    if mem.get("blocks_headless"):
                        parts.append("- This site blocks headless browsers.")
                    if mem.get("needs_stealth"):
                        parts.append("- Stealth mode required (already enabled).")
                    if mem.get("needs_headers"):
                        parts.append(f"- Requires headers: {mem['needs_headers']}")
                    if mem.get("rate_limited"):
                        parts.append("- Rate limiting detected. Go slowly.")
                    if mem.get("needs_auth"):
                        parts.append("- Authentication required.")
                    if mem.get("workaround"):
                        parts.append(f"- Workaround: {mem['workaround']}")
                    return "\n".join(parts)
        except Exception:
            pass
        return None
